"""
Enriches the papers people care about, slowly, from arXiv: their references (for the Cited tab
and the reference graph, aslite/refs.py) and two thumbnails (aslite/thumbs.py). Off unless
ENRICH_ENABLED=1 (docker/update.sh).

Per paper, up to four requests to arxiv.org:
0. its abstract page, for the license, if the snapshot doesn't have it (newer papers)
1. its HTML version (LaTeXML): the references, and where its first real figure is
2. that figure, made into the 'figure' thumbnail
3. its PDF, rendered into the 'page' thumbnail; the references come from it (numbered lists
   only) when there is no HTML version or its bibliography didn't convert
Thumbnails only for papers under an open license (aslite/thumbs.py says why): for the others,
2 is skipped, and 3 too unless the references need it.
A paper is done ('ok') with whatever of that worked; it isn't asked for again. A network error
or a broken answer marks it 'error': tried again after a day, at most 3 times.

Order: papers someone saved (the Cited tab is made of them), newest first, then the papers in
everyone's latest digest. One request per --pause seconds (ENRICH_PAUSE, at least 15, as
arxiv.org's robots.txt asks), so at most ~1400 papers a day, in practice the few people save.
On 429/503 it waits as long as arXiv says. Its own pause is kept outside db.arxiv_slot's lock,
so the API and a user's /import never wait behind it. Once a day, references that matched
nothing are matched again (papers added since, or a titles table built since).

    ENRICH_ENABLED=1 docker compose up -d   # in docker: started by docker/update.sh
    uv run python enrich_daemon.py --once   # what is queued, then exit
"""

import os
import time
import argparse
import urllib.error
import urllib.request

from aslite import db, thumbs
from aslite.db import get_papers_db, get_refs_db
from aslite.refs import Matcher, parse, parse_pdf, rows_for

PAUSE = 15  # seconds between requests: arxiv.org's robots.txt asks for Crawl-delay 15
USER_AGENT = 'arxiv-resanity (https://github.com/dorian-K/arxiv-sanity-lite)'
REFRESH = 600  # seconds between looking for new saved papers when nothing is queued
RETRY = 86400  # a failed paper is tried again after this long...
MAX_TRIES = 3  # ...this many times
MAX_HTML = 20_000_000
MAX_IMAGE = 15_000_000
MAX_PDF = 60_000_000  # bigger PDFs (some are 50 MB of figures) get no page thumbnail


class SlowDown(Exception):
    """arXiv asked us to wait (429/503)"""

    def __init__(self, retry_after):
        super().__init__(f"arXiv asked to wait {retry_after}s")
        self.retry_after = retry_after


class TooBig(Exception):
    pass


class Fetcher:
    """GETs from arxiv.org, one per `pause` seconds, inside the slot all arXiv calls share"""

    def __init__(self, pause):
        self.pause, self.last = pause, 0.0

    def get(self, url, max_bytes, timeout=90):
        """(final url, body), or (None, b'') if arXiv has no such page (404/410)"""
        time.sleep(max(0.0, self.last + self.pause - time.time()))
        with db.arxiv_slot():
            self.last = time.time()
            req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    body = r.read(max_bytes + 1)
                    final = r.geturl()
            except urllib.error.HTTPError as e:
                if e.code in (429, 503):
                    try:
                        wait = int(e.headers.get('Retry-After') or 600)
                    except ValueError:
                        wait = 600
                    raise SlowDown(max(60, wait)) from e
                if e.code in (404, 410):
                    return None, b''
                raise
        if len(body) > max_bytes:
            raise TooBig(url)
        return final, body


def paper_license(pid, fetch, rdb):
    """the paper's license code: from the snapshot's table, else from its abstract page"""
    code = rdb.license_of(pid)
    if code is None:
        url, body = fetch.get(f'https://arxiv.org/abs/{pid}', MAX_HTML)
        code = thumbs.abs_license(body.decode('utf-8', 'replace')) if url else ''
        rdb.put_licenses([(pid, code)])
    return code


def process(pid, fetch, matcher, license):
    """
    (refs rows, where they came from, figure made, page made) for one paper; see the docstring.
    Thumbnails only under an open license (thumbs.is_open): otherwise neither the figure nor the
    PDF is downloaded for them, and the PDF only if the references have to come from it
    """
    rows, refs_from, figure, page = [], '', False, False
    may_show = thumbs.is_open(license)
    url, body = fetch.get(f'https://arxiv.org/html/{pid}', MAX_HTML)
    html = body.decode('utf-8', 'replace') if url and '/html/' in url else ''
    if 'ltx_page_main' not in html and 'ltx_document' not in html:
        html = ''  # no HTML version: arXiv answers with a notice page (or 404)
    if html:
        parsed = parse(html)
        if parsed:
            rows, refs_from = rows_for(parsed, matcher), 'html'
        for img_url in thumbs.figure_urls(html, url)[:2] if may_show else []:
            img_final, img = fetch.get(img_url, MAX_IMAGE)
            if not img_final:
                continue
            try:
                db.save_thumb(pid, 'figure', thumbs.from_image(img))
                figure = True
                break
            except Exception as e:  # e.g. an svg, or a broken file: the next figure, or none
                print(f"  {pid}: figure {img_url}: {e}", flush=True)
    if not may_show and refs_from:
        return rows, refs_from, figure, page
    try:
        pdf_url, pdf = fetch.get(f'https://arxiv.org/pdf/{pid}', MAX_PDF)
    except TooBig:
        pdf_url, pdf = None, b''
        print(f"  {pid}: PDF over {MAX_PDF // 1_000_000} MB, skipped", flush=True)
    if pdf_url and pdf.startswith(b'%PDF'):
        if may_show:
            try:
                db.save_thumb(pid, 'page', thumbs.from_pdf(pdf))
                page = True
            except Exception as e:
                print(f"  {pid}: rendering the PDF: {e}", flush=True)
        if not refs_from:  # text and data mining (§44b UrhG): the PDF isn't kept
            try:
                parsed = parse_pdf(thumbs.pdf_text(pdf))
            except Exception as e:
                print(f"  {pid}: reading the PDF: {e}", flush=True)
                parsed = []
            if parsed:
                rows, refs_from = rows_for(parsed, matcher), 'pdf'
    return rows, refs_from, figure, page


def queue(pdb, rdb, now=None):
    """the papers to process, best first"""
    now = now or time.time()
    times = pdb.times()
    done = rdb.fetched()

    def due(p):
        if p not in times:
            return False
        st = done.get(p)
        return st is None or (st[0] == 'error' and st[2] < MAX_TRIES and now - st[1] > RETRY)

    saved = set()
    for tags in db.get_all_tags().values():
        saved = saved.union(*tags.values())
    digest = db.all_digest_pids() - saved
    newest = lambda ps: sorted(ps, key=lambda p: -times.get(p, 0))
    return [p for p in newest(saved) + newest(digest) if due(p)]


def rematch(rdb, matcher):
    """matches the references that matched nothing so far again; returns how many matched now"""
    found = []
    for citing, pos, title, raw in rdb.unmatched():
        cited, _ = matcher.match({'title': title, 'raw': raw, 'links': [], 'blocks': []})
        if cited:
            found.append((cited, citing, pos))
    rdb.set_cited(found)
    return len(found)


def main():
    parser = argparse.ArgumentParser(description='references and thumbnails of the papers people care about')
    parser.add_argument(
        '--pause',
        type=float,
        default=float(os.environ.get('ENRICH_PAUSE') or 20),
        help=f'seconds between requests to arxiv.org, at least {PAUSE} (default $ENRICH_PAUSE or 20)',
    )
    parser.add_argument('--once', action='store_true', help='exit when nothing is queued')
    parser.add_argument('--max', type=int, default=0, help='stop after this many papers (0: never)')
    args = parser.parse_args()
    fetch = Fetcher(max(PAUSE, args.pause))

    with get_papers_db() as pdb, get_refs_db() as rdb:
        matcher = Matcher(pdb, rdb)
        last_rematch, total = 0.0, 0
        while True:
            if time.time() - last_rematch > 86400:
                print(f"matched {rematch(rdb, matcher)} earlier references now", flush=True)
                last_rematch = time.time()
            todo = queue(pdb, rdb)
            if not todo:
                if args.once:
                    break
                time.sleep(REFRESH)
                continue
            print(f"{len(todo)} papers to enrich", flush=True)
            t0, n, n_refs, n_matched, n_thumbs = time.time(), 0, 0, 0, 0
            for pid in todo[:10]:  # then look at the queue again: newly saved papers go first
                try:
                    license = paper_license(pid, fetch, rdb)
                    rows, refs_from, figure, page = process(pid, fetch, matcher, license)
                except SlowDown as e:
                    print(f"{e}, waiting", flush=True)
                    time.sleep(e.retry_after)
                    break
                except Exception as e:  # network trouble, a broken page: tried again after RETRY
                    print(f"  {pid}: {type(e).__name__}: {e}", flush=True)
                    rdb.put_refs(pid, 'error')
                    continue
                rdb.put_refs(pid, 'ok', rows, refs_from, figure, page, license)
                n += 1
                n_refs += len(rows)
                n_matched += sum(1 for r in rows if r[1])
                n_thumbs += figure + page
                print(
                    f"  {pid}: {len(rows)} references ({refs_from or 'none'}), "
                    f"{sum(1 for r in rows if r[1])} matched; thumbnails: "
                    f"{', '.join(k for k, ok in (('figure', figure), ('page', page)) if ok) or 'none'}",
                    flush=True,
                )
            total += n
            if n:
                db.put_job(
                    'enrich',
                    time.time() - t0,
                    papers=n,
                    refs=n_refs,
                    matched=n_matched,
                    thumbs=n_thumbs,
                    left=len(todo) - n,
                )
            if args.max and total >= args.max:
                break


if __name__ == '__main__':
    main()
