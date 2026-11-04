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
With SUMMARIES_ENABLED=1 (and OPENROUTER_API_KEY), the text of 1 (else 3) also goes to a model
for a short summary (aslite/summary.py): no extra request for new papers; papers enriched before
are fetched once more (HTML, else PDF). At most SUMMARY_MAX_DAY summaries a day.
A paper is done ('ok') with whatever of that worked; it isn't asked for again (except, once,
the papers whose thumbnails were made before there were big ones). A network error
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
import json
import time
import argparse
import urllib.error
import urllib.request

from aslite import db, summary, thumbs
from aslite.db import get_papers_db, get_refs_db
from aslite.refs import Matcher, parse, parse_pdf, rows_for

PAUSE = 15  # seconds between requests: arxiv.org's robots.txt asks for Crawl-delay 15
USER_AGENT = 'arxiv-resanity (https://github.com/dorian-K/arxiv-sanity-lite)'
REFRESH = 600  # seconds between looking for new saved papers when nothing is queued
ASK_POLL = 5  # ...but papers asked for on the graph page are looked for this often
RETRY = 86400  # a failed paper is tried again after this long...
MAX_TRIES = 3  # ...this many times
MAX_HTML = 20_000_000
MAX_IMAGE = 15_000_000
MAX_PDF = 60_000_000  # bigger PDFs (some are 50 MB of figures) get no page thumbnail
SUMMARY_MAX_DAY = int(os.environ.get('SUMMARY_MAX_DAY') or 300)  # summaries a day at most (~$0.001 each)


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


def save_thumbs(pid, kind, big):
    """the card's thumbnail and the enlarged one, made from the big one"""
    card, big = thumbs.both(big)
    db.save_thumb(pid, kind, big, big=True)
    db.save_thumb(pid, kind, card)


def process(pid, fetch, matcher, license):
    """
    (refs rows, where they came from, figure made, page made, (text source, text)) for one paper;
    see the docstring. Thumbnails only under an open license (thumbs.is_open): otherwise neither
    the figure nor the PDF is downloaded for them, and the PDF only if the references have to
    come from it. The text (for the summary) is the HTML version's, else the PDF's if it was read
    """
    rows, refs_from, figure, page, text = [], '', False, False, ('', '')
    may_show = thumbs.is_open(license)
    url, body = fetch.get(f'https://arxiv.org/html/{pid}', MAX_HTML)
    html = body.decode('utf-8', 'replace') if url and '/html/' in url else ''
    if 'ltx_page_main' not in html and 'ltx_document' not in html:
        html = ''  # no HTML version: arXiv answers with a notice page (or 404)
    if html:
        text = ('html', summary.html_text(html))
        parsed = parse(html)
        if parsed:
            rows, refs_from = rows_for(parsed, matcher), 'html'
        for img_url in thumbs.figure_urls(html, url)[:2] if may_show else []:
            img_final, img = fetch.get(img_url, MAX_IMAGE)
            if not img_final:
                continue
            try:
                save_thumbs(pid, 'figure', thumbs.from_image(img, thumbs.BIG))
                figure = True
                break
            except Exception as e:  # e.g. an svg, or a broken file: the next figure, or none
                print(f"  {pid}: figure {img_url}: {e}", flush=True)
    if not may_show and refs_from:
        return rows, refs_from, figure, page, text
    try:
        pdf_url, pdf = fetch.get(f'https://arxiv.org/pdf/{pid}', MAX_PDF)
    except TooBig:
        pdf_url, pdf = None, b''
        print(f"  {pid}: PDF over {MAX_PDF // 1_000_000} MB, skipped", flush=True)
    if pdf_url and pdf.startswith(b'%PDF'):
        if may_show:
            try:
                save_thumbs(pid, 'page', thumbs.from_pdf(pdf, thumbs.BIG))
                page = True
            except Exception as e:
                print(f"  {pid}: rendering the PDF: {e}", flush=True)
        if not refs_from or not text[1]:  # text and data mining (§44b UrhG): the PDF isn't kept
            try:
                pdf_text = thumbs.pdf_text(pdf)
                parsed = parse_pdf(pdf_text) if not refs_from else []
            except Exception as e:
                print(f"  {pid}: reading the PDF: {e}", flush=True)
                pdf_text, parsed = '', []
            if not text[1] and pdf_text:
                text = ('pdf', summary.pdf_text(pdf_text))
            if parsed:
                rows, refs_from = rows_for(parsed, matcher), 'pdf'
    return rows, refs_from, figure, page, text


def requested_due(pdb, rdb, now=None):
    """
    the papers someone asked to analyze on the graph page (serve.py, /graph/analyze), oldest first:
    someone waits for them, so they go before everything else. requests that are done, failed for
    good or for a paper that never arrived are dropped
    """
    now = now or time.time()
    asked = rdb.requested()
    if not asked:
        return []
    have = pdb.get_many(asked)
    status = rdb.status_of(asked)
    out = []
    for p in asked:
        st = status.get(p)
        if p in have and (st is None or (st[0] == 'error' and st[2] < MAX_TRIES)):
            out.append(p)
        elif st is not None or p not in have:
            rdb.unrequest(p)
    return out


def queue(pdb, rdb, now=None):
    """the papers to process, best first"""
    now = now or time.time()
    first = requested_due(pdb, rdb, now)
    times = pdb.times()
    done = rdb.fetched()
    redo = rdb.without_big_thumbs()  # made again, once: they predate the enlarged view

    def due(p):
        if p not in times:
            return False
        st = done.get(p)
        return st is None or p in redo or (st[0] == 'error' and st[2] < MAX_TRIES and now - st[1] > RETRY)

    return first + [p for p in wanted(pdb, times) if due(p) and p not in first]


def wanted(pdb, times=None):
    """
    the papers people care about, best first: those of digests about to be mailed (send_emails.py
    prepares them hours ahead, so they have their summaries), then saved ones, newest first, then
    the other digest ones
    """
    times = times or pdb.times()
    soon = db.pending_digest_pids()
    saved = set()
    for tags in db.get_all_tags().values():
        saved = saved.union(*tags.values())
    saved -= soon
    digest = db.all_digest_pids() - saved - soon
    newest = lambda ps: sorted((p for p in ps if p in times), key=lambda p: -times[p])
    return newest(soon) + newest(saved) + newest(digest)


def needs_summary(st, now=None):
    """whether a paper with this summaries row (or None) should get a (full-text) summary now"""
    now = now or time.time()
    if st is None:
        return True
    status, t, tries, source = st
    if status == 'ok':
        return source == 'abstract'  # made from the abstract for a page or a mail: now from the text
    if status == 'pending':
        return now - t > 600
    return tries < MAX_TRIES and now - t > RETRY


def summary_queue(pdb, rdb, now=None):
    """the enriched papers still without a summary, best first (the text has to be fetched again)"""
    now = now or time.time()
    done = rdb.summarized()
    fetched = rdb.fetched()
    return [p for p in wanted(pdb) if fetched.get(p, ('',))[0] == 'ok' and needs_summary(done.get(p), now)]


def fetch_text(pid, fetch):
    """(source, text) of a paper for its summary: its HTML version, else its PDF, else ('', '')"""
    url, body = fetch.get(f'https://arxiv.org/html/{pid}', MAX_HTML)
    html = body.decode('utf-8', 'replace') if url and '/html/' in url else ''
    if 'ltx_page_main' in html or 'ltx_document' in html:
        text = summary.html_text(html)
        if text:
            return 'html', text
    try:
        url, pdf = fetch.get(f'https://arxiv.org/pdf/{pid}', MAX_PDF)
    except TooBig:
        return '', ''
    if url and pdf.startswith(b'%PDF'):
        try:
            return 'pdf', summary.pdf_text(thumbs.pdf_text(pdf))
        except Exception as e:
            print(f"  {pid}: reading the PDF: {e}", flush=True)
    return '', ''


def summarize(pid, pdb, rdb, text):
    """makes and stores the summary of a paper from (source, text); without text, from its abstract"""
    d = pdb.get_many([pid]).get(pid)
    if d is None:
        return False
    source, body = text if text and text[1] else ('notext', '')  # not 'abstract': trying again won't help
    try:
        fields = summary.summarize(' '.join(d['title'].split()), ' '.join(d['summary'].split()), body)
    except Exception as e:  # the model or the network: tried again after RETRY
        print(f"  {pid}: summary: {type(e).__name__}: {e}", flush=True)
        rdb.put_summary(pid, 'error')
        return False
    rdb.put_summary(pid, 'ok', summary.MODEL, source, fields)
    return True


def summarize_some(todo, fetch, pdb, rdb):
    """summaries of papers enriched before (their text is fetched again); returns how many were tried"""
    t0, n, ok = time.time(), 0, 0
    for pid in todo:
        if not summaries_left(rdb):
            break
        try:
            text = fetch_text(pid, fetch)
        except SlowDown as e:
            print(f"{e}, waiting", flush=True)
            time.sleep(e.retry_after)
            break
        except Exception as e:  # network trouble: summarized from the abstract would be worse, so later
            print(f"  {pid}: {type(e).__name__}: {e}", flush=True)
            rdb.put_summary(pid, 'error')
            n += 1
            continue
        made = summarize(pid, pdb, rdb, text)
        ok, n = ok + made, n + 1
        print(f"  {pid}: summary {'made' if made else 'failed'} ({text[0] or 'abstract'})", flush=True)
    if n:
        db.put_job('summaries', time.time() - t0, papers=n, made=ok)
    return n


def summaries_left(rdb):
    """how many more summaries today allows (SUMMARY_MAX_DAY, a cap on the cost)"""
    return max(0, SUMMARY_MAX_DAY - rdb.summaries_since(time.time() - 86400))


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
    parser.add_argument(
        '--try-summary', metavar='PID', help="print a paper's summary (needs OPENROUTER_API_KEY) and store nothing"
    )
    args = parser.parse_args()
    fetch = Fetcher(max(PAUSE, args.pause))
    if args.try_summary:
        with get_papers_db() as pdb:
            d = pdb.get_many([args.try_summary]).get(args.try_summary)
            if d is None:
                raise SystemExit(f"{args.try_summary} isn't in papers.sqlite")
            source, text = fetch_text(args.try_summary, fetch)
            print(f"{summary.MODEL}, from {source or 'the abstract'} ({len(text)} characters):", flush=True)
            fields = summary.summarize(' '.join(d['title'].split()), ' '.join(d['summary'].split()), text)
            print(json.dumps(fields, indent=2, ensure_ascii=False))
        return
    summaries = summary.enabled()
    print(f"summaries {'on, ' + summary.MODEL if summaries else 'off'}", flush=True)

    with get_papers_db() as pdb, get_refs_db() as rdb:
        matcher = Matcher(pdb, rdb)
        last_rematch, total = 0.0, 0
        while True:
            if time.time() - last_rematch > 86400:
                print(f"matched {rematch(rdb, matcher)} earlier references now", flush=True)
                last_rematch = time.time()
            todo = queue(pdb, rdb)
            # (not while someone waits for a paper they asked for on the graph page)
            if summaries and summaries_left(rdb) and not requested_due(pdb, rdb):
                # papers of a digest about to be mailed first, then the rest when nothing else is queued
                soon = db.pending_digest_pids()
                sq = [p for p in summary_queue(pdb, rdb) if p in soon or not todo]
                n = summarize_some(sq[:10], fetch, pdb, rdb)
                total += n
                if n:
                    continue
            if not todo:
                if args.once:
                    break
                # someone may ask for a paper on the graph page meanwhile: look every few seconds
                for _ in range(REFRESH // ASK_POLL):
                    time.sleep(ASK_POLL)
                    if requested_due(pdb, rdb):
                        break
                continue
            print(f"{len(todo)} papers to enrich", flush=True)
            t0, n, n_refs, n_matched, n_thumbs, n_summaries = time.time(), 0, 0, 0, 0, 0
            batch = todo[:10]  # then look at the queue again: newly saved papers go first
            for pid in batch:
                if args.max and total + n >= args.max:
                    break
                # a paper asked for on the graph page meanwhile: it goes before the rest of this batch
                if any(p not in batch for p in requested_due(pdb, rdb)):
                    break
                try:
                    license = paper_license(pid, fetch, rdb)
                    rows, refs_from, figure, page, text = process(pid, fetch, matcher, license)
                except SlowDown as e:
                    print(f"{e}, waiting", flush=True)
                    time.sleep(e.retry_after)
                    break
                except Exception as e:  # network trouble, a broken page: tried again after RETRY
                    print(f"  {pid}: {type(e).__name__}: {e}", flush=True)
                    rdb.put_refs(pid, 'error')
                    rdb.unrequest(pid)
                    continue
                rdb.put_refs(pid, 'ok', rows, refs_from, figure, page, license)
                rdb.unrequest(pid)
                if summaries and needs_summary(rdb.summarized().get(pid)) and summaries_left(rdb):
                    n_summaries += summarize(pid, pdb, rdb, text)
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
                    summaries=n_summaries,
                    left=len(todo) - n,
                )
            if args.max and total >= args.max:
                break


if __name__ == '__main__':
    main()
