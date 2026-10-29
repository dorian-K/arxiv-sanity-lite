"""
Fetches the references of the papers people care about from arXiv's HTML pages, slowly, for
"Cited by your saved papers" (aslite/refs.py). Off unless REFS_ENABLED=1 (docker/update.sh).

Order: papers someone saved (they make the "cited" lists), newest first, then the papers in
everyone's latest digest. Each paper is fetched once; one without an HTML version (or without a
bibliography in it) is remembered and never asked for again, a failed one is retried after a
day, at most 3 times. One page per --pause seconds (REFS_PAUSE, at least 15: arxiv.org's
robots.txt), at most a few thousand a day, so in practice a few hundred. On 429/503 it waits as
long as arXiv says. Once a day, references that matched nothing are matched again: papers added
since (or a titles table built since, import_snapshot.py --titles) may know them now.

    REFS_ENABLED=1 docker compose up -d     # in docker: started by docker/update.sh
    uv run python refs_daemon.py --once     # what is queued, then exit
"""

import os
import time
import argparse

from aslite import db
from aslite.db import get_papers_db, get_refs_db
from aslite.refs import PAUSE, Matcher, SlowDown, fetch_html, parse, rows_for

REFRESH = 600  # seconds between looking for new saved papers when nothing is queued
RETRY = 86400  # a failed fetch is tried again after this long...
MAX_TRIES = 3  # ...this many times


def queue(pdb, rdb, now=None):
    """the papers to fetch, best first"""
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
    parser = argparse.ArgumentParser(description="fetches papers' references from arXiv's HTML pages")
    parser.add_argument(
        '--pause',
        type=float,
        default=float(os.environ.get('REFS_PAUSE') or 20),
        help=f'seconds between pages, at least {PAUSE} (default $REFS_PAUSE or 20)',
    )
    parser.add_argument('--once', action='store_true', help='exit when nothing is queued')
    parser.add_argument('--max', type=int, default=0, help='stop after this many papers (0: never)')
    args = parser.parse_args()
    pause = max(PAUSE, args.pause)

    with get_papers_db() as pdb, get_refs_db() as rdb:
        matcher = Matcher(pdb, rdb)
        last_rematch, last_fetch, total = 0.0, 0.0, 0
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
            print(f"{len(todo)} papers to fetch references for", flush=True)
            t0, n, n_refs, n_matched = time.time(), 0, 0, 0
            for pid in todo[:20]:  # then look at the queue again: new saved papers go first
                # our own pace for arxiv.org pages, then the slot all arXiv calls share (so the
                # API and a user's /import never wait behind these 15 s)
                time.sleep(max(0.0, last_fetch + pause - time.time()))
                try:
                    with db.arxiv_slot():
                        last_fetch = time.time()
                        status, html = fetch_html(pid)
                    rows = rows_for(parse(html), matcher) if status == 'ok' else []
                    if status == 'ok' and not rows:
                        status = 'missing'
                except SlowDown as e:
                    print(f"{e}, waiting", flush=True)
                    time.sleep(e.retry_after)
                    break
                except Exception as e:  # e.g. a timeout: tried again after RETRY
                    print(f"{pid}: {e}", flush=True)
                    rdb.put_refs(pid, 'error')
                    continue
                rdb.put_refs(pid, status, rows)
                n += 1
                n_refs += len(rows)
                n_matched += sum(1 for r in rows if r[1])
                print(f"  {pid}: {status}, {len(rows)} references, {sum(1 for r in rows if r[1])} matched", flush=True)
            total += n
            if n:
                db.put_job('refs', time.time() - t0, papers=n, refs=n_refs, matched=n_matched, left=len(todo) - n)
            if args.max and total >= args.max:
                break


if __name__ == '__main__':
    main()
