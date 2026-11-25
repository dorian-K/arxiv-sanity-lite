"""
Fetches specific arxiv papers by id into the papers database, e.g. older papers
you want to tag, or the papers rated in an evaluation dataset (see eval.py).

    python fetch_papers.py 2504.08385 1706.03762
    python fetch_papers.py --file ids.txt     # one id per line

Afterwards run compute.py so the new papers get features. Every API call goes through
db.arxiv_slot (one per 3.5 s across all processes), honors arXiv's Retry-After and gives up
on a batch after a few tries.
"""

import re
import sys
import time
import random
import logging
import argparse
import urllib.error

from aslite import db
from aslite.arxiv import ARXIV_ID_RE, fetch_papers_by_id
from aslite.db import get_papers_db

TRIES = 5

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    parser = argparse.ArgumentParser(description='fetch arxiv papers by id')
    parser.add_argument('ids', nargs='*', help='arxiv ids, e.g. 2504.08385')
    parser.add_argument('-f', '--file', type=str, default='', help='file with one arxiv id per line')
    parser.add_argument('-b', '--batch', type=int, default=100, help='ids per arxiv api call (at most 100)')
    args = parser.parse_args()

    ids = list(args.ids)
    if args.file:
        with open(args.file) as f:
            ids += [line.strip() for line in f if line.strip()]
    ids = [re.sub(r'v\d+$', '', i) for i in ids]  # drop version suffixes
    bad = [i for i in ids if not ARXIV_ID_RE.fullmatch(i)]
    if bad:  # one of them in a batch made arXiv answer with an error for the whole batch
        logging.warning("skipping %d ids that aren't new-style arXiv ids: %s", len(bad), ', '.join(bad[:10]))
    ids = [i for i in ids if i not in bad]

    pdb = get_papers_db()
    todo = list(dict.fromkeys(i for i in ids if i not in pdb))
    logging.info("%d ids given, %d not in the database yet", len(ids), len(todo))

    nnew, batch_size = 0, max(1, min(100, args.batch))
    for k in range(0, len(todo), batch_size):
        batch = todo[k : k + batch_size]
        for attempt in range(TRIES):
            try:
                with db.arxiv_slot():
                    papers = fetch_papers_by_id(batch)
                break
            except Exception as e:
                retry_after = 0
                if isinstance(e, urllib.error.HTTPError) and e.code in (429, 503):
                    retry_after = (
                        int(e.headers.get('Retry-After') or 0)
                        if str(e.headers.get('Retry-After') or '').isdigit()
                        else 0
                    )
                wait = min(600, max(retry_after, 10 * 2**attempt)) + random.uniform(0, 5)
                logging.warning("arxiv api error (%s), retrying in %.0fs...", e, wait)
                time.sleep(wait)
        else:
            logging.error("giving up on batch starting at %d", k)
            continue
        nnew += pdb.upsert(list(papers.values()))
        logging.info(
            "fetched %d/%d (got %d papers back for %d ids)", k + len(batch), len(todo), len(papers), len(batch)
        )

    pdb.close()
    logging.info("added %d papers", nnew)
    sys.exit(0 if nnew > 0 else 1)
