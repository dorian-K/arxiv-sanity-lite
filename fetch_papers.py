"""
Fetches specific arxiv papers by id into the papers database, e.g. older papers
you want to tag, or the papers rated in an evaluation dataset (see eval.py).

    python fetch_papers.py 2504.08385 1706.03762
    python fetch_papers.py --file ids.txt     # one id per line

Afterwards run compute.py so the new papers get features.
"""

import re
import sys
import time
import random
import logging
import argparse

from aslite.arxiv import get_response_ids, parse_response
from aslite.db import get_papers_db

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    parser = argparse.ArgumentParser(description='fetch arxiv papers by id')
    parser.add_argument('ids', nargs='*', help='arxiv ids, e.g. 2504.08385')
    parser.add_argument('-f', '--file', type=str, default='', help='file with one arxiv id per line')
    parser.add_argument('-b', '--batch', type=int, default=100, help='ids per arxiv api call')
    args = parser.parse_args()

    ids = list(args.ids)
    if args.file:
        with open(args.file) as f:
            ids += [line.strip() for line in f if line.strip()]
    ids = [re.sub(r'v\d+$', '', i) for i in ids]  # drop version suffixes

    pdb = get_papers_db()
    todo = list(dict.fromkeys(i for i in ids if i not in pdb))
    logging.info("%d ids given, %d not in the database yet", len(ids), len(todo))

    nnew = 0
    for k in range(0, len(todo), args.batch):
        batch = todo[k : k + args.batch]
        for attempt in range(10):
            try:
                papers = parse_response(get_response_ids(batch))
                break
            except Exception as e:
                wait = min(120, 5 * 2**attempt) + random.uniform(0, 5)  # back off, arxiv rate limits
                logging.warning("arxiv api error (%s), retrying in %.0fs...", e, wait)
                time.sleep(wait)
        else:
            logging.error("giving up on batch starting at %d", k)
            continue
        nnew += pdb.upsert(papers)
        logging.info(
            "fetched %d/%d (got %d papers back for %d ids)", k + len(batch), len(todo), len(papers), len(batch)
        )
        time.sleep(3)  # arxiv asks for 3s between api calls

    pdb.close()
    logging.info("added %d papers", nnew)
    sys.exit(0 if nnew > 0 else 1)
