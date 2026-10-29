"""
This script is intended to wake up every 30 min or so (eg via cron),
it checks for any new arxiv papers via the arxiv API and stashes
them into a sqlite database.
"""

import sys
import time
import random
import logging
import argparse
import urllib.error

from aslite.arxiv import get_response, parse_response
from aslite.db import get_papers_db, arxiv_slot, put_job

MAX_TRIES = 5  # per page; then this run gives up, and the next one tries again


def fetch_page(q, k):
    """
    one page of results, politely: every call waits for its slot (>= 3.5 s apart, across
    processes), errors and empty pages back off exponentially, a 503 waits as long as
    Retry-After says. a short page is retried once, then taken as it is
    """
    short = None
    for attempt in range(MAX_TRIES):
        wait = 10 * 2**attempt + random.uniform(0, 5)
        try:
            with arxiv_slot():
                papers = parse_response(get_response(search_query=q, start_index=k))
            if len(papers) == 100 or (short is not None and papers):
                return papers
            if papers:
                short = papers
            logging.warning("got %d papers instead of 100", len(papers))
        except urllib.error.HTTPError as e:
            retry_after = e.headers.get('Retry-After', '') if e.headers else ''
            if retry_after.isdigit():
                wait = max(wait, int(retry_after))
            logging.warning("arxiv answered %s", e.code)
        except Exception as e:  # network trouble, timeouts
            logging.warning(e)
        logging.warning("trying again in %d s", wait)
        time.sleep(wait)
    return short


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, format='%(name)s %(levelname)s %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p'
    )

    parser = argparse.ArgumentParser(description='Arxiv Daemon')
    parser.add_argument('-n', '--num', type=int, default=100, help='up to how many papers to fetch')
    parser.add_argument('-s', '--start', type=int, default=0, help='start at what index')
    parser.add_argument(
        '-b',
        '--break-after',
        type=int,
        default=3,
        help='how many 0 new papers in a row would cause us to stop early? or 0 to disable.',
    )
    args = parser.parse_args()
    print(args)
    """
    Quick note on the break_after argument: In a typical setting where one wants to update
    the papers database you'd choose a slightly higher num, but then break out early in case
    we've reached older papers that are already part of the database, to spare the arxiv API.
    """

    # query string of papers to look for
    q = 'cat:cs.CV+OR+cat:cs.LG+OR+cat:cs.CL+OR+cat:cs.AI+OR+cat:cs.NE+OR+cat:cs.RO'

    pdb = get_papers_db()
    prevn = len(pdb)
    t_start = time.time()

    def done(status, new):
        put_job('arxiv', time.time() - t_start, new=new, ok=status != 2)  # for /stats
        sys.exit(status)

    def store(p):
        pdb.upsert([p])

    # fetch the latest papers
    total_updated = 0
    zero_updates_in_a_row = 0
    for k in range(args.start, args.start + args.num, 100):
        logging.info('querying arxiv api for query %s at start_index %d' % (q, k))

        papers = fetch_page(q, k)
        if not papers:
            logging.error("no papers from arxiv after %d tries, giving up for this run", MAX_TRIES)
            done(2 if total_updated == 0 else 0, total_updated)

        # process the batch of retrieved papers
        nhad, nnew, nreplace = 0, 0, 0
        for p in papers:
            pid = p['_id']
            if pid in pdb:
                # a new version, not a newer timestamp: stored times may differ by the timezone
                # offset of the machine that once fetched them (see calendar.timegm in aslite/arxiv.py)
                if int(p['_version']) > int(pdb[pid]['_version']):
                    # replace, this one is newer
                    store(p)
                    nreplace += 1
                else:
                    # we already had this paper, nothing to do
                    nhad += 1
            else:
                # new, simple store into database
                store(p)
                nnew += 1
        prevn = len(pdb)
        total_updated += nreplace + nnew

        # some diagnostic information on how things are coming along
        logging.info(papers[0]['_time_str'])
        logging.info(
            "k=%d, out of %d: had %d, replaced %d, new %d. now have: %d" % (k, len(papers), nhad, nreplace, nnew, prevn)
        )

        # early termination criteria
        if nnew == 0:
            zero_updates_in_a_row += 1
            if args.break_after > 0 and zero_updates_in_a_row >= args.break_after:
                logging.info("breaking out early, no new papers %d times in a row" % (args.break_after,))
                break
            elif k == 0:
                logging.info("our very first call for the latest there were no new papers, exitting")
                break
        else:
            zero_updates_in_a_row = 0

        # zzz
        time.sleep(3 + random.uniform(0, 3))  # arXiv asks for >= 3s between requests

    # exit with OK status if anything at all changed, 1 if nothing did, 2 if arxiv failed us
    done(0 if total_updated > 0 else 1, total_updated)
