"""
Bulk-imports years of papers through arXiv's OAI-PMH interface, which is meant for
harvesting and returns ~1300 records per request (the search API used by
arxiv_daemon.py pages 100 at a time). Records have the same fields as the ones from
arxiv_daemon.py, so compute.py, serve.py etc. work on the result unchanged.

    python harvest_papers.py --years 5
    ARXIV_SANITY_DATA=data/5y python harvest_papers.py --years 5   # into a separate corpus

It can be stopped and restarted: progress is saved after every page.
"""

import os
import re
import sys
import time
import json
import logging
import argparse
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

from aslite.arxiv import make_paper, CATEGORIES
from aslite.db import get_papers_db, DATA_DIR

OAI = 'https://oaipmh.arxiv.org/oai'
NS = {'oai': 'http://www.openarchives.org/OAI/2.0/', 'raw': 'http://arxiv.org/OAI/arXivRaw/'}
STATE_FILE = os.path.join(DATA_DIR, 'harvest_state.json')
PAUSE = 5  # seconds between requests; arXiv asks for at least 3, be nicer than that
USER_AGENT = 'arxiv-sanity-lite harvest_papers.py (one-off bulk import, polite: 1 request / 5s)'


def fetch(url):
    """GET with arXiv's OAI flow control: 503 + Retry-After means wait and ask again"""
    for attempt in range(20):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 503 and attempt > 3:
                raise
            wait = int(e.headers.get('Retry-After') or 30)
            logging.info("http %d, waiting %ds", e.code, wait)
        except (urllib.error.URLError, TimeoutError) as e:
            wait = min(300, 10 * 2**attempt)
            logging.warning("%s, retrying in %ds", e, wait)
        time.sleep(wait)
    raise RuntimeError('giving up on ' + url)


def split_authors(s):
    """ "A, B and C" / "A, B, and C" -> [A, B, C]; commas inside (affiliations) are kept"""
    s = ' '.join(s.split())
    out, depth, cur = [], 0, ''
    for ch in s:
        depth += ch == '('
        depth -= ch == ')'
        if ch == ',' and depth == 0:
            out.append(cur)
            cur = ''
        else:
            cur += ch
    out.append(cur)
    names = []
    for part in out:
        names += re.split(r'^\s*and\s+|\s+and\s+', part) if depth == 0 else [part]
    return [{'name': n.strip()} for n in names if n.strip()]


def to_paper(meta):
    """arXivRaw record -> (paper, first submission timestamp), see aslite.arxiv.make_paper"""
    g = lambda tag: (meta.findtext('raw:' + tag, default='', namespaces=NS) or '').strip()
    dates = [v.findtext('raw:date', namespaces=NS) for v in meta.findall('raw:version', NS)]
    names = [a['name'] for a in split_authors(g('authors'))]
    return make_paper(g('id'), dates, g('title'), g('abstract'), names, g('categories'), g('comments'))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    parser = argparse.ArgumentParser(description='bulk-import papers from arXiv OAI-PMH')
    parser.add_argument('--years', type=float, default=5, help='papers first submitted in the last this many years')
    parser.add_argument(
        '--set', default='cs', help='OAI set to harvest (then filtered to the arxiv_daemon.py categories)'
    )
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    since = time.time() - args.years * 365.25 * 86400
    since_day = time.strftime('%Y-%m-%d', time.gmtime(since))
    state = json.load(open(STATE_FILE)) if os.path.isfile(STATE_FILE) else {}
    if state.get('since') != since_day:
        state = {'since': since_day, 'token': None, 'pages': 0, 'kept': 0, 'seen': 0}
    logging.info("harvesting set %s, papers first submitted since %s, into %s", args.set, since_day, DATA_DIR)

    pdb = get_papers_db()
    while True:
        if state['token']:
            url = f"{OAI}?verb=ListRecords&resumptionToken={urllib.request.quote(state['token'])}"
        elif state['pages'] == 0:
            # the OAI datestamp is the last change of a record, so everything submitted since `since`
            # has a datestamp after it; older papers that were merely updated get filtered below
            url = f"{OAI}?verb=ListRecords&metadataPrefix=arXivRaw&set={args.set}&from={since_day}"
        else:
            break
        root = ET.fromstring(fetch(url))
        err = root.find('oai:error', NS)
        if err is not None:
            if err.get('code') == 'noRecordsMatch':
                break
            sys.exit(f"OAI error {err.get('code')}: {err.text}")

        papers = []
        for rec in root.iterfind('.//oai:record', NS):
            meta = rec.find('oai:metadata/raw:arXivRaw', NS)
            state['seen'] += 1
            if meta is None:
                continue  # deleted record
            r = to_paper(meta)
            if r and r[1] >= since and CATEGORIES & {t['term'] for t in r[0]['tags']}:
                papers.append(r[0])

        n_new = pdb.upsert(papers)  # only writes papers that are new or have a newer version

        tok = root.find('.//oai:resumptionToken', NS)
        state['token'] = tok.text.strip() if tok is not None and tok.text and tok.text.strip() else None
        state['pages'] += 1
        state['kept'] += n_new
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
        total = tok.get('completeListSize') if tok is not None else None
        logging.info(
            "page %d: %d records seen%s, %d papers stored (%d in db)",
            state['pages'],
            state['seen'],
            f' of {total}' if total else '',
            state['kept'],
            len(pdb),
        )
        if not state['token']:
            break
        time.sleep(PAUSE)

    pdb.close()
    logging.info("done: %d papers stored", state['kept'])
