"""
Utils for dealing with arxiv API and related processing
"""

import time
import logging
import urllib.request
import feedparser
from collections import OrderedDict
from email.utils import parsedate_to_datetime

logger = logging.getLogger(__name__)


def get_response(search_query, start_index=0):
    """pings arxiv.org API to fetch a batch of 100 papers"""
    # fetch raw response
    base_url = 'http://export.arxiv.org/api/query?'
    add_url = 'search_query=%s&sortBy=lastUpdatedDate&start=%d&max_results=100' % (search_query, start_index)
    # add_url = 'search_query=%s&sortBy=submittedDate&start=%d&max_results=100' % (search_query, start_index)
    search_query = base_url + add_url
    logger.debug(f"Searching arxiv for {search_query}")
    with urllib.request.urlopen(search_query) as url:
        response = url.read()

    if url.status != 200:
        logger.error("arxiv did not return status 200 response")

    return response


def get_response_ids(id_list):
    """pings arxiv.org API to fetch specific papers by id (up to ~100 per call)"""
    base_url = 'http://export.arxiv.org/api/query?'
    add_url = 'id_list=%s&max_results=%d' % (','.join(id_list), len(id_list))
    with urllib.request.urlopen(base_url + add_url, timeout=60) as url:
        response = url.read()
    return response


def encode_feedparser_dict(d):
    """helper function to strip feedparser objects using a deep copy"""
    if isinstance(d, feedparser.FeedParserDict) or isinstance(d, dict):
        return {k: encode_feedparser_dict(d[k]) for k in d.keys()}
    elif isinstance(d, list):
        return [encode_feedparser_dict(k) for k in d]
    else:
        return d


def parse_arxiv_url(url):
    """
    examples is http://arxiv.org/abs/1512.08756v2
    we want to extract the raw id (1512.08756) and the version (2)
    """
    ix = url.rfind('/')
    assert ix >= 0, 'bad url: ' + url
    idv = url[ix + 1 :]  # extract just the id (and the version)
    parts = idv.split('v')
    assert len(parts) == 2, 'error splitting id and version in idv string: ' + idv
    return idv, parts[0], int(parts[1])


def parse_response(response):

    out = []
    parse = feedparser.parse(response)
    for e in parse.entries:
        j = encode_feedparser_dict(e)
        # extract / parse id information
        idv, rawid, version = parse_arxiv_url(j['id'])
        j['_idv'] = idv
        j['_id'] = rawid
        j['_version'] = version
        j['_time'] = time.mktime(j['updated_parsed'])
        j['_time_str'] = time.strftime('%b %d %Y', j['updated_parsed'])
        # delete apparently spurious and redundant information
        del j['summary_detail']
        del j['title_detail']
        out.append(j)

    return out


def filter_latest_version(idvs):
    """
    for each idv filter the list down to only the most recent version
    """

    pid_to_v = OrderedDict()
    for idv in idvs:
        pid, v = idv.split('v')
        pid_to_v[pid] = max(int(v), pid_to_v.get(pid, 0))

    filt = [f"{pid}v{v}" for pid, v in pid_to_v.items()]
    return filt


# daemon categories; the bulk importers (harvest_papers.py, import_snapshot.py) filter on these too
CATEGORIES = {'cs.CV', 'cs.LG', 'cs.CL', 'cs.AI', 'cs.NE', 'cs.RO'}


def make_paper(pid, version_dates, title, abstract, authors, categories, comments=''):
    """
    builds a paper dict in the shape arxiv_daemon.py stores (feedparser-like) from bulk
    metadata. version_dates are the RFC 2822 dates of v1..vN, authors a list of names.
    returns (paper, first submission timestamp), or None for old-style ids (hep-th/0408218),
    which parse_arxiv_url would store under a mangled id
    """
    if '/' in pid or not version_dates:
        return None
    first = parsedate_to_datetime(version_dates[0])
    last = parsedate_to_datetime(version_dates[-1])
    v = len(version_dates)
    cats = categories.split()
    updated = last.utctimetuple()
    paper = {
        '_id': pid,
        '_idv': f'{pid}v{v}',
        '_version': v,
        '_time': time.mktime(updated),  # same convention as parse_response
        '_time_str': time.strftime('%b %d %Y', updated),
        'id': f'http://arxiv.org/abs/{pid}v{v}',
        'title': ' '.join(title.split()),
        'summary': abstract.strip(),
        'authors': [{'name': n} for n in authors],
        'tags': [{'term': c, 'scheme': 'http://arxiv.org/schemas/atom', 'label': None} for c in cats],
        'arxiv_primary_category': {'term': cats[0] if cats else ''},
        'arxiv_comment': comments or '',
        'published': first.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'updated': last.strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    return paper, first.timestamp()
