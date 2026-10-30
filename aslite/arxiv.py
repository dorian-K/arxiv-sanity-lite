"""
Utils for dealing with arxiv API and related processing
"""

import re
import time
import calendar
import logging
import urllib.request
import feedparser
from collections import OrderedDict
from email.utils import parsedate_to_datetime

logger = logging.getLogger(__name__)


def get_response(search_query, start_index=0):
    """
    pings arxiv.org API to fetch a batch of 100 papers. raises on http errors (a 503 carries
    Retry-After); callers space calls >= 3s (see db.arxiv_slot)
    """
    base_url = 'https://export.arxiv.org/api/query?'
    add_url = 'search_query=%s&sortBy=lastUpdatedDate&start=%d&max_results=100' % (search_query, start_index)
    logger.debug(f"Searching arxiv for {base_url + add_url}")
    req = urllib.request.Request(base_url + add_url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as url:
        return url.read()


def get_response_ids(id_list):
    """pings arxiv.org API to fetch specific papers by id (up to ~100 per call)"""
    base_url = 'https://export.arxiv.org/api/query?'
    add_url = 'id_list=%s&max_results=%d' % (','.join(id_list), len(id_list))
    req = urllib.request.Request(base_url + add_url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as url:
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
        j['_time'] = calendar.timegm(j['updated_parsed'])  # a UTC struct_time; mktime would read it as local
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


# new-style arXiv ids (since 2007), with an optional version: 2401.12345, 0704.0001v2
ARXIV_ID_RE = re.compile(r'(?<![\d])(\d{4}\.\d{4,5})(?:v\d+)?(?![\d])')
# old-style ids like hep-th/0408218, which we can't store (the id has a slash)
OLD_ARXIV_ID_RE = re.compile(r'\b[a-z-]+(?:\.[A-Z]{2})?/\d{7}\b')
USER_AGENT = 'arxiv-resanity (self-hosted; daily update and papers on user request, >= 3s apart)'


def find_arxiv_id(text):
    """the arXiv id in a url or bare id (arxiv.org/abs|pdf/..., alphaxiv, hf.co/papers, ...), or None"""
    m = ARXIV_ID_RE.search(text)
    return m.group(1) if m else None


def fetch_papers_by_id(ids):
    """
    papers for up to 100 new-style ids in one API call, as {pid: paper}. ids arXiv doesn't
    know are missing from the result. callers must space calls >= 3s (see db.arxiv_slot)
    """
    assert len(ids) <= 100 and all(ARXIV_ID_RE.fullmatch(i) for i in ids)
    url = 'https://export.arxiv.org/api/query?id_list=%s&max_results=%d' % (','.join(ids), len(ids))
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        response = r.read()
    out = {}
    for e in feedparser.parse(response).entries:
        try:
            j = encode_feedparser_dict(e)
            idv, rawid, version = parse_arxiv_url(j['id'])
            if not j.get('title') or not j.get('updated_parsed'):
                continue  # arXiv's placeholder for an id it doesn't have
            j.update(_idv=idv, _id=rawid, _version=version)
            j['_time'] = calendar.timegm(j['updated_parsed'])
            j['_time_str'] = time.strftime('%b %d %Y', j['updated_parsed'])
            out[rawid] = j
        except (AssertionError, KeyError, ValueError):
            continue
    return out


# what an author's comment says when the paper went through peer review: "Accepted at NeurIPS
# 2026", "To appear in TPAMI", "Published in Nature", "Oral at ICML". not "Submitted to",
# "Under review" or "the published version"
VENUE_RE = re.compile(
    r'\b(accepted|accept|to appear (at|in)|published (at|in|by|as|to|online)|camera[- ]ready'
    r'|(in )?proceedings of|(oral|spotlight|poster)( presentation| paper)? (at|in))\b',
    re.IGNORECASE,
)
VENUE_NOT_RE = re.compile(r'\b(not|under review|submitted|for review|author[- ]accepted manuscript)\b', re.IGNORECASE)
_BOILERPLATE_RE = re.compile(r'^\s*\d+\s*(pages?|figures?|figs?|tables?|pp|appendi(x|ces))\b', re.IGNORECASE)


def venue(comment='', journal_ref=''):
    """
    where the paper was published or accepted, as the authors put it: the journal reference if
    there is one, else the clause of the comment that says so ('' for neither). a hint for
    readers, not a verified fact: arXiv doesn't check either field
    """
    if journal_ref and journal_ref.strip():
        return ' '.join(journal_ref.split())
    for part in re.split(r'(?<=[.;|])\s+|\s*\n\s*', comment or ''):
        if VENUE_RE.search(part) and not VENUE_NOT_RE.search(part):
            chunks = part.split(',')
            while len(chunks) > 1 and _BOILERPLATE_RE.match(chunks[0]):
                chunks.pop(0)  # "12 pages, 4 figures, accepted at ..."
            part = ' '.join(','.join(chunks).split()).rstrip('.;|')
            return part if len(part) <= 100 else part[:99].rstrip() + '…'
    return ''


# daemon categories, with what to call them; the bulk importers (harvest_papers.py,
# import_snapshot.py) filter on these too, and the about page, /agents and the agents'
# instructions list them (categories_text)
CATEGORY_NAMES = {
    'cs.LG': 'machine learning',
    'cs.CV': 'computer vision',
    'cs.CL': 'language',
    'cs.AI': 'AI',
    'cs.NE': 'neural computing',
    'cs.RO': 'robotics',
    'cs.SD': 'sound',
    'eess.AS': 'audio and speech',
}
CATEGORIES = set(CATEGORY_NAMES)


def categories_text():
    """'machine learning (cs.LG), computer vision (cs.CV), ... and audio and speech (eess.AS)'"""
    parts = [f'{name} ({c})' for c, name in CATEGORY_NAMES.items()]
    return ', '.join(parts[:-1]) + ' and ' + parts[-1]


def make_paper(pid, version_dates, title, abstract, authors, categories, comments='', journal_ref=''):
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
        '_time': calendar.timegm(updated),  # same as parse_response: true unix time of the UTC date
        '_time_str': time.strftime('%b %d %Y', updated),
        'id': f'http://arxiv.org/abs/{pid}v{v}',
        'title': ' '.join(title.split()),
        'summary': abstract.strip(),
        'authors': [{'name': n} for n in authors],
        'tags': [{'term': c, 'scheme': 'http://arxiv.org/schemas/atom', 'label': None} for c in cats],
        'arxiv_primary_category': {'term': cats[0] if cats else ''},
        'arxiv_comment': comments or '',
        'arxiv_journal_ref': ' '.join((journal_ref or '').split()),
        'published': first.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'updated': last.strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    return paper, first.timestamp()
