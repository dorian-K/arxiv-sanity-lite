"""
Where papers come from. arXiv is the site's source: arxiv_daemon.py, the bulk importers and the
snapshot fill papers.sqlite from it. Other sources (the ACL Anthology so far) only ever add single
papers someone asks for: a user's /import, the graph's open box, a cited paper clicked on the
graph. We never crawl them.

Everything that depends on where a paper lives goes through the Source of its id: what its ids
look like, finding one in a link, its pages (abstract, pdf, html), its year, fetching its metadata
(one request per `batch` ids, inside the source's own slot), and its license. Code elsewhere never
builds a url of a paper site itself. Ids of different sources can't be confused: arXiv's are
2401.12345 (or cs/0701001, which we can't store), the Anthology's 2026.emnlp-main.2349 or P19-1001.
"""

import calendar
import html as htmllib
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser

from aslite import arxiv


class Source:
    name = ''  # short, stable
    label = ''  # for people: 'arXiv'
    venue_note = ''  # where a paper's venue comes from, shown next to it
    host = ''  # the site its pages are on: requests to it share one slot
    pause = 3.5  # seconds between metadata requests across all processes
    batch = 1  # ids per metadata request
    id_re = re.compile(r'(?!)')  # a stored id, whole

    def owns(self, pid):
        return bool(self.id_re.fullmatch(pid or ''))

    def find(self, text):
        """the id in a link or a bare id, or None"""
        raise NotImplementedError

    def abs_url(self, pid):
        raise NotImplementedError

    def pdf_url(self, pid):
        raise NotImplementedError

    def html_url(self, pid):
        """its full text as html (arXiv's LaTeXML), None if the source has none"""
        return None

    def year(self, pid):
        """the year its id says, or None"""
        return None

    def fetch(self, pids):
        """{pid: paper dict} for up to `batch` ids, one request; unknown ids are left out. raises on network errors"""
        raise NotImplementedError

    def license(self, pid):
        """its license code (aslite/thumbs.py), or None if only its abstract page says"""
        return None

    def slot(self):
        """the turn to send one request to this source (db.host_slot)"""
        from aslite import db  # db links papers through this module

        return db.host_slot(self.name, self.pause)


class ArXiv(Source):
    name, label, host, batch = 'arxiv', 'arXiv', 'arxiv.org', 100
    venue_note = 'from arXiv metadata, unverified'
    id_re = re.compile(r'\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7}')

    def find(self, text):
        return arxiv.find_arxiv_id(text)

    def abs_url(self, pid):
        return f'https://arxiv.org/abs/{pid}'

    def pdf_url(self, pid):
        return f'https://arxiv.org/pdf/{pid}'

    def html_url(self, pid):
        return f'https://arxiv.org/html/{pid}'

    def year(self, pid):
        m = re.match(r'(\d\d)\d\d\.', pid) or re.search(r'/(\d\d)', pid)
        return ((2000 if int(m.group(1)) < 90 else 1900) + int(m.group(1))) if m else None

    def fetch(self, pids):
        return arxiv.fetch_papers_by_id(pids)

    def slot(self):
        from aslite import db

        return db.arxiv_slot()


# Anthology ids: 2026.emnlp-main.2349 (since 2020: year, volume, paper) or P19-1001 (before)
ACL_ID = r'\d{4}\.[a-z][a-z\d]*(?:-[a-z\d]+)*\.\d+|[A-Z]\d{2}-\d{4}'
# in a link to the Anthology (also the old aclweb.org ones) or an ACL DOI (10.18653/v1/<id>)
ACL_IN = re.compile(
    rf'(?:aclanthology\.org/|aclweb\.org/anthology/(?:[A-Z]/[A-Z]\d\d/)?|10\.18653/(?:v1/)?)({ACL_ID})(?:v\d+)?(?![\w-])',
    re.I,
)
USER_AGENT = 'arxiv-resanity (self-hosted; single papers on user request)'


class _AclPage(HTMLParser):
    """the citation_* meta tags of an Anthology paper page, its abstract and its informal citation"""

    def __init__(self):
        super().__init__()
        self.meta, self.abstract, self.cite = {}, [], []
        self._in = None  # 'abstract' | 'cite' while inside one
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'meta' and (a.get('name') or '').startswith('citation_'):
            self.meta.setdefault(a['name'], []).append(a.get('content') or '')
        if self._in:
            self._depth += 1
        elif tag == 'div' and 'acl-abstract' in (a.get('class') or ''):
            self._in, self._depth = 'abstract', 1
        elif a.get('id') == 'citeRichText':
            self._in, self._depth = 'cite', 1

    def handle_endtag(self, tag):
        if self._in:
            self._depth -= 1
            if self._depth == 0:
                self._in = None

    def handle_data(self, data):
        if self._in == 'abstract':
            self.abstract.append(data)
        elif self._in == 'cite':
            self.cite.append(data)


def parse_acl_page(pid, page):
    """the paper dict of an Anthology paper page (shaped like arXiv's, see db._paper_to_row), or None"""
    p = _AclPage()
    p.feed(page)
    meta = lambda k: [' '.join(htmllib.unescape(v).split()) for v in p.meta.get(k, [])]
    title = (meta('citation_title') or [''])[0]
    if not title:
        return None
    abstract = ' '.join(''.join(p.abstract).split())
    abstract = re.sub(r'^Abstract\s*', '', abstract)
    m = re.match(r'(\d{4})(?:/(\d{1,2}))?', (meta('citation_publication_date') or [''])[0])
    year = int(m.group(1)) if m else ACL.year(pid) or 1970
    month = int(m.group(2)) if m and m.group(2) else 1
    day = datetime(year, min(max(month, 1), 12), 1, tzinfo=timezone.utc)
    # "(Kuan et al., EMNLP 2026)": the venue as people cite it, else the volume's full name
    cite = re.findall(r'\(([^()]*)\)', ' '.join(''.join(p.cite).split()))
    venue = cite[-1].rsplit(', ', 1)[-1] if cite and ', ' in cite[-1] else ''
    venue = venue or (meta('citation_conference_title') or meta('citation_journal_title') or [''])[0]
    return {
        '_id': pid,
        '_idv': f'{pid}v1',
        '_version': 1,
        '_time': calendar.timegm(day.utctimetuple()),
        '_time_str': day.strftime('%b %d %Y'),
        'published': day.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'title': title,
        'summary': abstract,
        'authors': [{'name': n} for n in meta('citation_author')],
        'tags': [{'term': 'cs.CL'}],  # computational linguistics, the Anthology's field
        'arxiv_comment': '',
        'arxiv_journal_ref': venue,  # what venue() shows as where it was published
    }


class Acl(Source):
    name, label, host, pause = 'acl', 'the ACL Anthology', 'aclanthology.org', 5
    venue_note = 'from the ACL Anthology'
    id_re = re.compile(ACL_ID)

    def find(self, text):
        m = ACL_IN.search(text)
        if m:
            return self.norm(m.group(1))
        bare = re.sub(r'(?:v\d+)?(?:\.pdf)?/?$', '', text.strip())
        return self.norm(bare) if re.fullmatch(ACL_ID, bare, re.I) else None

    @staticmethod
    def norm(pid):
        return pid.upper() if re.fullmatch(r'[A-Z]\d{2}-\d{4}', pid, re.I) else pid.lower()

    def abs_url(self, pid):
        return f'https://aclanthology.org/{pid}/'

    def pdf_url(self, pid):
        return f'https://aclanthology.org/{pid}.pdf'

    def year(self, pid):
        if pid[:4].isdigit():
            return int(pid[:4])
        yy = int(pid[1:3])
        return (2000 if yy < 60 else 1900) + yy

    def fetch(self, pids):
        out = {}
        for pid in pids:
            req = urllib.request.Request(self.abs_url(pid), headers={'User-Agent': USER_AGENT})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    page = r.read(5_000_000).decode('utf-8', 'replace')
            except urllib.error.HTTPError as e:
                if e.code in (404, 410):
                    continue
                raise
            paper = parse_acl_page(pid, page)
            if paper:
                out[pid] = paper
        return out

    def license(self, pid):
        # the Anthology's terms: CC BY 4.0 from 2016 on, CC BY-NC-SA 3.0 before
        year = self.year(pid) or 0
        return 'by/4.0' if year >= 2016 else 'by-nc-sa/3.0'


ARXIV, ACL = ArXiv(), Acl()
SOURCES = (ARXIV, ACL)


def source_of(pid):
    """the Source a stored id belongs to (arXiv for anything unknown, as before there were others)"""
    return next((s for s in SOURCES if s.owns(pid)), ARXIV)


def is_paper_id(pid):
    return any(s.owns(pid) for s in SOURCES)


def find_paper_id(text):
    """the paper id in a link, DOI or bare id of any source, or None. Anthology links first: they
    are unambiguous, while arXiv's pattern would find digits in anything"""
    return ACL.find(text) or ARXIV.find(text)


def abs_url(pid):
    return source_of(pid).abs_url(pid)


def pdf_url(pid):
    return source_of(pid).pdf_url(pid)


def links(pid):
    """the paper's own pages, for cards and graph nodes"""
    s = source_of(pid)
    return {'url': s.abs_url(pid), 'pdf': s.pdf_url(pid), 'source': s.label, 'venue_note': s.venue_note}


def year(pid):
    return source_of(pid).year(pid)


def source_of_url(url):
    """the Source whose site a url is on, else None"""
    host = re.sub(r'^https?://(?:www\.)?([^/:]+).*$', r'\1', url or '')
    return next((s for s in SOURCES if host == s.host or host.endswith('.' + s.host)), None)
