"""
Where papers come from. arXiv is the site's source: arxiv_daemon.py, the bulk importers and the
snapshot fill papers.sqlite from it. Other sources (the ACL Anthology, PMLR, JMLR, the CVF's open
access proceedings, NeurIPS's proceedings) only ever add single papers someone asks for: a user's
/import, the graph's open box, a cited paper clicked on the graph. We never crawl them.

Everything that depends on where a paper lives goes through the Source of its id: what its ids
look like, finding one in a link, its pages (abstract, pdf, html), its year, fetching its metadata
(one request per `batch` ids, inside the source's own slot), and its license. Code elsewhere never
builds a url of a paper site itself. Ids of different sources can't be confused: arXiv's are
2401.12345 (or cs/0701001, which we can't store), the Anthology's 2026.emnlp-main.2349 or P19-1001,
the others' start with their name (pmlr:v202.aamand23a). Ids never contain a '/', since they go
into paths (/add/<pid>/<tag>).

The proceedings sites all put the citation_* meta tags (Google Scholar's) on a paper's page, so
one parser (CitationPage) reads them all; a new site of that kind is a small subclass.
"""

import calendar
import html as htmllib
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser

from aslite import arxiv

USER_AGENT = 'arxiv-resanity (self-hosted; single papers on user request)'


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

    def find_link(self, text):
        """the id in a link to one of its pages (or a DOI of its own) anywhere in text, or None"""
        return None

    def find(self, text):
        """the id in a link, or a bare id, or None"""
        pid = self.find_link(text)
        return pid or (text.strip() if self.owns(text.strip()) else None)

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
        """its license code (aslite/thumbs.py; '' = none that allows a thumbnail), or None if only its abstract page says"""
        return ''

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

    def by_title(self, title, same=None):
        """the arXiv paper with this title for which same(paper) holds (a search of the API), or None"""
        return arxiv.search_title(title, same)

    def license(self, pid):
        return None  # the snapshot's table, else the abstract page (enrich_daemon.paper_license)

    def slot(self):
        from aslite import db

        return db.arxiv_slot()


class _CitationMeta(HTMLParser):
    """a paper page's citation_* meta tags, the text of its abstract (the first element whose id or
    class matches `abstract_re`) and of the element with id `cite_id` (the Anthology's informal citation)"""

    def __init__(self, abstract_re, cite_id=None):
        super().__init__()
        self.meta, self.abstract, self.cite = {}, [], []
        self.abstract_re, self.cite_id = abstract_re, cite_id
        self._in, self._depth, self._seen_abstract = None, 0, False

    def handle_starttag(self, tag, attrs):
        a = {k: v or '' for k, v in attrs}
        if tag == 'meta' and a.get('name', '').startswith('citation_'):
            self.meta.setdefault(a['name'], []).append(a.get('content', ''))
        if tag in ('meta', 'br', 'img', 'input', 'link', 'hr'):
            return  # void elements: no end tag to count
        if self._in:
            self._depth += 1
        elif not self._seen_abstract and self.abstract_re.search(a.get('id', '') + ' ' + a.get('class', '')):
            self._in, self._depth, self._seen_abstract = 'abstract', 1, True
        elif self.cite_id and a.get('id') == self.cite_id:
            self._in, self._depth = 'cite', 1

    def handle_endtag(self, tag):
        if self._in and tag not in ('meta', 'br', 'img', 'input', 'link', 'hr'):
            self._depth -= 1
            if self._depth == 0:
                self._in = None

    def handle_data(self, data):
        if self._in == 'abstract':
            self.abstract.append(data)
        elif self._in == 'cite':
            self.cite.append(data)


# full venue names in citation_conference_title / citation_journal_title, and what people call them
VENUE_SHORT = [
    ('Empirical Methods in Natural Language Processing', 'EMNLP'),
    ('International Conference on Machine Learning', 'ICML'),
    ('Neural Information Processing Systems', 'NeurIPS'),
    ('Computer Vision and Pattern Recognition', 'CVPR'),
    ('International Conference on Computer Vision', 'ICCV'),
    ('Winter Conference on Applications of Computer Vision', 'WACV'),
    ('Artificial Intelligence and Statistics', 'AISTATS'),
    ('Conference on Learning Theory', 'COLT'),
    ('Uncertainty in Artificial Intelligence', 'UAI'),
    ('Conference on Robot Learning', 'CoRL'),
    ('Learning for Dynamics and Control', 'L4DC'),
    ('Algorithmic Learning Theory', 'ALT'),
    ('Journal of Machine Learning Research', 'JMLR'),
]


def short_venue(full, year):
    """'ICML 2023' for 'Proceedings of the 40th International Conference on Machine Learning', else the full name"""
    short = next((s for name, s in VENUE_SHORT if name.lower() in full.lower()), '')
    if short and 'workshop' not in full.lower():
        return f'{short} {year}' if year else short
    return full


class CitationPage(Source):
    """a site whose paper pages carry citation_* meta tags: one request to the page per paper"""

    category = 'cs.LG'  # stored as the paper's category, so filters and the about page make sense of it
    abstract_re = re.compile(r'abstract', re.I)  # the id or class of the element holding the abstract

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
            paper = self.parse(pid, page)
            if paper:
                out[pid] = paper
        return out

    def venue(self, meta, cite, year):
        full = (meta('citation_conference_title') or meta('citation_journal_title') or [''])[0]
        return short_venue(full, year)

    def parse(self, pid, page, cite_id=None):
        """the paper dict of its page (shaped like arXiv's, see db._paper_to_row), or None if it isn't one"""
        p = _CitationMeta(self.abstract_re, cite_id)
        p.feed(page)
        meta = lambda k: [' '.join(htmllib.unescape(v).split()) for v in p.meta.get(k, [])]
        title = (meta('citation_title') or [''])[0]
        if not title:
            return None
        abstract = ' '.join(''.join(p.abstract).split())
        abstract = re.sub(r'^Abstract:?\s*', '', abstract)
        m = re.match(r'(\d{4})(?:[/-](\d{1,2}))?(?:[/-](\d{1,2}))?', (meta('citation_publication_date') or [''])[0])
        year = int(m.group(1)) if m else self.year(pid) or 1970
        month = min(max(int(m.group(2)), 1), 12) if m and m.group(2) else 1
        day = min(max(int(m.group(3)), 1), 28) if m and m.group(3) else 1
        when = datetime(year, month, day, tzinfo=timezone.utc)
        # "Kirillov, Alexander" (CVF, NeurIPS) reads "Alexander Kirillov" everywhere else
        authors = [re.sub(r'^([^,]+), ([^,]+)$', r'\2 \1', a) for a in meta('citation_author')]
        return {
            '_id': pid,
            '_idv': f'{pid}v1',
            '_version': 1,
            '_time': calendar.timegm(when.utctimetuple()),
            '_time_str': when.strftime('%b %d %Y'),
            'published': when.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'title': title,
            'summary': abstract,
            'authors': [{'name': a} for a in authors],
            'tags': [{'term': self.category}],
            'arxiv_comment': '',
            'arxiv_journal_ref': self.venue(meta, ' '.join(''.join(p.cite).split()), year),  # what venue() shows
        }


# Anthology ids: 2026.emnlp-main.2349 (since 2020: year, volume, paper) or P19-1001 (before)
ACL_ID = r'\d{4}\.[a-z][a-z\d]*(?:-[a-z\d]+)*\.\d+|[A-Z]\d{2}-\d{4}'
# in a link to the Anthology (also the old aclweb.org ones) or an ACL DOI (10.18653/v1/<id>)
ACL_IN = re.compile(
    rf'(?:aclanthology\.org/|aclweb\.org/anthology/(?:[A-Z]/[A-Z]\d\d/)?|10\.18653/(?:v1/)?)({ACL_ID})(?:v\d+)?(?![\w-])',
    re.I,
)


class Acl(CitationPage):
    name, label, host, pause, category = 'acl', 'the ACL Anthology', 'aclanthology.org', 5, 'cs.CL'
    venue_note = 'from the ACL Anthology'
    id_re = re.compile(ACL_ID)
    abstract_re = re.compile(r'\bacl-abstract\b')  # not the corrections form's "Correct abstract if needed"

    def find_link(self, text):
        m = ACL_IN.search(text or '')
        return self.norm(m.group(1)) if m else None

    def find(self, text):
        bare = re.sub(r'(?:v\d+)?(?:\.pdf)?/?$', '', text.strip())
        return self.find_link(text) or (self.norm(bare) if re.fullmatch(ACL_ID, bare, re.I) else None)

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

    def parse(self, pid, page, cite_id='citeRichText'):
        return super().parse(pid, page, cite_id)

    def venue(self, meta, cite, year):
        # "(Kuan et al., EMNLP 2026)": the venue as people cite it, else the volume's full name
        last = re.findall(r'\(([^()]*)\)', cite)
        if last and ', ' in last[-1]:
            return last[-1].rsplit(', ', 1)[-1]
        return (meta('citation_conference_title') or meta('citation_journal_title') or [''])[0]

    def license(self, pid):
        # the Anthology's terms: CC BY 4.0 from 2016 on, CC BY-NC-SA 3.0 before
        return 'by/4.0' if (self.year(pid) or 0) >= 2016 else 'by-nc-sa/3.0'


class Pmlr(CitationPage):
    """Proceedings of Machine Learning Research: ICML, AISTATS, COLT, CoRL, ... (pmlr:v202.aamand23a)"""

    name, label, host = 'pmlr', 'PMLR', 'proceedings.mlr.press'
    venue_note = 'from PMLR'
    id_re = re.compile(r'pmlr:v\d+\.[a-z\d-]+')

    def find_link(self, text):
        m = re.search(
            r'(?:proceedings\.mlr\.press|jmlr\.org/proceedings/papers)/(v\d+)/([a-z\d-]+?)(?:\.html|\.pdf|/\2\.pdf)',
            text or '',
        )
        return f'pmlr:{m.group(1)}.{m.group(2)}' if m else None

    def _parts(self, pid):
        return pid[len('pmlr:') :].split('.', 1)

    def abs_url(self, pid):
        v, key = self._parts(pid)
        return f'https://proceedings.mlr.press/{v}/{key}.html'

    def pdf_url(self, pid):
        v, key = self._parts(pid)
        return f'https://proceedings.mlr.press/{v}/{key}/{key}.pdf'

    def year(self, pid):
        m = re.search(r'(\d\d)[a-z]*$', self._parts(pid)[1])  # aamand23a: 2023
        return 2000 + int(m.group(1)) if m else None


class Jmlr(CitationPage):
    """the Journal of Machine Learning Research (jmlr:v24.18-080)"""

    name, label, host = 'jmlr', 'JMLR', 'jmlr.org'
    venue_note = 'from JMLR'
    id_re = re.compile(r'jmlr:v\d+\.[\w-]+')

    def find_link(self, text):
        m = re.search(r'jmlr\.org/papers/(?:v|volume)(\d+)/([\w-]+?)(?:\.html|\.pdf|/|$|\s)', text or '')
        return f'jmlr:v{m.group(1)}.{m.group(2)}' if m else None

    def _parts(self, pid):
        v, key = pid[len('jmlr:v') :].split('.', 1)
        return v, key

    def abs_url(self, pid):
        v, key = self._parts(pid)
        return f'https://jmlr.org/papers/v{v}/{key}.html'

    def pdf_url(self, pid):
        v, key = self._parts(pid)
        return f'https://jmlr.org/papers/volume{v}/{key}/{key}.pdf'

    def year(self, pid):
        return 1999 + int(self._parts(pid)[0])  # volume 1 is 2000


class Cvf(CitationPage):
    """the CVF's open access proceedings: CVPR, ICCV, WACV and their workshops
    (cvf:ICCV2023.Kirillov_Segment_Anything_ICCV_2023_paper; the folder's '/' as '~' for workshops)"""

    name, label, host, category = 'cvf', 'CVF Open Access', 'openaccess.thecvf.com', 'cs.CV'
    venue_note = 'from CVF Open Access'
    id_re = re.compile(r'cvf:[\w~-]+\.[\w-]+_paper')

    def find_link(self, text):
        m = re.search(
            r'openaccess\.thecvf\.com/(content/[\w/-]+?|content_\w+)/(?:html|papers)/([\w-]+_paper)\.(?:html|pdf)',
            text or '',
        )
        if not m:
            return None
        folder = m.group(1)[len('content/') :] if m.group(1).startswith('content/') else m.group(1)
        return f'cvf:{folder.replace("/", "~")}.{m.group(2)}'

    def _parts(self, pid):
        folder, name = pid[len('cvf:') :].split('.', 1)
        folder = folder.replace('~', '/')
        return (folder if folder.startswith('content_') else 'content/' + folder), name

    def abs_url(self, pid):
        folder, name = self._parts(pid)
        return f'https://openaccess.thecvf.com/{folder}/html/{name}.html'

    def pdf_url(self, pid):
        folder, name = self._parts(pid)
        return f'https://openaccess.thecvf.com/{folder}/papers/{name}.pdf'

    def year(self, pid):
        m = re.search(r'_(\d{4})_paper$', pid)
        return int(m.group(1)) if m else None


class NeurIps(CitationPage):
    """NeurIPS's proceedings (neurips:2017.3f5ee243547dee91fbd053c1c4a845aa, since 2022 with the track:
    neurips:2023.<hash>-Conference), the part after -Abstract/-Paper in its file names"""

    name, label, host = 'neurips', 'NeurIPS Proceedings', 'proceedings.neurips.cc'
    venue_note = 'from the NeurIPS proceedings'
    id_re = re.compile(r'neurips:\d{4}\.[0-9a-f]{32}(?:-\w+)?')

    def find_link(self, text):
        m = re.search(
            r'(?:proceedings\.neurips\.cc|papers\.nips\.cc)/paper(?:_files/paper)?/(\d{4})/(?:hash|file)/'
            r'([0-9a-f]{32})-(?:Abstract|Paper)(-\w+?)?\.(?:html|pdf)',
            text or '',
        )
        return f'neurips:{m.group(1)}.{m.group(2)}{m.group(3) or ""}' if m else None

    def _parts(self, pid):
        year, rest = pid[len('neurips:') :].split('.', 1)
        return year, rest[:32], rest[32:]

    def abs_url(self, pid):
        year, h, track = self._parts(pid)
        return f'https://proceedings.neurips.cc/paper_files/paper/{year}/hash/{h}-Abstract{track}.html'

    def pdf_url(self, pid):
        year, h, track = self._parts(pid)
        return f'https://proceedings.neurips.cc/paper_files/paper/{year}/file/{h}-Paper{track}.pdf'

    def year(self, pid):
        return int(self._parts(pid)[0])


ARXIV, ACL, PMLR, JMLR, CVF, NEURIPS = ArXiv(), Acl(), Pmlr(), Jmlr(), Cvf(), NeurIps()
SOURCES = (ARXIV, ACL, PMLR, JMLR, CVF, NEURIPS)
OTHERS = SOURCES[1:]


def source_of(pid):
    """the Source a stored id belongs to (arXiv for anything unknown, as before there were others)"""
    return next((s for s in SOURCES if s.owns(pid)), ARXIV)


def is_paper_id(pid):
    return any(s.owns(pid) for s in SOURCES)


def find_paper_id(text):
    """the paper id in a link, DOI or bare id of any source, or None. The other sites' links first:
    they are unambiguous, while arXiv's pattern would find digits in anything"""
    for s in OTHERS:
        pid = s.find(text)
        if pid:
            return pid
    return ARXIV.find(text)


def find_linked(text):
    """the id of a paper outside arXiv that a link or DOI in text points to (a reference's), or None"""
    return next((pid for s in OTHERS if (pid := s.find_link(text))), None)


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
