"""
References of papers, from arXiv's HTML version (LaTeXML), for "Cited by your saved papers".

arXiv renders most papers to HTML, where every reference is its own <li class="ltx_bibitem">,
and bibtex-made bibliographies label their parts (ltx_bib_title, ltx_bib_author, ...). That
splits the list cleanly where a PDF's text is garbled (two columns, author-year styles), and the
page is ~0.4 MB instead of a ~12 MB PDF. On 20 recommended papers: 962 references, 56% matched
to a paper in the database, all 12 checked by hand right (PDF: 808 and 38%).

A reference is matched by the arXiv id in its links or text, else by its title: the labelled
one (or the block after the authors), looked up exactly (normalized) in the table of all arXiv
titles (db.put_titles, built from the Hugging Face snapshot) and in our own papers; else by our
paper whose whole title appears in the reference. A match is an arXiv id, which may or may not
be in papers.sqlite (other categories); the page joins at display time, so papers added later
count too. No other service is asked (no Semantic Scholar, no OpenAlex): the user's choice.
"""

import re
import hashlib
import unicodedata
import urllib.error
import urllib.request
from html.parser import HTMLParser


PAUSE = 15  # seconds between pages: arxiv.org's robots.txt asks for Crawl-delay 15
USER_AGENT = 'arxiv-resanity (https://github.com/dorian-K/arxiv-sanity-lite)'
ARXIV_ID = re.compile(r'(?:arxiv[:\s/.]*|abs/|pdf/|html/)(\d{4}\.\d{4,5})', re.I)
DOI = re.compile(r'\b(10\.\d{4,9}/[^\s,;"<>]+[^\s,;.)"<>])')
VOID = {'br', 'img', 'hr', 'input', 'meta', 'wbr', 'col', 'source'}


def norm(s):
    """a title as lowercase words, without accents and punctuation"""
    plain = ''.join(c for c in unicodedata.normalize('NFKD', s or '') if not unicodedata.combining(c))
    return ' '.join(re.findall(r'[a-z0-9]+', plain.lower()))


def title_hash(title):
    """the key of a title in the titles table: 8 bytes of the sha1 of its normalized form"""
    return int.from_bytes(hashlib.sha1(norm(title).encode()).digest()[:8], 'big', signed=True)


class _Bib(HTMLParser):
    """per reference: its text (without "Cited by"), links, labelled title and blocks"""

    def __init__(self):
        super().__init__()
        self.refs, self.cur, self.stack = [], None, []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get('class') or ''
        if self.cur is None:
            if tag == 'li' and 'ltx_bibitem' in cls:
                self.cur, self.stack = {'text': [], 'links': [], 'title': [], 'blocks': []}, ['li']
            return
        if tag in VOID:
            return
        self.stack.append(cls)
        if tag == 'a' and a.get('href'):
            self.cur['links'].append(a['href'])
        if 'ltx_bibblock' in cls and 'ltx_bib_cited' not in cls:
            self.cur['blocks'].append([])

    def handle_endtag(self, tag):
        if self.cur is None or tag in VOID or not self.stack:
            return
        self.stack.pop()
        if not self.stack:
            c, self.cur = self.cur, None

            def join(parts):
                return ' '.join(''.join(parts).split())

            self.refs.append(
                {
                    'raw': join(c['text']),
                    'links': c['links'],
                    'title': join(c['title']).rstrip(' .'),
                    'blocks': [join(b).rstrip(' .') for b in c['blocks']],
                }
            )

    def handle_data(self, d):
        if self.cur is None or any('ltx_bib_cited' in c for c in self.stack):
            return  # "Cited by: §2" is about the citing paper, not the reference
        self.cur['text'].append(d)
        if any('ltx_bib_title' in c for c in self.stack):
            self.cur['title'].append(d)
        if self.cur['blocks'] and any('ltx_bibblock' in c for c in self.stack):
            self.cur['blocks'][-1].append(d)


def parse(html):
    """[{raw, links, title, blocks}] of the bibliography in an arXiv HTML page"""
    p = _Bib()
    p.feed(html)
    return p.refs


def candidate_titles(ref):
    """what the title of a reference probably is: the labelled one, else the blocks after the authors"""
    return [ref['title']] if ref.get('title') else [b for b in ref.get('blocks', [])[1:3] if b]


class Matcher:
    """finds the arXiv id a reference points to (see the module docstring)"""

    def __init__(self, pdb, rdb):
        self.pdb, self.rdb = pdb, rdb

    def _ours_by_title(self, title):
        """our paper with exactly this title: a phrase query on the title column, shortest first"""
        words = norm(title).split()
        if len(words) < 3:
            return None
        q = 'title : "' + ' '.join(words[:12]) + '"'
        rows = self.pdb.con.execute(
            'SELECT p.id, p.title FROM papers_fts JOIN papers p ON p.n = papers_fts.rowid '
            'WHERE papers_fts MATCH ? ORDER BY length(p.title) LIMIT 50',
            (q,),
        ).fetchall()
        return next((pid for pid, t in rows if norm(t) == ' '.join(words)), None)

    def _ours_in_text(self, raw):
        """our paper whose whole title appears in the reference (bibliographies without labels)"""
        nraw = norm(raw)
        chunks = [c for c in re.split(r'(?<=[.?!])\s+|[“”"]', raw) if len(norm(c).split()) >= 3]
        for c in chunks[:8]:
            q = 'title : "' + ' '.join(norm(c).split()[:6]) + '"'
            rows = self.pdb.con.execute(
                'SELECT p.id, p.title FROM papers_fts JOIN papers p ON p.n = papers_fts.rowid '
                'WHERE papers_fts MATCH ? ORDER BY length(p.title) LIMIT 20',
                (q,),
            ).fetchall()
            for pid, t in rows:
                nt = norm(t)
                if len(nt.split()) >= 3 and nt in nraw:
                    return pid
        return None

    def match(self, ref):
        """(arXiv id or None, how: id | title | text | None)"""
        a = ARXIV_ID.search(' '.join(ref.get('links', []))) or ARXIV_ID.search(ref.get('raw', ''))
        if a:
            return a.group(1), 'id'
        for t in candidate_titles(ref):
            pid = self.rdb.lookup_title(title_hash(t)) if len(norm(t).split()) >= 3 else None
            pid = pid or self._ours_by_title(t)
            if pid:
                return pid, 'title'
        pid = self._ours_in_text(ref.get('raw', ''))
        return (pid, 'text') if pid else (None, None)


def rows_for(refs, matcher):
    """the rows db.put_refs stores for a parsed bibliography"""
    out = []
    for i, r in enumerate(refs):
        cited, _ = matcher.match(r)
        d = DOI.search(' '.join(r['links'])) or DOI.search(r['raw'])
        title = (candidate_titles(r) or [''])[0]
        out.append((i, cited, title[:500], d.group(1) if d else None, r['raw'][:1000]))
    return out


class SlowDown(Exception):
    """arXiv asked us to wait (429/503): retry_after seconds"""

    def __init__(self, retry_after):
        super().__init__(f"arXiv asked to wait {retry_after}s")
        self.retry_after = retry_after


def fetch_html(pid, timeout=60):
    """
    (status, html) of a paper's HTML version: 'ok', or 'missing' (no HTML for it, or no
    bibliography in it). Raises SlowDown on 429/503, other errors as they are. The caller holds
    db.arxiv_slot(pause=PAUSE)
    """
    req = urllib.request.Request(f'https://arxiv.org/html/{pid}', headers={'User-Agent': USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            html = r.read(20_000_000).decode('utf-8', 'replace')
            final = r.geturl()
    except urllib.error.HTTPError as e:
        if e.code in (429, 503):
            try:
                wait = int(e.headers.get('Retry-After') or 600)
            except ValueError:
                wait = 600
            raise SlowDown(max(60, wait)) from e
        if e.code in (404, 410):
            return 'missing', ''
        raise
    if '/html/' not in final or 'ltx_bibitem' not in html:
        return 'missing', ''
    return 'ok', html
