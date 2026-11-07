"""
References of papers, from arXiv's HTML version (LaTeXML), for the Cited tab and the reference graph.

arXiv renders most papers to HTML, where every reference is its own <li class="ltx_bibitem">,
and bibtex-made bibliographies label their parts (ltx_bib_title, ltx_bib_author, ...). That
splits the list cleanly where a PDF's text is garbled (two columns, author-year styles), and the
page is ~0.4 MB instead of a ~12 MB PDF. On 20 recommended papers: 962 references, 56% matched
to a paper in the database, all 12 checked by hand right (PDF: 808 and 38%).

A reference is matched by the arXiv id in its links or text, else by its title: the labelled
one (or the block after the authors), looked up exactly (normalized) in the table of all arXiv
titles (RefsDB.put_titles, built from the Hugging Face snapshot) and in our own papers; else by our
paper whose whole title appears in the reference, or a piece of it that is exactly some arXiv
title. Papers without an HTML version (arXiv has HTML only from Dec 2023 on) fall back to their PDF
(parse_pdf: numbered lists, else author-year ones split where an entry starts with 'Surname, ';
enrich_daemon.py fetches it anyway for the page thumbnail). A match is an arXiv id, which may or may not
be in papers.sqlite (other categories); the page joins at display time, so papers added later
count too. No other service is asked (no Semantic Scholar, no OpenAlex): the user's choice.
"""

import re
import hashlib
import unicodedata
from html.parser import HTMLParser

from aslite.sources import ACL, ACL_IN


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


# PDFs (papers without an HTML version): the reference list after its heading, up to an
# appendix. Numbered lists ([1] ...) split at their numbers; author-year ones where a line ending a
# sentence is followed by one starting an entry: 'Surname, First' (Adam, most ML papers before
# 2018), or names followed by the year, 'First Last, and First Last. 2019.' (ACL) or
# 'Last, F. and Last, G. (2019).' (APA)
PDF_HEAD = re.compile(
    r'^\s*(?:\d+\.?\s*)?(references|bibliography|references and notes|literature cited)\s*$', re.I | re.M
)
PDF_END = re.compile(
    r'\n\s*(?:(?:[A-Z\d]{1,3}\.?\s+)?(?i:appendix|appendices|supplementary material)\b|[A-H]\.?\s+[A-Z][a-z]+(?: [A-Za-z]+){0,6}\s*\n)'
)
PDF_NUM = re.compile(r'(?:^|\n)\s*\[(\d{1,3})\]\s+')
# an author-year entry's first line: "Amari, Shun-Ichi.", "Hinton, G.E. and", "van der Maaten, L"
PDF_AUTHOR = re.compile(
    r"^(?:(?:van|von|de|der|den|la|le|da|di|del|du)\s+)*[A-Z][\w'’\-]+(?:\s[A-Z][\w'’\-]+)?,\s+[A-Z]"
)
YEAR = re.compile(r'\b(?:19|20)\d\d[a-z]?\b')
# names (no full stop but after an initial) and then the year: 'Samuel R. Bowman, and Chris Potts. 2015.'
NAMES_YEAR = re.compile(
    r"^(?=[A-Z])(?:[^.()\d]|(?<![A-Za-z])[A-Z]\.|\bet al\.){2,400}?(?:\.\s+(?:19|20)\d\d[a-z]?\.|\s\((?:19|20)\d\d[a-z]?\)\.?)(?:\s|$)"
)


# when parse_pdf learned author-year lists: papers read before it without finding a list may be
# read once more when someone asks for them on the graph page (may_reread)
PDF_AUTHOR_YEAR_SINCE = 1793836800  # 2026-11-05: author-year lists, also ACL and APA style


def may_reread(st):
    """whether a paper with this RefsDB.status_of entry may be read again: no list found, by an older parser"""
    return bool(st) and st[0] == 'ok' and not st[1] and st[3] < PDF_AUTHOR_YEAR_SINCE


def parse_pdf(text):
    """[{raw, links, title, blocks}] of a numbered reference list in a PDF's text, [] if there's none"""
    heads = list(PDF_HEAD.finditer(text))
    if not heads:
        return []
    t = text[heads[-1].end() :]
    end = PDF_END.search(t)
    if end and end.start() > 1500:
        t = t[: end.start()]
    t = re.sub(r'[\ufffe\u00ad]\s*', '', t)  # hyphens pdfium marks as soft: "Im\ufffeproving"
    parts = PDF_NUM.split(t)
    if len(parts) >= 7:  # at least 3 numbered references
        refs = [' '.join(parts[i + 1].split()) for i in range(1, len(parts) - 1, 2)]
    else:
        refs = _author_year(t, text)
    return [{'raw': r, 'links': [], 'title': '', 'blocks': []} for r in refs if r]


def _author_year(t, text):
    """the entries of an author-year list, [] if t doesn't look like one"""
    seen = {}
    for line in text.split('\n'):
        seen[line.strip()] = seen.get(line.strip(), 0) + 1
    # page numbers and running heads ("Published as a conference paper at ...") are no part of any entry
    keep = [x.strip() for x in t.split('\n') if x.strip() and not x.strip().isdigit() and seen.get(x.strip(), 0) < 3]
    refs, cur = [], []
    for i, line in enumerate(keep):
        if cur and cur[-1].endswith('.') and (PDF_AUTHOR.match(line) or NAMES_YEAR.match(' '.join(keep[i : i + 4]))):
            refs.append(' '.join(cur))
            cur = []
        cur.append(line)
    if cur:
        refs.append(' '.join(cur))
    refs = [' '.join(r.split()) for r in refs]
    # a list of references, not some other text: several entries, nearly all with a year, none huge
    if len(refs) < 5 or sum(bool(YEAR.search(r)) for r in refs) < 0.8 * len(refs) or max(map(len, refs)) > 1500:
        return []
    return refs


def _pieces(raw):
    """the sentence-like pieces of a reference, one of which is usually its title"""
    return [c.strip(' .,') for c in re.split(r'(?<=[.?!])\s+|[“”"]', raw) if len(norm(c).split()) >= 3]


def candidate_titles(ref):
    """what the title of a reference probably is: the labelled one, else the blocks after the authors"""
    return [ref['title']] if ref.get('title') else [b for b in ref.get('blocks', [])[1:3] if b]


class Matcher:
    """finds the paper id (arXiv, else another source's) a reference points to (see the module docstring)"""

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
        for c in _pieces(raw)[:8]:
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
        """(paper id or None, how: id | title | text | None)"""
        a = ARXIV_ID.search(' '.join(ref.get('links', []))) or ARXIV_ID.search(ref.get('raw', ''))
        if a:
            return a.group(1), 'id'
        for t in candidate_titles(ref):
            pid = self.rdb.lookup_title(title_hash(t)) if len(norm(t).split()) >= 3 else None
            pid = pid or self._ours_by_title(t)
            if pid:
                return pid, 'title'
        raw = ref.get('raw', '')
        for c in _pieces(raw)[:10]:  # a piece of the text that is exactly some arXiv paper's title
            pid = self.rdb.lookup_title(title_hash(c))
            if pid:
                return pid, 'text'
        pid = self._ours_in_text(raw)
        if pid:
            return pid, 'text'
        # no arXiv paper: a link or DOI to another source's (the ACL Anthology). kept as the cited id,
        # so the graph shows it and one click there fetches it (never fetched just for being cited)
        pid = ACL.find(' '.join(ref.get('links', []))) or (ACL_IN.search(raw) and ACL.find(raw))
        return (pid, 'id') if pid else (None, None)


def rows_for(refs, matcher):
    """the rows db.put_refs stores for a parsed bibliography"""
    out = []
    for i, r in enumerate(refs):
        cited, _ = matcher.match(r)
        d = DOI.search(' '.join(r['links'])) or DOI.search(r['raw'])
        title = (candidate_titles(r) or [''])[0]
        out.append((i, cited, title[:500], d.group(1) if d else None, r['raw'][:1000]))
    return out
