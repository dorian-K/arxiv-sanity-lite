"""
Thumbnails of papers, made by enrich_daemon.py, two kinds per paper:

- 'figure': the first real figure of arXiv's HTML version (LaTeXML), usually the overview
  figure. Figures made of small panels (teaser grids of 86 px tiles) are skipped for the next
  figure with one image at least MIN_FIGURE px wide.
- 'page': the first page of the PDF, rendered with pypdfium2 (no ImageMagick, no shell).

Both are JPEGs of at most SIZE, in data/thumbs/ (db.save_thumb). Which one the cards show is
THUMB_KIND in serve.py; the other is the fallback. Images from papers are untrusted input:
Pillow's decompression-bomb limit is lowered, and every failure here just means no thumbnail.

Only papers under an open license get thumbnails (is_open): a thumbnail is a copy of the
author's work, which German law (no fair use) allows only with permission, however small it is.
Creative Commons licenses give that permission with credit, so cards show the license next to
the image; resizing is a technical modification every CC license allows, even ND. arXiv's
default license (half of the papers) only lets arXiv distribute them: no thumbnail, and the
figure and page aren't even downloaded. Reading a PDF's references is text and data mining
(§44b UrhG), and the PDF isn't kept. The license comes from the snapshot (import_snapshot.py
--titles), or for newer papers from the abstract page (abs_license).
"""

import io
import re
from html.parser import HTMLParser
from urllib.parse import urljoin

SIZE = (480, 360)
MIN_FIGURE = 200  # px: narrower images are panels or icons
MAX_PIXELS = 40_000_000
ALLOWED = 'https://arxiv.org/html/'  # figures are only fetched from arXiv's own HTML pages
_CC = re.compile(
    r'creativecommons\.org/(?:licenses/([a-z-]+)/(\d\.\d)|publicdomain/(zero)/(\d\.\d)|licenses/(publicdomain))'
)


def license_code(url):
    """a license url (as arXiv gives it) as a short code: 'by/4.0', 'by-nc-sa/4.0', 'zero/1.0',
    'publicdomain', 'arxiv' (arXiv's default license, nonexclusive-distrib), '' if unknown"""
    url = url or ''
    m = _CC.search(url)
    if m:
        if m.group(5):
            return 'publicdomain'
        kind, ver = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        return f'{kind}/{ver}'
    return 'arxiv' if 'arxiv.org/licenses' in url else ''


def is_open(code):
    """whether the license allows showing a (small) copy, with credit"""
    return bool(code) and code != 'arxiv'


def license_label(code):
    """'CC BY 4.0', 'CC0 1.0', 'Public domain'"""
    if code == 'publicdomain':
        return 'Public domain'
    kind, _, ver = code.partition('/')
    return f'CC0 {ver}' if kind == 'zero' else f"CC {kind.upper()} {ver}"


def license_url(code):
    if code == 'publicdomain':
        return 'https://creativecommons.org/licenses/publicdomain/'
    kind, _, ver = code.partition('/')
    if kind == 'zero':
        return f'https://creativecommons.org/publicdomain/zero/{ver}/'
    return f'https://creativecommons.org/licenses/{kind}/{ver}/'


def abs_license(html):
    """the license code from a paper's abstract page (arxiv.org/abs/<id>), '' if it shows none"""
    m = re.search(r'class="abs-license"[^<]*<a href="([^"]+)"', html)
    return license_code(m.group(1)) if m else ''


class _Figures(HTMLParser):
    """per top-level <figure class="ltx_figure">: its images as (src, declared width or None)"""

    def __init__(self):
        super().__init__()
        self.figures, self.depth = [], 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get('class') or ''
        if tag == 'figure':
            if self.depth == 0 and 'ltx_figure' in cls and 'ltx_table' not in cls:
                self.figures.append([])
                self.depth = 1
            elif self.depth:
                self.depth += 1
        elif tag == 'img' and self.depth and 'ltx_graphics' in cls and a.get('src'):
            try:
                width = int(a.get('width') or 0) or None
            except ValueError:
                width = None
            self.figures[-1].append((a['src'], width))

    def handle_endtag(self, tag):
        if tag == 'figure' and self.depth:
            self.depth -= 1


def figure_urls(html, page_url, n=3):
    """
    the image of each of the first figures that has one big enough image, best first: a figure
    with a single image, or the widest of its images if that is at least MIN_FIGURE px
    """
    p = _Figures()
    p.feed(html)
    out = []
    for imgs in p.figures:
        if not imgs:
            continue
        src, width = max(imgs, key=lambda x: x[1] or 0)
        if len(imgs) == 1 or (width or 0) >= MIN_FIGURE:
            if width is not None and width < MIN_FIGURE:
                continue
            url = urljoin(page_url, src)
            if url.startswith(ALLOWED):
                out.append(url)
        if len(out) >= n:
            break
    return out


def _jpeg(img):
    from PIL import Image

    if img.mode in ('RGBA', 'LA', 'P'):
        img = img.convert('RGBA')
        bg = Image.new('RGB', img.size, 'white')  # transparent figures on white, as on the page
        bg.paste(img, mask=img.getchannel('A'))
        img = bg
    else:
        img = img.convert('RGB')
    img.thumbnail(SIZE)
    out = io.BytesIO()
    img.save(out, 'JPEG', quality=82, optimize=True)
    return out.getvalue()


def from_image(data):
    """a thumbnail of an image file (png, jpg, gif, ...); raises on anything else"""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    with Image.open(io.BytesIO(data)) as img:
        img.load()
        if img.width < 32 or img.height < 32:
            raise ValueError('too small to be a figure')
        return _jpeg(img)


def _pdf(data):
    import pypdfium2 as pdfium

    return pdfium.PdfDocument(data)


def from_pdf(data):
    """a thumbnail of the first page of a PDF"""
    pdf = _pdf(data)
    try:
        page = pdf[0]
        scale = SIZE[0] / max(page.get_width(), 1)
        return _jpeg(page.render(scale=scale).to_pil())
    finally:
        pdf.close()


def pdf_text(data, max_pages=60):
    """the text of a PDF, page by page (for the references of papers without an HTML version)"""
    pdf = _pdf(data)
    try:
        return '\n'.join(pdf[i].get_textpage().get_text_range() for i in range(min(len(pdf), max_pages)))
    finally:
        pdf.close()
