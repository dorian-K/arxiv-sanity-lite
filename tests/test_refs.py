"""references and thumbnails (aslite/refs.py, aslite/thumbs.py, enrich_daemon.py) and the Cited tab"""

import io
import sys
import urllib.error

from conftest import login
from aslite import db, refs, thumbs


def bibitem(n, title=None, blocks=(), link=None):
    """a <li> as LaTeXML writes it: labelled (bibtex) or as plain blocks"""
    parts = [f'<span class="ltx_tag ltx_tag_bibitem">[{n}]</span>']
    if title is not None:
        parts += [
            '<span class="ltx_bibblock"><span class="ltx_text ltx_bib_author">A. Author, B. Author</span>'
            '<span class="ltx_text ltx_bib_year"> (2024)</span></span>',
            f'<span class="ltx_bibblock"><span class="ltx_text ltx_bib_title">{title}</span>.</span>',
        ]
    parts += [f'<span class="ltx_bibblock">{b}.</span>' for b in blocks]
    if link:
        parts.append(f'<span class="ltx_bibblock">External Links: <a href="{link}">Link</a></span>')
    parts.append('<span class="ltx_bibblock ltx_bib_cited">Cited by: <a href="#S1">§1</a>.</span>')
    return f'<li id="bib.bib{n}" class="ltx_bibitem">{"".join(parts)}</li>'


def page(*items, figures=''):
    return (
        '<html><body><div class="ltx_page_main">'
        + figures
        + '<ul class="ltx_biblist">'
        + ''.join(items)
        + '</ul></div></body></html>'
    )


def png(w=600, h=400):
    from PIL import Image

    out = io.BytesIO()
    Image.new('RGBA', (w, h), (200, 30, 30, 128)).save(out, 'PNG')
    return out.getvalue()


def pdf_bytes():
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument.new()
    doc.new_page(612, 792)
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


def test_parse_splits_references_and_finds_their_titles():
    html = page(
        bibitem(1, 'Attention is all you need', link='https://arxiv.org/abs/1706.03762'),
        bibitem(2, blocks=['C. Writer and D. Writer', 'A hand-written reference title', 'In Some Venue, 2020']),
    )
    a, b = refs.parse(html)
    assert a['title'] == 'Attention is all you need' and 'https://arxiv.org/abs/1706.03762' in a['links']
    assert 'Cited by' not in a['raw'] and '§1' not in a['raw']
    assert refs.candidate_titles(b) == ['A hand-written reference title', 'In Some Venue, 2020']
    assert refs.norm('Über-Fast  GANs: a Study!') == 'uber fast gans a study'


def test_parse_pdf_reads_numbered_lists_only():
    body = '\n'.join(f'[{i}] A. Person. Some paper number {i}. In Venue, 2020.' for i in range(1, 6))
    got = refs.parse_pdf('1 Introduction\nblah blah\nReferences\n' + body)
    assert len(got) == 5 and got[0]['raw'] == 'A. Person. Some paper number 1. In Venue, 2020.'
    assert refs.parse_pdf('References\nSmith, J. (2020). A paper. Journal.\nDoe, A. (2019). Another.') == []
    assert refs.parse_pdf('no heading at all [1] x [2] y [3] z') == []


def test_references_are_matched_by_id_title_table_and_our_titles(corpus):
    pid, paper = next(iter(corpus.items()))
    with db.get_papers_db() as pdb, db.get_refs_db() as rdb:
        rdb.put_titles([(refs.title_hash('A Quantum Paper In Another Category'), '2101.00001')])
        m = refs.Matcher(pdb, rdb)
        assert m.match({'links': ['https://arxiv.org/abs/2401.12345v2'], 'raw': ''}) == ('2401.12345', 'id')
        assert m.match({'title': 'A quantum paper in another category', 'raw': ''}) == ('2101.00001', 'title')
        assert m.match({'title': paper['title'].lower(), 'raw': ''}) == (pid, 'title')  # ours, not in the table
        raw = f"[3] Someone. {paper['title']}. In Proceedings of Things, 2021."
        assert m.match({'raw': raw, 'links': [], 'blocks': []}) == (pid, 'text')
        # a PDF reference: no labels, but one piece of it is a title from the table
        raw = 'J. Doe and K. Roe. A quantum paper in another category. Phys. Rev. A, 2019.'
        assert m.match({'raw': raw, 'links': [], 'blocks': []}) == ('2101.00001', 'text')
        assert m.match({'title': 'Nothing like this exists anywhere', 'raw': 'x'}) == (None, None)


def test_the_figure_thumbnail_is_the_first_real_figure():
    grid = ''.join(f'<img class="ltx_graphics" src="p{i}.png" width="86" height="86">' for i in range(4))
    figs = (
        '<figure class="ltx_table"><img class="ltx_graphics" src="t.png" width="900"></figure>'
        f'<figure class="ltx_figure">{grid}</figure>'  # a teaser grid of small panels: skipped
        '<figure class="ltx_figure"><img class="ltx_graphics" src="2601.00001v1/x1.png" width="640"></figure>'
        '<figure class="ltx_figure"><img class="ltx_graphics" src="https://evil.example/x.png" width="640"></figure>'
    )
    urls = thumbs.figure_urls(page(figures=figs), 'https://arxiv.org/html/2601.00001v1')
    assert urls == ['https://arxiv.org/html/2601.00001v1/x1.png']  # nothing from other hosts
    from PIL import Image

    for made in (thumbs.from_image(png()), thumbs.from_pdf(pdf_bytes())):
        img = Image.open(io.BytesIO(made))
        assert img.format == 'JPEG' and img.width <= thumbs.SIZE[0] and img.height <= thumbs.SIZE[1]


def test_licenses():
    code = thumbs.license_code
    assert code('http://creativecommons.org/licenses/by/4.0/') == 'by/4.0'
    assert code('http://creativecommons.org/licenses/by-nc-nd/4.0/') == 'by-nc-nd/4.0'
    assert code('http://creativecommons.org/publicdomain/zero/1.0/') == 'zero/1.0'
    assert code('http://creativecommons.org/licenses/publicdomain/') == 'publicdomain'
    assert code('http://arxiv.org/licenses/nonexclusive-distrib/1.0/') == 'arxiv'
    assert code(None) == ''
    assert [thumbs.is_open(c) for c in ('by/4.0', 'zero/1.0', 'arxiv', '')] == [True, True, False, False]
    assert thumbs.license_label('by-nc-sa/4.0') == 'CC BY-NC-SA 4.0' and thumbs.license_label('zero/1.0') == 'CC0 1.0'
    page_ = '<div class="abs-license"><a href="http://creativecommons.org/licenses/by-sa/4.0/" title="Rights">'
    assert thumbs.abs_license(page_) == 'by-sa/4.0' and thumbs.abs_license('<html></html>') == ''


def fake_arxiv(monkeypatch, site):
    """arXiv replaced by site {url: body}: None is a 404, an exception is raised"""
    import enrich_daemon

    asked = []

    def get(self, url, max_bytes, timeout=90):
        asked.append(url)
        body = site.get(url)
        if isinstance(body, Exception):
            raise body
        if body is None:
            return None, b''
        # arXiv redirects /html/<id> to the latest version
        final = url + 'v1' if url.startswith('https://arxiv.org/html/') and url.count('/') == 4 else url
        return final, body if isinstance(body, bytes) else body.encode()

    monkeypatch.setattr(enrich_daemon.Fetcher, 'get', get)
    return asked


def run_daemon(monkeypatch):
    import enrich_daemon

    monkeypatch.setattr(sys, 'argv', ['enrich_daemon.py', '--once'])
    enrich_daemon.main()


def test_saved_papers_get_references_and_thumbnails_with_fallbacks(client, corpus, monkeypatch):
    import enrich_daemon

    pids = sorted(corpus)
    with_html, without_html, default_license = pids[:3]
    cited = pids[10:13]
    login(client)
    for p in (with_html, without_html, default_license):
        client.post(f'/add/{p}/cv')
    r = client.get('/?rank=cited&tags=cv&format=json').get_json()
    assert "aren't known yet" in r['notice']  # nothing fetched yet: say so

    with db.get_refs_db() as rdb:  # from the snapshot (import_snapshot.py --titles); without_html is too new
        rdb.put_licenses([(with_html, 'by/4.0'), (default_license, 'arxiv')])
    titles = [corpus[p]['title'] for p in cited]
    fig = f'<figure class="ltx_figure"><img class="ltx_graphics" src="{with_html}v1/x1.png" width="640"></figure>'
    site = {
        f'https://arxiv.org/html/{with_html}': page(*(bibitem(i, t) for i, t in enumerate(titles)), figures=fig),
        f'https://arxiv.org/html/{with_html}v1/x1.png': png(),
        f'https://arxiv.org/pdf/{with_html}': pdf_bytes(),
        # not in the snapshot: the license from the abstract page; no HTML version: the PDF stands in
        f'https://arxiv.org/abs/{without_html}': '<div class="abs-license"><a href="http://creativecommons.org/licenses/by-sa/4.0/">',
        f'https://arxiv.org/html/{without_html}': None,
        f'https://arxiv.org/pdf/{without_html}': pdf_bytes(),
        # arXiv's default license: references only, no figure, no PDF
        f'https://arxiv.org/html/{default_license}': page(
            bibitem(0, titles[2]), figures=fig.replace(with_html, default_license)
        ),
    }
    asked = fake_arxiv(monkeypatch, site)
    pdf_refs = '\n'.join(
        f'[{i}] Some One. {t}. In Venue, 2021.' for i, t in enumerate([titles[0], 'x y z', 'u v w'], 1)
    )
    monkeypatch.setattr(enrich_daemon.thumbs, 'pdf_text', lambda data: 'References\n' + pdf_refs)
    run_daemon(monkeypatch)
    assert sorted(asked) == sorted(site)  # each page once, and nothing else
    with db.get_refs_db() as rdb:
        done = {p: s[0] for p, s in rdb.fetched().items()}
        c = rdb.counts()
    assert done == {with_html: 'ok', without_html: 'ok', default_license: 'ok'}
    assert (c['html'], c['pdf'], c['figure'], c['page']) == (2, 1, 1, 2)
    asked.clear()
    run_daemon(monkeypatch)
    assert asked == []  # done is done

    got = client.get('/?rank=cited&tags=cv&format=json').get_json()
    assert [(p['id'], p['weight']) for p in got['papers']][:2] == [(cited[0], 2), (cited[2], 2)]
    assert {p['id'] for p in got['papers']} == set(cited)
    assert 'From the references of 3 of the 3 papers' in got['notice']
    assert b'Cited (3)' in client.get('/?rank=saved&tags=cv').data

    # cards show the figure, or the first page where there is no figure, with the license
    saved = {p['id']: p for p in client.get('/?rank=saved&tags=cv&format=json').get_json()['papers']}
    assert saved[with_html]['thumb_url'] == f'/thumb/figure/{with_html}'
    assert saved[with_html]['thumb_license'] == 'CC BY 4.0'
    assert saved[without_html]['thumb_url'] == f'/thumb/page/{without_html}'
    assert saved[without_html]['thumb_license'] == 'CC BY-SA 4.0'
    assert saved[default_license]['thumb_url'] == ''
    r = client.get(saved[with_html]['thumb_url'])
    assert r.status_code == 200 and r.mimetype == 'image/jpeg' and r.data[:2] == b'\xff\xd8'
    assert client.get('/thumb/figure/..%2F..%2Fusers.sqlite').status_code == 404
    # an image made under a license that doesn't allow showing it is never served
    db.save_thumb(default_license, 'page', thumbs.from_pdf(pdf_bytes()))
    assert client.get(f'/thumb/page/{default_license}').status_code == 404


def test_failures_are_retried_a_day_later_at_most_three_times(client, corpus, monkeypatch):
    import enrich_daemon

    pid = sorted(corpus)[0]
    login(client)
    client.post(f'/add/{pid}/cv')
    fake_arxiv(monkeypatch, {f'https://arxiv.org/html/{pid}': urllib.error.URLError('timed out')})
    run_daemon(monkeypatch)
    with db.get_refs_db() as rdb, db.get_papers_db() as pdb:
        assert rdb.fetched()[pid][0] == 'error'
        assert enrich_daemon.queue(pdb, rdb) == []  # not right away
        rdb.con.execute('UPDATE fetched SET t = 0')
        rdb.con.commit()
        assert enrich_daemon.queue(pdb, rdb) == [pid]  # a day later
        rdb.con.execute('UPDATE fetched SET tries = 3')
        rdb.con.commit()
        assert enrich_daemon.queue(pdb, rdb) == []  # three times is enough


def test_references_matched_later(corpus):
    import enrich_daemon

    with db.get_papers_db() as pdb, db.get_refs_db() as rdb:
        rdb.put_refs(
            '2601.99999', 'ok', [(0, None, 'Some Physics Paper Title', None, '[1] X. Some Physics Paper Title.')]
        )
        m = refs.Matcher(pdb, rdb)
        assert enrich_daemon.rematch(rdb, m) == 0
        rdb.put_titles([(refs.title_hash('Some physics paper title'), '1901.00002')])  # --titles ran since
        assert enrich_daemon.rematch(rdb, m) == 1
        assert rdb.cited_by(['2601.99999'])[0] == {'1901.00002': {'2601.99999'}}
