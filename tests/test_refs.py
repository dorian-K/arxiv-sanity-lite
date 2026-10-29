"""references from arXiv's HTML pages (aslite/refs.py, refs_daemon.py) and the "Cited" tab"""

import contextlib
import sys

from conftest import login
from aslite import db, refs


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


def page(*items):
    return '<html><body><ul class="ltx_biblist">' + ''.join(items) + '</ul></body></html>'


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
        assert m.match({'title': 'Nothing like this exists anywhere', 'raw': 'x'}) == (None, None)


def run_daemon(monkeypatch, pages):
    """refs_daemon.py --once with arXiv replaced by pages {pid: html or None (no HTML version)}"""
    import refs_daemon

    asked = []

    def fetch(pid):
        asked.append(pid)
        return ('ok', pages[pid]) if pages.get(pid) else ('missing', '')

    monkeypatch.setattr(refs_daemon, 'fetch_html', fetch)
    monkeypatch.setattr(refs_daemon, 'PAUSE', 0)
    monkeypatch.setattr(refs_daemon.db, 'arxiv_slot', contextlib.nullcontext)
    monkeypatch.setattr(sys, 'argv', ['refs_daemon.py', '--once', '--pause', '0'])
    refs_daemon.main()
    return asked


def test_daemon_reads_saved_papers_once_and_the_cited_tab_shows_what_they_cite(client, corpus, monkeypatch):
    pids = sorted(corpus)
    saved, cited = pids[:2], pids[10:13]
    login(client)
    for p in saved:
        client.post(f'/add/{p}/cv')
    r = client.get('/?rank=cited&tags=cv&format=json').get_json()
    assert "aren't known yet" in r['notice']  # nothing fetched yet: say so

    titles = [corpus[p]['title'] for p in cited]
    pages = {
        saved[0]: page(*(bibitem(i, t) for i, t in enumerate(titles)), bibitem(9, 'Not on this site at all')),
        saved[1]: page(bibitem(0, titles[0]), bibitem(1, link=f'https://arxiv.org/abs/{saved[0]}')),
    }
    asked = run_daemon(monkeypatch, pages)
    assert sorted(asked) == sorted(saved)
    with db.get_refs_db() as rdb:
        assert {p: s[0] for p, s in rdb.fetched().items()} == {saved[0]: 'ok', saved[1]: 'ok'}
    assert run_daemon(monkeypatch, pages) == []  # each paper once

    page_ = client.get('/?rank=cited&tags=cv&format=json').get_json()
    got = [(p['id'], p['weight']) for p in page_['papers']]
    assert got[0] == (cited[0], 2)  # cited by both saved papers first
    assert {p for p, _ in got} == set(cited)  # saved[0] is cited too, but it's saved: left out
    assert 'From the references of 2 of the 2 papers' in page_['notice']
    assert b'Cited (3)' in client.get('/?rank=saved&tags=cv').data


def test_papers_without_html_are_not_asked_again_and_failures_are_retried_later(client, corpus, monkeypatch):
    import refs_daemon

    pid = sorted(corpus)[0]
    login(client)
    client.post(f'/add/{pid}/cv')
    assert run_daemon(monkeypatch, {}) == [pid]  # no HTML version
    assert run_daemon(monkeypatch, {}) == []
    with db.get_refs_db() as rdb:
        rdb.con.execute("UPDATE fetched SET status = 'error', t = 0")
        rdb.con.commit()
        pdb = db.get_papers_db()
        assert refs_daemon.queue(pdb, rdb) == [pid]  # a day later
        rdb.con.execute('UPDATE fetched SET tries = 3')
        rdb.con.commit()
        assert refs_daemon.queue(pdb, rdb) == []  # three times is enough


def test_references_matched_later(corpus):
    import refs_daemon

    with db.get_papers_db() as pdb, db.get_refs_db() as rdb:
        rdb.put_refs(
            '2601.99999', 'ok', [(0, None, 'Some Physics Paper Title', None, '[1] X. Some Physics Paper Title.')]
        )
        m = refs.Matcher(pdb, rdb)
        assert refs_daemon.rematch(rdb, m) == 0
        rdb.put_titles([(refs.title_hash('Some physics paper title'), '1901.00002')])  # --titles ran since
        assert refs_daemon.rematch(rdb, m) == 1
        assert rdb.cited_by(['2601.99999'])[0] == {'1901.00002': {'2601.99999'}}
