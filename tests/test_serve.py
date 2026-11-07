import os

from conftest import login


def test_pages_render(client, corpus):
    # every page is the app's html; its data comes from /ui/
    for url in ['/', '/?rank=random', '/about', '/stats', '/profile', '/digest', '/import', '/agents', '/inspect']:
        r = client.get(url)
        assert r.status_code == 200 and b'type="importmap"' in r.data, url
    for url in ['/ui/me', '/ui/list', '/ui/list?rank=random', '/ui/list?q=convolution', '/ui/stats', '/ui/profile']:
        assert client.get(url).status_code == 200, url
    for name in ('about', 'privacy', 'imprint'):
        assert '<h1>' in client.get(f'/ui/page/{name}').get_json()['html']
    pid = next(iter(corpus))
    assert client.get(f'/ui/inspect?pid={pid}').get_json()['paper']['id'] == pid


def test_the_app_shell(client):
    import re

    r = client.get('/?rank=time')
    page = r.data.decode()
    # every module in the import map carries its mtime, so browsers can keep them for good
    for url in re.findall(r'"(/static/app/[^"]+)"', page):
        assert '?v=' in url, url
    assert r.headers['Cache-Control'] == 'no-cache'
    # the one inline script, the import map, is allowed by its hash and nothing else inline is
    csp = r.headers['Content-Security-Policy']
    assert "script-src 'self' 'sha256-" in csp and 'unsafe-inline' not in csp.split('style-src')[0]
    assert client.get('/ui/nothing').status_code == 404


def test_tag_rank_and_downvote(client, corpus):
    login(client)
    vision = [p for p in corpus if corpus[p]['tags'][0]['term'] == 'cs.CV']
    for p in vision[:4]:
        assert client.post(f'/add/{p}/cv').data.startswith(b'ok')
    assert client.post(f'/addneg/{vision[4]}/cv').data.startswith(b'ok')

    for model in ['svm', 'logreg', 'lda']:
        res = client.get(f'/ui/list?rank=tags&tags=cv&model={model}').get_json()
        assert res['papers'] and res['papers'][0]['id'].startswith('2601.')

    # liking a downvoted paper moves it from neg_tags to tags
    client.post(f'/add/{vision[4]}/cv')
    from aslite.db import get_tags

    assert vision[4] in get_tags('alice')['cv']
    assert vision[4] not in get_tags('alice', neg=True).get('cv', set())


def test_passwords(client):
    assert login(client, password='short').status_code == 400  # too short to sign up
    assert login(client).status_code == 302  # signs up
    client.post('/logout')
    assert login(client, password='wrong password').status_code == 403
    assert client.post('/add/2601.00001/x').data.startswith(b'error, not logged in')
    assert login(client).status_code == 302
    assert client.post('/add/2601.00001/x').data.startswith(b'ok')
    assert login(client, user='bad name!').status_code == 400


def test_sessions_end_when_the_password_changes(client):
    from werkzeug.security import generate_password_hash

    from aslite import db

    login(client)
    assert client.post('/add/2601.00001/x').data.startswith(b'ok')
    db.set_password('alice', generate_password_hash('another password'))  # e.g. flask set-password
    assert client.post('/add/2601.00001/x').data.startswith(b'error, not logged in')


def test_accounts_without_password_cant_be_claimed(client):
    from aslite import db

    db.tag_paper('legacy', '2601.00001', 'x')  # from before passwords existed
    assert login(client, user='legacy').status_code == 403
    assert client.post('/add/2601.00001/x').data.startswith(b'error, not logged in')


def test_tag_names_and_query_params_are_checked(client, corpus):
    login(client)
    for bad in ('a,b', 'tab%09here'):
        assert client.post(f'/add/2601.00001/{bad}').data.startswith(b'error')
    for q in ('svm_c=1e300', 'svm_c=nan', 'rank=evil', 'time_filter=-5', 'time_filter=1e9'):
        assert client.get('/ui/list?' + q).status_code == 200


def test_import_suggests_existing_tags(client, corpus):
    login(client)
    a, b = list(corpus)[:2]
    client.post(f'/add/{a}/cv')
    client.post(f'/addneg/{b}/nlp')
    assert client.get('/ui/import').get_json()['user_tags'] == ['cv', 'nlp']


def test_state_changes_need_same_site_posts(client):
    login(client)
    assert client.get('/add/2601.00001/x').status_code == 405  # no more GET <img src=...> tricks
    r = client.post('/add/2601.00001/x', headers={'Sec-Fetch-Site': 'cross-site'})
    assert r.status_code == 403
    assert client.post('/add/2601.00001/x', headers={'Sec-Fetch-Site': 'same-origin'}).data.startswith(b'ok')


def test_closed_signups(client, monkeypatch):
    import serve

    login(client)  # alice signs up while signups are open
    client.post('/logout')
    monkeypatch.setattr(serve, 'ALLOW_SIGNUP', False)
    assert login(client, user='bob').status_code == 403
    assert login(client).status_code == 302  # existing accounts still work


def test_search_uses_the_index(client, corpus):
    from aslite.db import get_papers_db

    search_papers = get_papers_db().search
    pids, scores = search_papers('vision convol')  # prefix match
    assert pids and scores == sorted(scores, reverse=True)
    assert all(corpus[p]['tags'][0]['term'] == 'cs.CV' for p in pids)
    assert search_papers('"; DROP TABLE papers; --') == ([], [])  # user input never reaches FTS syntax
    assert b'2601.' in client.get('/ui/list?q=vision').data


def test_import_papers(client, corpus, monkeypatch):
    import random

    import serve
    from aslite import db
    from conftest import make_paper

    old = make_paper(1, 'vision', random.Random(1))
    old.update(_id='1706.03762', _idv='1706.03762v7', title='an old paper about attention')
    calls = []

    def fake_fetch(ids):
        calls.append(list(ids))
        return {i: old for i in ids if i == '1706.03762'}

    from aslite import sources

    monkeypatch.setattr(sources.ARXIV, 'fetch', fake_fetch)
    monkeypatch.setattr(db, 'arxiv_slot', lambda: __import__('contextlib').nullcontext())
    have = next(iter(corpus))
    text = '\n'.join(
        [
            f'https://arxiv.org/abs/{have}v1',  # already here
            'https://arxiv.org/pdf/1706.03762v7.pdf',  # fetched, outside any date window
            '2501.99999',  # arXiv doesn't know it
            'hep-th/0408218',
            'just some words',
            '',
        ]
    )
    assert client.post('/import', data={'text': text}).status_code == 403  # logged out: nothing happens
    assert calls == []

    login(client)
    r = client.post('/import', data={'text': text, 'tag': 'reading'})
    assert r.status_code == 200
    page = r.data.decode()
    for s in ['already here', 'added from arXiv', 'not found on arXiv', 'old-style', 'no paper id or link found']:
        assert s in page, s
    assert calls == [['1706.03762', '2501.99999']]  # one api call for everything missing
    assert db.get_tags('alice')['reading'] == {have, '1706.03762'}
    assert os.path.exists(db.COMPUTE_FLAG_FILE)
    assert db.get_papers_db()['1706.03762']['title'] == 'an old paper about attention'
    assert b'1706.03762' in client.get('/ui/list?q=arxiv.org/abs/1706.03762').data

    # fetches count against the daily limit; papers we already have don't
    monkeypatch.setattr(serve, 'FETCH_LIMIT_USER', 3)
    r = client.post('/import', data={'text': '2502.00001\n2502.00002\n1706.03762', 'tag': 'reading'})
    assert 'over the daily fetch limit' in r.data.decode()
    assert calls[-1] == ['2502.00001']
    assert '2502.00002' not in db.get_tags('alice')['reading']  # not fetched: not tagged either


ACL_PAGE = """<html><head>
<meta content="A Benchmark of Spoken &amp; Written Errors" name=citation_title>
<meta content="Ada Lovelace" name=citation_author><meta content="Alan Turing" name=citation_author>
<meta content="Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing" name=citation_conference_title>
<meta content="2026/10" name=citation_publication_date></head><body>
<div class="card-body acl-abstract"><h5 class=card-title>Abstract</h5><span>We study <i>errors</i> in speech.</span></div>
<dl><dt>Cite (Informal):</dt><dd><span id=citeRichText><a href=x>A Benchmark</a> (Lovelace &amp; Turing, EMNLP 2026)</span></dd></dl>
</body></html>"""


def test_papers_from_the_acl_anthology(client, corpus, monkeypatch):
    """a paper that isn't on arXiv: added by its Anthology link, then shown with its own links everywhere"""
    import contextlib

    from aslite import db, sources

    pid = '2026.emnlp-main.2349'
    assert sources.find_paper_id(f'https://aclanthology.org/{pid}/') == pid
    assert sources.find_paper_id(f'https://aclanthology.org/{pid}.pdf') == pid
    assert sources.find_paper_id('https://doi.org/10.18653/v1/2020.acl-main.1') == '2020.acl-main.1'
    assert sources.find_paper_id('https://www.aclweb.org/anthology/p19-1001.pdf') == 'P19-1001'
    assert sources.find_paper_id('https://arxiv.org/abs/1706.03762v7') == '1706.03762'
    assert sources.find_paper_id('see 2401.12345') == '2401.12345'
    assert sources.source_of(pid) is sources.ACL and sources.source_of('2401.12345') is sources.ARXIV
    assert (sources.year(pid), sources.year('P15-1001'), sources.year('2401.12345')) == (2026, 2015, 2024)
    assert sources.ACL.license('P15-1001') == 'by-nc-sa/3.0' and sources.ACL.license(pid) == 'by/4.0'

    paper = sources.parse_acl_page(pid, ACL_PAGE)
    assert paper['title'] == 'A Benchmark of Spoken & Written Errors'
    assert [a['name'] for a in paper['authors']] == ['Ada Lovelace', 'Alan Turing']
    assert paper['summary'] == 'We study errors in speech.'
    assert paper['arxiv_journal_ref'] == 'EMNLP 2026' and paper['_time_str'] == 'Oct 01 2026'
    assert sources.parse_acl_page(pid, '<html>404</html>') is None

    calls = []
    monkeypatch.setattr(sources.ACL, 'fetch', lambda ids: calls.append(ids) or {i: paper for i in ids if i == pid})
    monkeypatch.setattr(sources.ACL, 'slot', lambda: contextlib.nullcontext())
    login(client)
    r = client.post('/import', data={'text': f'https://aclanthology.org/{pid}/\n2026.acl-long.9999', 'tag': 'speech'})
    got = {x['line']: x['status'] for x in r.get_json()['results']}
    assert got == {
        f'https://aclanthology.org/{pid}/': 'added from the ACL Anthology',
        '2026.acl-long.9999': 'not found on the ACL Anthology',
    }
    assert calls == [[pid], ['2026.acl-long.9999']]  # one request per paper, each in the Anthology's slot
    assert db.get_tags('alice')['speech'] == {pid}
    card = next(p for p in client.get('/ui/list?rank=saved&tags=speech').get_json()['papers'] if p['id'] == pid)
    assert card['url'] == f'https://aclanthology.org/{pid}/' and card['pdf'] == f'https://aclanthology.org/{pid}.pdf'
    assert card['venue'] == 'EMNLP 2026' and card['venue_note'] == 'from the ACL Anthology'
    node = client.get(f'/ui/graph?pid={pid}').get_json()['nodes'][0]
    assert (node['year'], node['url'], node['here']) == ('2026', card['url'], True)
    assert client.get(f'/ui/list?q=aclanthology.org/{pid}').get_json()['papers'][0]['id'] == pid


def test_rate_limits(client, corpus, monkeypatch):
    import serve

    monkeypatch.setattr(
        serve, 'RATE_LIMITS', {'rank': (2, 60), 'search': (3, 60), 'login': (2, 300), 'import': (1, 3600)}
    )
    pids = list(corpus)
    # distinct queries: a repeated one comes from the cache and costs nothing
    assert [client.get(f'/ui/list?rank=pid&pid={p}').status_code for p in pids[:3]] == [200, 200, 429]
    assert client.get(f'/ui/list?rank=pid&pid={pids[0]}').status_code == 200  # cached
    assert [client.get(f'/ui/list?q=vision{i}').status_code for i in range(4)] == [200, 200, 200, 429]
    assert client.get('/ui/list').status_code == 200  # cheap pages are never limited
    login(client)  # signs up; the login limit counts this attempt too
    assert login(client, password='wrong password').status_code == 403
    assert login(client, password='wrong password').status_code == 429
    # logged in, a client is the user rather than the ip: a fresh budget, then the limit again
    codes = [client.get(f'/ui/list?rank=pid&pid={p}').status_code for p in pids[3:6]]
    assert codes == [200, 200, 429]
    r = client.get(f'/ui/list?rank=pid&pid={pids[6]}')
    assert r.status_code == 429 and int(r.headers['Retry-After']) >= 1


def test_read_only_view_of_shared_accounts(client, corpus):
    from aslite import db

    pid = next(iter(corpus))
    login(client)
    client.post(f'/add/{pid}/cv')
    client.post('/register_email', data={'email': 'alice@example.org'})
    client.post('/logout')

    # off by default: an empty password gets nowhere
    assert login(client, password='').status_code == 403
    login(client)
    client.post('/sharing', data={'public': '1'})
    assert db.get_user('alice')['public'] == 1
    client.post('/logout')

    r = login(client, password='')
    assert r.status_code == 302
    assert client.get('/ui/me').get_json()['readonly'] is True
    assert client.get('/ui/list?rank=tags&tags=cv').status_code == 200
    assert b'alice@example.org' not in client.get('/ui/profile').data  # the email stays private
    for url in [f'/add/{pid}/x', f'/sub/{pid}/cv', f'/addneg/{pid}/cv', '/del/cv', '/import', '/digest', '/sharing']:
        assert client.post(url, data={'text': '2501.00001', 'public': '0', 'email': ''}).status_code == 403, url
    assert db.get_tags('alice') == {'cv': {pid}} and db.get_user('alice')['public'] == 1

    # the owner turns sharing off: the read-only session ends on its next request
    db.set_public('alice', False)
    assert client.get('/ui/me').get_json()['user'] == ''
    assert client.post(f'/add/{pid}/x').data.startswith(b'error, not logged in')
    # and a wrong password is still wrong, sharing or not
    db.set_public('alice', True)
    assert login(client, password='wrong password').status_code == 403


def test_tag_check_and_rank_percentiles(client, corpus):
    vision = [p for p in corpus if corpus[p]['tags'][0]['term'] == 'cs.CV']
    language = [p for p in corpus if corpus[p]['tags'][0]['term'] == 'cs.CL']
    login(client)
    for p in vision[:8] + language[:1]:  # one paper in the wrong tag
        client.post(f'/add/{p}/cv')
    r = client.get('/tag_check?tag=cv').get_json()
    assert r['n'] == 9 and set(r['ranks']) == set(vision[:8] + language[:1])
    assert r['median'] < 0.2  # the vision papers predict each other (1/6 of this corpus is vision; random would be 0.5)
    assert [m['id'] for m in r['misfits']][:1] == [language[0]]  # and the odd one out is flagged first
    assert client.get('/tag_check?tag=nope').status_code == 404
    # rankings carry each paper's rank among all papers
    res = client.get('/ui/list?rank=tags&tags=cv').get_json()
    assert all(0 < p['pct'] <= 1 for p in res['papers'])


def test_tag_pages_and_landing(client, corpus):
    from aslite import db

    login(client)
    assert client.get('/ui/list').get_json()['gvars']['rank'] == 'time'  # nothing saved yet: the newest papers
    vision = [p for p in corpus if corpus[p]['tags'][0]['term'] == 'cs.CV']
    for p in vision[:3]:
        client.post(f'/add/{p}/cv')
    # with tags, the main page is the best recent papers for them
    res = client.get('/ui/list').get_json()
    g = res['gvars']
    assert (g['rank'], g['tags'], g['time_filter']) == ('tags', 'all', '30')
    assert all(p.get('pct') for p in res['papers']) and res['papers']
    # page 2 is /?page_number=2: still "For you" (it once fell back to the newest papers, which
    # looked like random recommendations)
    import serve

    assert client.get('/ui/list?page_number=2').get_json()['gvars']['rank'] == 'tags'
    assert serve.FOR_YOU_DAYS == 30
    # every tag has a page with its saved papers, newest first
    res = client.get('/ui/list?rank=saved&tags=cv').get_json()
    assert res['tag_view'] and res['n_saved'] == 3
    times = db.get_papers_db().times()
    ids = [p['id'] for p in res['papers']]
    assert ids == sorted(vision[:3], key=times.get, reverse=True)


def test_unindexed_papers_are_explained(client, corpus):
    import random

    from aslite import db
    from conftest import make_paper

    login(client)
    new = make_paper(999, 'vision', random.Random(5))
    new.update(_id='2602.99999', _idv='2602.99999v1')
    db.get_papers_db().upsert([new])  # added after compute.py ran: not in the features yet
    client.post('/add/2602.99999/fresh')
    res = client.get('/ui/list?rank=tags&tags=fresh').get_json()
    assert res['papers'] == [] and 'added recently' in res['notice']
    res = client.get('/ui/list?rank=pid&pid=2602.99999').get_json()
    assert 'added recently' in res['notice']


def test_import_needs_a_tag(client, corpus):
    login(client)
    r = client.post('/import', data={'text': next(iter(corpus))})
    assert r.status_code == 400 and b'Pick a tag' in r.data


def test_for_you_can_let_each_tag_pick_in_turns(client, corpus):
    login(client)
    vision = [p for p in corpus if corpus[p]['tags'][0]['term'] == 'cs.CV']
    language = [p for p in corpus if corpus[p]['tags'][0]['term'] != 'cs.CV']
    for p in vision[:4]:
        client.post(f'/add/{p}/cv')
    for p in language[:4]:
        client.post(f'/add/{p}/nlp')
    # the default: one model on all tags together, no per-tag labels
    papers = client.get('/ui/list?rank=tags&tags=all&skip_have=yes').get_json()['papers']
    assert papers and not any(p.get('for_tag') for p in papers)
    papers = client.get('/ui/list?rank=tags&tags=all&skip_have=yes&mix=turns').get_json()['papers']
    # one model per tag, taking turns: every paper says which tag picked it, and both get a say
    assert [p['for_tag'] for p in papers[:4]] == ['cv', 'nlp', 'cv', 'nlp']
    assert all(corpus[p['id']]['tags'][0]['term'] == 'cs.CV' for p in papers[:6] if p['for_tag'] == 'cv')
    assert len({p['id'] for p in papers}) == len(papers)
    # the C used by default is shown
    from aslite.rank import DEFAULT_C

    g = client.get('/ui/list?rank=tags&tags=cv&model=logreg&feats=tfidf').get_json()['gvars']
    assert g['default_c'] == f'{DEFAULT_C[("logreg", "tfidf")]:g}'


def test_downvotes_are_visible_and_only_hide_for_their_tag(client, corpus):
    login(client)
    vision = [p for p in corpus if corpus[p]['tags'][0]['term'] == 'cs.CV']
    for p in vision[:4]:
        client.post(f'/add/{p}/cv')
        client.post(f'/add/{p}/images')
    bad = vision[5]
    client.post(f'/addneg/{bad}/cv')
    # the tag page lists its downvotes in their own tab
    res = client.get('/ui/list?rank=down&tags=cv').get_json()
    assert res['n_down'] == 1 and [p['id'] for p in res['papers']] == [bad]
    assert client.get('/ui/list?rank=down&tags=images').get_json()['papers'] == []
    # hidden from the tag it was marked for, but still recommended for the other one
    ids = lambda tag: [p['id'] for p in client.get(f'/ui/list?rank=tags&tags={tag}&skip_have=yes').get_json()['papers']]
    assert bad not in ids('cv') and bad in ids('images')
    assert not set(vision[:4]) & set(ids('images'))  # saved papers stay hidden everywhere


def test_shared_links(client, corpus):
    from aslite import db

    vision = [p for p in corpus if corpus[p]['tags'][0]['term'] == 'cs.CV']
    login(client, user='bob')
    for p in vision[:4]:
        client.post(f'/add/{p}/cv')
    client.post(f'/addneg/{vision[6]}/cv')
    client.post('/logout')
    # not shared: a link shows nothing of bob's
    res = client.get('/ui/list?u=bob&rank=saved&tags=cv').get_json()
    assert res['not_shared'] == 'bob' and res['papers'] == []
    db.set_public('bob', True)
    # shared: anyone with the link sees bob's tag, read-only, no login needed
    res = client.get('/ui/list?u=bob&rank=saved&tags=cv').get_json()
    assert res['viewing'] and res['owner'] == 'bob' and vision[0] in [p['id'] for p in res['papers']]
    assert client.get('/ui/list?u=bob&rank=tags&tags=cv').get_json()['papers']
    assert client.get('/tag_check?tag=cv&u=bob').status_code == 200
    assert client.get('/ui/digest?u=bob').get_json()['owner'] == 'bob'
    # a logged-in visitor can't vote on it, but can copy it into their own tags
    login(client)  # alice
    assert client.get('/ui/list?u=bob').get_json()['gvars']['tags'] == 'all'  # bob's "For you"
    r = client.post('/copy_tag', data={'owner': 'bob', 'tag': 'cv', 'as': 'bobs-cv'})
    assert r.status_code == 200 and 'bobs-cv' in r.get_json()['url']
    assert db.get_tags('alice')['bobs-cv'] == set(vision[:4])
    assert db.get_tags('alice', neg=True)['bobs-cv'] == {vision[6]}
    assert db.get_tags('bob')['cv'] == set(vision[:4])  # bob's own tag is untouched
    # writes never go to the owner, whatever the url says
    client.post(f'/add/{vision[5]}/cv?u=bob')
    assert vision[5] not in db.get_tags('bob')['cv']
    # sharing off again: links and copying stop at once
    db.set_public('bob', False)
    assert vision[0] not in client.get('/ui/list?u=bob&rank=saved&tags=cv').data.decode()
    assert client.post('/copy_tag', data={'owner': 'bob', 'tag': 'cv'}).status_code == 404


def test_search_ignores_the_pages_filters(client, corpus):
    login(client)
    pid = next(iter(corpus))
    client.post(f'/add/{pid}/cv')
    title = corpus[pid]['title']
    # the search box sends the page's filters along (here: those of "For you"); they don't apply
    r = client.get('/ui/list?' + 'q=' + title.split()[0] + '&rank=tags&time_filter=1&skip_have=yes')
    assert pid in r.data.decode()
    # unless search is picked explicitly
    r = client.get('/ui/list?q=' + title.split()[0] + '&rank=search&skip_have=yes')
    assert pid not in [p['id'] for p in r.get_json()['papers']]


def test_unknown_paper_and_odd_tag_names(client, corpus):
    login(client)
    assert 'no paper with this id' in client.get('/ui/list?rank=pid&pid=2999.99999').get_json()['notice']
    for bad in ('.', '..'):
        assert client.post('/import', data={'text': next(iter(corpus)), 'tag': bad}).status_code == 400


def test_errors_are_shown_by_the_app_or_as_pages(client, corpus):
    r = client.get('/ui/inspect?pid=2999.99999')
    assert r.status_code == 404 and r.get_json()['title'] == 'Unknown paper'
    # a plain form post (the oauth consent page's login) gets a page of the site
    r = client.post('/login', data={'username': 'bad name!', 'password': 'x'})
    assert r.status_code == 400 and b'<header' in r.data and b'Usernames are' in r.data
    r = client.post('/login', data={'username': 'bad name!', 'password': 'x'}, headers={'Accept': 'application/json'})
    assert r.status_code == 400 and 'Usernames are' in r.get_json()['error']


def test_newest_and_random_keep_to_the_time_window(client, corpus):
    # paper i came out i hours ago (conftest)
    for rank in ('time', 'random'):
        ids = [p['id'] for p in client.get(f'/ui/list?rank={rank}&time_filter=1').get_json()['papers']]
        assert ids and all(int(p.split('.')[1]) < 24 for p in ids), rank
    ids = [p['id'] for p in client.get('/ui/list?rank=time').get_json()['papers']]
    assert ids == sorted(ids)[:25]  # newest first


def test_static_files_are_versioned_and_cached(client):
    page = client.get('/about').get_data(as_text=True)
    assert '/static/app/lib.js?v=' in page and '/static/style.css?v=' in page
    assert 'immutable' in client.get('/static/app/lib.js?v=1').headers['Cache-Control']
    assert 'immutable' not in client.get('/static/app/lib.js').headers.get('Cache-Control', '')


def test_published_filter_and_venues(client, corpus):
    from aslite import db

    pids = sorted(corpus)
    with db.get_papers_db() as pdb:
        p = pdb[pids[3]]
        p['arxiv_comment'], p['_version'] = '8 pages. Accepted at NeurIPS 2026', 2
        pdb.upsert([p])
    for url in ('/?rank=time&published=yes', '/?rank=search&q=paper&published=yes'):
        ids = [x['id'] for x in client.get('/ui/list' + url[1:]).get_json()['papers']]
        assert ids == [pids[3]], url
    page = client.get('/ui/list?rank=time').get_json()['papers']
    assert next(x for x in page if x['id'] == pids[3])['venue'] == 'Accepted at NeurIPS 2026'
    # recommendations too
    login(client)
    for x in pids[:6]:
        client.post(f'/add/{x}/t')
    res = client.get('/ui/list?rank=tags&tags=t&published=yes').get_json()['papers']
    assert all(x['venue'] for x in res)


def test_old_papers_db_gets_the_new_columns(tmp_path, monkeypatch):
    """a papers.sqlite from before journal_ref/venue: both columns added, venues filled in once"""
    import sqlite3

    from aslite import db

    path = str(tmp_path / 'old.sqlite')
    con = sqlite3.connect(path)
    schema = db.PAPERS_SCHEMA.replace(
        ",\n    journal_ref TEXT NOT NULL DEFAULT '',  -- where it was published, if the authors said so\n    venue TEXT NOT NULL DEFAULT ''         -- aslite.arxiv.venue() of comment and journal_ref",
        '',
    )
    assert 'venue' not in schema.split('CREATE INDEX')[0]
    con.executescript(schema)
    con.execute(
        "INSERT INTO papers (id, version, time, time_str, published, title, summary, authors, categories, comment) "
        "VALUES ('2601.00001', 1, 0, '', '', 't', 's', '[]', '[]', 'Accepted to ACL 2026')"
    )
    con.commit()
    con.close()
    with db.get_papers_db(path) as pdb:
        assert pdb.venues() == {'2601.00001': 'Accepted to ACL 2026'}
        assert pdb.search('t')[0] == ['2601.00001']


def test_signup_asks_about_a_public_profile(client):
    from aslite import db

    asks = lambda: client.get('/ui/me').get_json()['ask_public']
    assert not asks()
    client.post('/login', data={'username': 'pub', 'password': 'correct horse'})
    assert db.get_user('pub')['public'] == 0 and asks()  # private until answered
    client.post('/sharing', data={'public': '1'})
    assert db.get_user('pub')['public'] == 1 and not asks()  # asked once
    client.post('/logout')
    client.post('/login', data={'username': 'priv', 'password': 'correct horse'})
    client.post('/sharing', data={})  # unticked
    assert db.get_user('priv')['public'] == 0
    client.post('/logout')
    client.post('/login', data={'username': 'priv', 'password': 'correct horse'})  # a login doesn't ask again
    assert not asks()


def test_search_fixes_typos_unless_asked_not_to(client, corpus, monkeypatch):
    import serve
    from aslite.spell import Speller

    assert serve.get_speller() is not None  # compute.py wrote the words
    pid, p = next(iter(corpus.items()))
    word = max(p['title'].lower().split(), key=len)
    monkeypatch.setattr(serve, 'get_speller', lambda: Speller({word: 100}))
    r = client.get('/ui/list?rank=search&q=' + word[0] + word[2:]).get_json()  # one letter missing
    assert r['gvars']['corrected'] == word and pid in [p['id'] for p in r['papers']]
    r = client.get('/ui/list?rank=search&exact=yes&q=' + word[0] + word[2:]).get_json()
    assert r['gvars']['corrected'] == ''


def test_stats_show_page_speed_and_background_work(client, corpus):
    from aslite import db

    login(client)
    vision = [p for p in corpus if corpus[p]['tags'][0]['term'] == 'cs.CV']
    for p in vision[:3]:
        client.post(f'/add/{p}/cv')
    client.get('/ui/list?rank=search&q=vision')
    client.get('/ui/list?rank=tags&tags=cv')
    db.put_job('embed', 12.0, papers=8, left=400)
    stats = client.get('/ui/stats').get_json()
    labels = [p['label'] for p in stats['pages']]
    for label in ('Search', 'Vote: save', 'Recommendations', 'Training a model (after a vote, or while a page waits)'):
        assert label in labels
    assert stats['computes'] and stats['computes'][0]['peak']  # conftest's compute.py run
    assert stats['embed']['left'] == 400 and stats['embed']['eta'] == '50.0 days'


def test_read_marks_and_the_to_read_tab(client, corpus):
    pids = sorted(corpus)[:3]
    login(client)
    for p in pids:
        client.post(f'/add/{p}/cv')
    client.post(f'/add/{pids[2]}/nlp')
    assert client.post(f'/read/{pids[0]}').data == b'ok'

    def to_read(tags):
        return {p['id'] for p in client.get(f'/ui/list?rank=unread&tags={tags}').get_json()['papers']}

    assert to_read('cv') == set(pids[1:]) and to_read('nlp') == {pids[2]} and to_read('all') == set(pids[1:])
    res = client.get('/ui/list?rank=saved&tags=cv').get_json()
    assert res['n_unread'] == 2 and res['n_saved'] == 3
    marks = {p['id']: p['read'] for p in client.get('/ui/list?rank=saved&tags=cv').get_json()['papers']}
    assert marks == {pids[0]: True, pids[1]: False, pids[2]: False}
    client.post(f'/read/{pids[2]}')
    assert to_read('all') == {pids[1]}
    client.post(f'/unread/{pids[0]}')
    assert to_read('cv') == {pids[0], pids[1]}
    assert client.post('/read/9999.99999').data.startswith(b'error')
    assert [r['pid'] for r in client.get('/export').get_json()['read']] == [pids[2]]
    client.post('/logout')
    assert client.post(f'/read/{pids[1]}').data.startswith(b'error, not logged in')


def test_old_users_db_gets_the_reads_table(tmp_path, monkeypatch):
    """a users.sqlite from before read marks: the table appears, the old tags stay"""
    import sqlite3

    from aslite import db

    path = str(tmp_path / 'users.sqlite')
    con = sqlite3.connect(path)
    con.executescript(db.USERS_SCHEMA + 'DROP TABLE reads;')
    assert 'reads' not in {r[0] for r in con.execute("SELECT name FROM sqlite_master")}
    con.execute("INSERT INTO tags (user, tag, pid, neg) VALUES ('alice', 'cv', '2601.00001', 0)")
    con.commit()
    con.close()
    monkeypatch.setattr(db, 'USERS_DB_FILE', path)
    assert db.get_reads('alice') == set()
    db.set_read('alice', '2601.00001')
    assert db.get_reads('alice') == {'2601.00001'} and db.get_tags('alice') == {'cv': {'2601.00001'}}


def test_removing_a_tags_last_saved_paper_deletes_the_tag(client, corpus):
    from aslite import db

    pids = sorted(corpus)
    login(client)
    client.post(f'/add/{pids[0]}/foo')
    client.post(f'/add/{pids[1]}/foo')
    client.post(f'/addneg/{pids[2]}/foo')
    db.set_digest_size('alice', 'foo', 3)
    client.post(f'/sub/{pids[0]}/foo')
    assert db.get_tags('alice') == {'foo': {pids[1]}}  # one left: still there
    client.post(f'/sub/{pids[1]}/foo')
    assert db.get_tags('alice') == {} and db.get_tags('alice', neg=True) == {} and db.get_digest_sizes('alice') == {}
    assert 'foo' not in client.get('/ui/import').get_json()['user_tags']


def test_health_not_counted(client, monkeypatch):
    import serve

    seen = []
    monkeypatch.setattr(serve.metrics, 'record', lambda name, sec: seen.append(name))
    assert client.get('/health').data == b'ok'
    client.get('/about')
    assert seen == ['shell']
