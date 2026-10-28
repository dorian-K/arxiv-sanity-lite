import os

from conftest import login


def test_pages_render(client, corpus):
    for url in ['/', '/?rank=random', '/?q=convolution', '/about', '/stats', '/profile']:
        r = client.get(url)
        assert r.status_code == 200, url
    pid = next(iter(corpus))
    assert client.get(f'/inspect?pid={pid}').status_code == 200


def test_tag_rank_and_downvote(client, corpus):
    login(client)
    vision = [p for p in corpus if corpus[p]['tags'][0]['term'] == 'cs.CV']
    for p in vision[:4]:
        assert client.post(f'/add/{p}/cv').data.startswith(b'ok')
    assert client.post(f'/addneg/{vision[4]}/cv').data.startswith(b'ok')

    for model in ['svm', 'logreg']:
        # the page comes back at once with a placeholder, and fetches the ranking as json
        r = client.get(f'/?rank=tags&tags=cv&model={model}')
        assert r.status_code == 200 and b'"pending": true' in r.data
        res = client.get(f'/?rank=tags&tags=cv&model={model}&format=json').get_json()
        assert res['papers'] and res['papers'][0]['id'].startswith('2601.')
        # from then on it is cached: the page itself has the papers
        r = client.get(f'/?rank=tags&tags=cv&model={model}')
        assert b'"pending": false' in r.data and b'2601.' in r.data

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
        assert client.get('/?' + q).status_code == 200


def test_import_suggests_existing_tags(client, corpus):
    login(client)
    a, b = list(corpus)[:2]
    client.post(f'/add/{a}/cv')
    client.post(f'/addneg/{b}/nlp')
    page = client.get('/import').data
    assert b'<option value="cv">' in page and b'<option value="nlp">' in page


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
    assert b'2601.' in client.get('/?q=vision').data


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

    monkeypatch.setattr(serve, 'fetch_papers_by_id', fake_fetch)
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
    assert b'log in' in client.post('/import', data={'text': text}).data  # logged out: nothing happens
    assert calls == []

    login(client)
    r = client.post('/import', data={'text': text, 'tag': 'reading'})
    assert r.status_code == 200
    page = r.data.decode()
    for s in ['already here', 'added from arXiv', 'not found on arXiv', 'old-style', 'no arXiv id found']:
        assert s in page, s
    assert calls == [['1706.03762', '2501.99999']]  # one api call for everything missing
    assert db.get_tags('alice')['reading'] == {have, '1706.03762'}
    assert os.path.exists(db.COMPUTE_FLAG_FILE)
    assert db.get_papers_db()['1706.03762']['title'] == 'an old paper about attention'
    assert b'1706.03762' in client.get('/?q=arxiv.org/abs/1706.03762').data

    # fetches count against the daily limit; papers we already have don't
    monkeypatch.setattr(serve, 'FETCH_LIMIT_USER', 3)
    r = client.post('/import', data={'text': '2502.00001\n2502.00002\n1706.03762', 'tag': 'reading'})
    assert 'over the daily fetch limit' in r.data.decode()
    assert calls[-1] == ['2502.00001']


def test_rate_limits(client, corpus, monkeypatch):
    import serve

    monkeypatch.setattr(
        serve, 'RATE_LIMITS', {'rank': (2, 60), 'search': (3, 60), 'login': (2, 300), 'import': (1, 3600)}
    )
    pids = list(corpus)
    # distinct queries: a repeated one comes from the cache and costs nothing
    assert [client.get(f'/?rank=pid&pid={p}&format=json').status_code for p in pids[:3]] == [200, 200, 429]
    assert client.get(f'/?rank=pid&pid={pids[0]}&format=json').status_code == 200  # cached
    assert [client.get(f'/?q=vision{i}').status_code for i in range(4)] == [200, 200, 200, 429]
    assert client.get('/').status_code == 200  # cheap pages are never limited
    login(client)  # signs up; the login limit counts this attempt too
    assert login(client, password='wrong password').status_code == 403
    assert login(client, password='wrong password').status_code == 429
    # logged in, a client is the user rather than the ip: a fresh budget, then the limit again
    codes = [client.get(f'/?rank=pid&pid={p}&format=json').status_code for p in pids[3:6]]
    assert codes == [200, 200, 429]
    r = client.get(f'/?rank=pid&pid={pids[6]}&format=json')
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
    assert b'read-only' in client.get('/?rank=tags&tags=cv').data
    assert client.get('/?rank=tags&tags=cv').status_code == 200
    assert b'alice@example.org' not in client.get('/profile').data  # the email stays private
    for url in [f'/add/{pid}/x', f'/sub/{pid}/cv', f'/addneg/{pid}/cv', '/del/cv', '/import', '/digest', '/sharing']:
        assert client.post(url, data={'text': '2501.00001', 'public': '0', 'email': ''}).status_code == 403, url
    assert db.get_tags('alice') == {'cv': {pid}} and db.get_user('alice')['public'] == 1

    # the owner turns sharing off: the read-only session ends on its next request
    db.set_public('alice', False)
    assert b'read-only' not in client.get('/').data
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
    res = client.get('/?rank=tags&tags=cv&format=json').get_json()
    assert all(0 < p['pct'] <= 1 for p in res['papers'])
    assert 'tagcheckwrap' in client.get('/?rank=tags&tags=cv').data.decode()


def test_tag_pages_and_landing(client, corpus):
    from aslite import db

    login(client)
    assert b'"rank": "time"' in client.get('/').data  # nothing saved yet: the newest papers
    vision = [p for p in corpus if corpus[p]['tags'][0]['term'] == 'cs.CV']
    for p in vision[:3]:
        client.post(f'/add/{p}/cv')
    # with tags, the main page is the best recent papers for them
    page = client.get('/').data.decode()
    assert '"rank": "tags"' in page and '"tags": "all"' in page and '"time_filter": "30"' in page
    # every tag has a page with its saved papers, newest first
    page = client.get('/?rank=saved&tags=cv').data.decode()
    assert 'Saved (3)' in page and all(p in page for p in vision[:3]) and vision[3] not in page
    times = db.get_papers_db().times()
    shown = [p for p in page.split('"id": "')[1:]]
    ids = [x.split('"')[0] for x in shown]
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
    res = client.get('/?rank=tags&tags=fresh&format=json').get_json()
    assert res['papers'] == [] and 'added recently' in res['notice']
    res = client.get('/?rank=pid&pid=2602.99999&format=json').get_json()
    assert 'added recently' in res['notice']


def test_import_needs_a_tag(client, corpus):
    login(client)
    r = client.post('/import', data={'text': next(iter(corpus))})
    assert r.status_code == 400 and b'Pick a tag' in r.data
