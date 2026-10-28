import importlib
import os

import pytest


@pytest.fixture
def client(features):
    import serve

    importlib.reload(serve)  # module-level state must not leak between temp dirs
    serve.app.config['TESTING'] = True
    with serve.app.test_client() as c:
        yield c


def login(client, user='alice', password='correct horse'):
    return client.post('/login', data={'username': user, 'password': password})


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
        r = client.get(f'/?rank=tags&tags=cv&model={model}')
        assert r.status_code == 200
        assert b'2601.' in r.data

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
    r = client.post('/import', data={'text': '2502.00001\n2502.00002\n1706.03762'})
    assert 'over the daily fetch limit' in r.data.decode()
    assert calls[-1] == ['2502.00001']


def test_rate_limits(client, corpus, monkeypatch):
    import serve

    monkeypatch.setattr(
        serve, 'RATE_LIMITS', {'rank': (2, 60), 'search': (3, 60), 'login': (2, 300), 'import': (1, 3600)}
    )
    pid = next(iter(corpus))
    assert [client.get(f'/?rank=pid&pid={pid}').status_code for _ in range(3)] == [200, 200, 429]
    assert [client.get('/?q=vision').status_code for _ in range(4)] == [200, 200, 200, 429]
    assert client.get('/').status_code == 200  # cheap pages are never limited
    login(client)  # signs up; the login limit counts this attempt too
    assert login(client, password='wrong password').status_code == 403
    assert login(client, password='wrong password').status_code == 429
    # logged in, a client is the user rather than the ip: a fresh budget, then the limit again
    codes = [client.get(f'/?rank=pid&pid={pid}').status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    r = client.get(f'/?rank=pid&pid={pid}')
    assert r.status_code == 429 and int(r.headers['Retry-After']) >= 1
