import importlib

import pytest


@pytest.fixture
def client(features):
    import serve
    importlib.reload(serve) # module-level state must not leak between temp dirs
    serve.app.config['TESTING'] = True
    with serve.app.test_client() as c:
        yield c


def login(client, user='alice'):
    client.post('/login', data={'username': user})


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
        assert client.get(f'/add/{p}/cv').data.startswith(b'ok')
    assert client.get(f'/addneg/{vision[4]}/cv').data.startswith(b'ok')

    for model in ['svm', 'logreg']:
        r = client.get(f'/?rank=tags&tags=cv&model={model}')
        assert r.status_code == 200
        assert b'2601.' in r.data

    # liking a downvoted paper moves it from neg_tags to tags
    client.get(f'/add/{vision[4]}/cv')
    from aslite.db import get_tags_db, get_neg_tags_db
    with get_tags_db() as tdb, get_neg_tags_db() as ndb:
        assert vision[4] in tdb['alice']['cv']
        assert vision[4] not in ndb['alice'].get('cv', set())
