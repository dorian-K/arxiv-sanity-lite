"""the agent interface (agents.py): MCP, the JSON API, access tokens, notes"""

import re

from conftest import login

from aslite import db


def rpc(client, method, params=None, token=None, id=1):
    headers = {'Authorization': 'Bearer ' + token} if token else {}
    msg = {'jsonrpc': '2.0', 'id': id, 'method': method, **({'params': params} if params is not None else {})}
    r = client.post('/mcp', json=msg, headers=headers)
    assert r.status_code == 200, r.data
    return r.get_json()


def call(client, name, token=None, **args):
    """(parsed result or None, error text or None) of a tools/call"""
    import json

    res = rpc(client, 'tools/call', {'name': name, 'arguments': args}, token)['result']
    text = res['content'][0]['text']
    return (None, text) if res['isError'] else (json.loads(text), None)


def vision_and_language(corpus):
    pids = sorted(corpus)
    return [p for p in pids if 'vision' in corpus[p]['title']], [p for p in pids if 'language' in corpus[p]['title']]


def make_token(client, scope='read'):
    login(client)
    page = client.post('/agents/token', data={'name': 'test agent', 'scope': scope}).data.decode()
    client.post('/logout')
    return re.search(r'asr_[A-Za-z0-9_-]+', page).group()


def test_anyone_can_search_and_rank_like_these_papers(client, corpus):
    vision, language = vision_and_language(corpus)
    init = rpc(client, 'initialize', {'protocolVersion': '2025-06-18', 'capabilities': {}, 'clientInfo': {'name': 't'}})
    assert init['result']['protocolVersion'] == '2025-06-18' and 'rank_like' in init['result']['instructions']
    assert client.post('/mcp', json={'jsonrpc': '2.0', 'method': 'notifications/initialized'}).status_code == 202
    names = {t['name'] for t in rpc(client, 'tools/list')['result']['tools']}
    assert {'search', 'rank_like', 'get_papers', 'references'} <= names and 'list_tags' not in names

    # the main use: papers like these, without the input papers, the same topic first
    got, err = call(client, 'rank_like', like=vision[:3], limit=10, abstracts=False)
    assert err is None
    ids = [p['id'] for p in got['papers']]
    assert not set(ids) & set(vision[:3]) and set(ids[:5]) <= set(vision)
    assert all(0 < p['top'] <= 1 for p in got['papers']) and 'abstract' not in got['papers'][0]
    # links work as ids, papers that aren't here are reported, unlike pushes a topic away
    got, _ = call(client, 'rank_like', like=['https://arxiv.org/abs/' + vision[0], '2501.99999'], limit=5)
    assert got['not_here'] == ['2501.99999'] and got['papers'][0]['abstract']
    got, _ = call(client, 'rank_like', like=vision[:2] + language[:2], unlike=language[2:6], limit=5)
    assert set(p['id'] for p in got['papers']) <= set(vision)

    got, _ = call(client, 'search', query='segmentation', limit=3)
    assert len(got['papers']) == 3 and got['more']
    got, _ = call(client, 'get_papers', ids=[vision[0], 'nonsense'])
    assert (
        got['papers'][0]['id'] == vision[0]
        and got['not_arxiv_ids'] == ['nonsense']
        and 'saved_to' not in got['papers'][0]
    )

    # mistakes come back to the agent as tool errors, user tools need a token
    assert 'unknown argument' in call(client, 'search', query='x', nope=1)[1]
    assert 'between' in call(client, 'search', query='x', limit=1000)[1]
    assert 'access token' in call(client, 'list_tags')[1]
    assert rpc(client, 'nope')['error']['code'] == -32601

    # the same as plain JSON
    r = client.get(f'/api/rank_like?like={vision[0]},{vision[1]}&limit=3&abstracts=false').get_json()
    assert len(r['papers']) == 3
    assert client.get('/api/list_tags').status_code == 401
    assert client.get('/api/save_papers?ids=x&tag=y').status_code == 405
    assert 'rank_like' in client.get('/llms.txt').data.decode()
    listed = {t['name'] for t in client.get('/api').get_json()['tools']}
    assert names < listed and {'list_tags', 'set_note'} <= listed  # the api lists every tool, with its access


def test_a_token_reads_and_organizes_the_users_papers(client, corpus):
    vision, language = vision_and_language(corpus)
    login(client)
    for p in vision[:3]:
        client.post(f'/add/{p}/cv')
    client.post('/logout')
    token = make_token(client, 'write')

    got, err = call(client, 'list_tags', token)
    assert (
        err is None
        and got['user'] == 'alice'
        and got['tags'] == [{'name': 'cv', 'saved': 3, 'not_for': 0, 'to_read': 3}]
    )
    got, _ = call(client, 'recommend', token, tag='cv', days=None, limit=5, abstracts=False)
    assert got['papers'] and set(p['id'] for p in got['papers']) <= set(vision[3:])  # saved ones left out
    assert "no tag 'nlp'" in call(client, 'recommend', token, tag='nlp')[1]

    got, _ = call(client, 'save_papers', token, ids=[vision[3], 'https://arxiv.org/abs/' + vision[4]], tag='cv')
    assert set(got['status'].values()) == {'saved'}
    call(client, 'mark_read', token, ids=vision[:2])
    got, _ = call(client, 'tag_papers', token, tag='cv', list='to_read')
    assert {p['id'] for p in got['papers']} == set(vision[2:5])
    got, _ = call(client, 'set_note', token, id=vision[2], text='  the segmentation baseline  ')
    assert got['note'] == 'the segmentation baseline'
    got, _ = call(client, 'get_papers', token, ids=[vision[2]])
    assert got['papers'][0]['note'] == 'the segmentation baseline' and got['papers'][0]['saved_to'] == ['cv']
    call(client, 'mark_not_for', token, ids=[language[0]], tag='cv')
    assert db.get_tags('alice', neg=True) == {'cv': {language[0]}}
    got, _ = call(client, 'rank_like', token, like=vision[:2], exclude_rated=True, limit=20)
    assert not {p['id'] for p in got['papers']} & set(vision[:5])

    # the note is on the card, for the user only; read marks too
    login(client)
    card = {p['id']: p for p in client.get('/?rank=saved&tags=cv&format=json').get_json()['papers']}
    assert card[vision[2]]['note'] == 'the segmentation baseline' and card[vision[0]]['read']
    client.post(f'/note/{vision[2]}', data={'text': ''})
    assert db.get_notes('alice') == {}

    # the cookie alone gets nothing from the agent endpoints: no page can use them as its visitor
    assert 'access token' in call(client, 'list_tags')[1]
    export = client.get('/export').get_json()
    assert export['agent_tokens'][0]['name'] == 'test agent' and 'asr_' not in str(export)

    # revoked: 401 everywhere
    tid = db.list_tokens('alice')[0]['id']
    client.post('/agents/revoke', data={'id': tid})
    r = client.post(
        '/mcp', json={'jsonrpc': '2.0', 'id': 1, 'method': 'ping'}, headers={'Authorization': 'Bearer ' + token}
    )
    assert r.status_code == 401


def test_read_only_tokens_cant_change_anything(client, corpus):
    vision, _ = vision_and_language(corpus)
    token = make_token(client)
    names = {t['name'] for t in rpc(client, 'tools/list', token=token)['result']['tools']}
    assert 'list_tags' in names and 'save_papers' not in names
    assert 'read-only' in call(client, 'save_papers', token, ids=[vision[0]], tag='cv')[1]
    auth = {'Authorization': 'Bearer ' + token}
    assert client.post(f'/add/{vision[0]}/cv', headers=auth).status_code == 403  # nor the site's own endpoints
    assert client.post('/agents/token', data={'name': 'x'}, headers=auth).status_code in (302, 403)  # nor make tokens
    assert db.get_tags('alice') == {} and len(db.list_tokens('alice')) == 1
    assert client.get('/api/list_tags', headers={'Authorization': 'Bearer asr_wrong'}).status_code == 401


def test_references_and_citations(client, corpus):
    vision, _ = vision_and_language(corpus)
    got, _ = call(client, 'references', id=vision[0])
    assert 'not processed' in got['status']
    with db.get_refs_db() as rdb:
        rdb.put_refs(
            vision[0], 'ok', [(0, vision[1], 'A title', None, 'raw text'), (1, None, '', '10.1/x', 'Some book')], 'html'
        )
    got, _ = call(client, 'references', id=vision[0])
    assert got['references'][0] == {'n': 1, 'title': 'A title', 'arxiv_id': vision[1], 'on_site': True}
    assert got['references'][1]['text'] == 'Some book' and got['references'][1]['doi'] == '10.1/x'
    got, _ = call(client, 'cited_by', id=vision[1])
    assert [p['id'] for p in got['papers']] == [vision[0]] and got['papers_with_known_references'] == 1


def test_deleting_the_account_deletes_tokens_and_notes(client, corpus):
    vision, _ = vision_and_language(corpus)
    token = make_token(client, 'write')
    call(client, 'set_note', token, id=vision[0], text='x')
    login(client)
    client.post('/delete_account', data={'password': 'correct horse'})
    assert db.get_notes('alice') == {} and db.list_tokens('alice') == []
    assert client.get('/api/list_tags', headers={'Authorization': 'Bearer ' + token}).status_code == 401


CALLBACK = 'https://claude.ai/api/mcp/auth_callback'


def pkce():
    import base64
    import hashlib
    import secrets

    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode()
    return verifier, challenge


def authorize_url(cid, challenge, **more):
    from urllib.parse import urlencode

    q = {
        'response_type': 'code',
        'client_id': cid,
        'redirect_uri': CALLBACK,
        'code_challenge': challenge,
        'code_challenge_method': 'S256',
        'state': 'xyz',
        'scope': 'read write',
        **more,
    }
    return '/oauth/authorize?' + urlencode(q)


def grant(client, cid, access='write', decision='allow'):
    """(verifier, the redirect back to the app) after the user answers the consent page"""
    from urllib.parse import parse_qs, urlparse

    verifier, challenge = pkce()
    url = authorize_url(cid, challenge)
    form = dict(parse_qs(urlparse(url).query, keep_blank_values=True))
    form = {k: v[0] for k, v in form.items()} | {'decision': decision, 'access': access}
    r = client.post('/oauth/authorize', data=form)
    assert r.status_code == 302
    return verifier, urlparse(r.headers['Location'])


def test_claude_ai_connects_with_oauth(client, corpus):
    from urllib.parse import parse_qs

    vision, _ = vision_and_language(corpus)
    # 1. the connector finds out where to log in
    r = client.post('/mcp/me', json={'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {}})
    assert r.status_code == 401 and 'resource_metadata="' in r.headers['WWW-Authenticate']
    meta = client.get('/.well-known/oauth-protected-resource/mcp/me').get_json()
    assert meta['resource'].endswith('/mcp/me')
    server = client.get('/.well-known/oauth-authorization-server').get_json()
    assert server['code_challenge_methods_supported'] == ['S256'] and server['registration_endpoint'].endswith(
        '/oauth/register'
    )
    # 2. registers
    r = client.post('/oauth/register', json={'client_name': 'Claude', 'redirect_uris': [CALLBACK]})
    assert r.status_code == 201
    cid = r.get_json()['client_id']
    for bad in (['http://evil.example/cb'], ['https://claude.ai/cb#x'], [], ['x'] * 6):
        assert client.post('/oauth/register', json={'redirect_uris': bad}).status_code == 400
    # 3. the user logs in on the consent page and comes back to it
    _, challenge = pkce()
    page = client.get(authorize_url(cid, challenge)).data.decode()
    assert 'Log in' in page and 'claude.ai' in page
    r = client.post(
        '/login', data={'username': 'alice', 'password': 'correct horse', 'next': authorize_url(cid, challenge)}
    )
    assert r.headers['Location'].startswith('/oauth/authorize?')
    r = client.get(r.headers['Location'])
    assert 'Allow' in r.data.decode() and "form-action 'self' https://claude.ai" in r.headers['Content-Security-Policy']
    # 4. allows, gets a code back at its callback, swaps it for a token
    verifier, back = grant(client, cid)
    q = parse_qs(back.query)
    assert back.netloc == 'claude.ai' and q['state'] == ['xyz'] and 'iss' in q
    code = q['code'][0]
    token_req = {
        'grant_type': 'authorization_code',
        'code': code,
        'client_id': cid,
        'redirect_uri': CALLBACK,
        'code_verifier': verifier,
    }
    client.post('/logout')
    r = client.post('/oauth/token', data=token_req)
    assert r.status_code == 200 and r.headers['Cache-Control'] == 'no-store'
    token = r.get_json()['access_token']
    assert r.get_json()['scope'] == 'write'
    assert client.post('/oauth/token', data=token_req).get_json()['error'] == 'invalid_grant'  # single use
    # 5. and works as the user
    auth = {'Authorization': 'Bearer ' + token}
    r = client.post('/mcp/me', json={'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'}, headers=auth)
    assert 'save_papers' in {t['name'] for t in r.get_json()['result']['tools']}
    assert call(client, 'save_papers', token, ids=[vision[0]], tag='cv')[0]['status'] == {vision[0]: 'saved'}
    # connecting again replaces the app's token; /agents lists it
    login(client)
    verifier, back = grant(client, cid, access='read')
    r = client.post(
        '/oauth/token', data=token_req | {'code': parse_qs(back.query)['code'][0], 'code_verifier': verifier}
    )
    assert r.get_json()['scope'] == 'read'
    assert [t['name'] for t in db.list_tokens('alice')] == ['Claude (connected)']
    assert client.post('/mcp/me', json={'jsonrpc': '2.0', 'id': 1, 'method': 'ping'}, headers=auth).status_code == 401


def test_oauth_refuses_what_it_must(client, corpus, monkeypatch):
    from urllib.parse import parse_qs

    cid = client.post('/oauth/register', json={'client_name': 'x', 'redirect_uris': [CALLBACK]}).get_json()['client_id']
    login(client)
    _, challenge = pkce()
    # never redirect to an address the app didn't register, nor for an unknown app
    for url in (
        authorize_url(cid, challenge, redirect_uri='https://evil.example/cb'),
        authorize_url('nope', challenge),
        authorize_url(cid, challenge, code_challenge_method='plain'),
    ):
        r = client.get(url)
        assert r.status_code == 400 and "Can't connect" in r.data.decode()
    # deny: back to the app with an error, no code
    _, back = grant(client, cid, decision='deny')
    assert parse_qs(back.query)['error'] == ['access_denied'] and 'code' not in parse_qs(back.query)
    # a wrong verifier (a stolen code) uses the code up
    verifier, back = grant(client, cid)
    req = {
        'grant_type': 'authorization_code',
        'code': parse_qs(back.query)['code'][0],
        'client_id': cid,
        'redirect_uri': CALLBACK,
        'code_verifier': 'wrong' * 10,
    }
    assert client.post('/oauth/token', data=req).get_json()['error'] == 'invalid_grant'
    assert client.post('/oauth/token', data=req | {'code_verifier': verifier}).get_json()['error'] == 'invalid_grant'
    # another client, or another redirect, can't use a code
    verifier, back = grant(client, cid)
    req = req | {'code': parse_qs(back.query)['code'][0], 'code_verifier': verifier, 'redirect_uri': CALLBACK + 'x'}
    assert client.post('/oauth/token', data=req).get_json()['error'] == 'invalid_grant'
    # expired codes are gone
    monkeypatch.setattr(db, 'OAUTH_CODE_SECONDS', -1)
    verifier, back = grant(client, cid)
    req = req | {'code': parse_qs(back.query)['code'][0], 'code_verifier': verifier, 'redirect_uri': CALLBACK}
    assert client.post('/oauth/token', data=req).get_json()['error'] == 'invalid_grant'
    assert db.list_tokens('alice') == []
    # a login's next never leads off the site
    client.post('/logout')
    r = client.post('/login', data={'username': 'alice', 'password': 'correct horse', 'next': '//evil.example/x'})
    assert r.headers['Location'] in ('/', 'http://localhost/')


def test_old_users_db_gets_the_token_client_column(tmp_path, monkeypatch):
    import sqlite3

    path = str(tmp_path / 'users.sqlite')
    con = sqlite3.connect(path)
    con.executescript(
        db.USERS_SCHEMA.replace(
            ",\n    client TEXT NOT NULL DEFAULT ''  -- the OAuth client it was issued to ('': made by hand on /agents)",
            '',
        )
    )
    assert 'client' not in {r[1] for r in con.execute('PRAGMA table_info(tokens)')}
    con.execute("INSERT INTO tokens (hash, user, name, write, created) VALUES ('h', 'alice', 'old', 0, 0)")
    con.commit()
    con.close()
    monkeypatch.setattr(db, 'USERS_DB_FILE', path)
    assert db.list_tokens('alice')[0]['name'] == 'old'
    db.create_token('alice', 'new', client='c1')
    assert {t['name'] for t in db.list_tokens('alice')} == {'old', 'new'}


def test_agent_tags_tokens_keep_the_users_own_tags(client, corpus):
    vision, _ = vision_and_language(corpus)
    login(client)
    client.post(f'/add/{vision[0]}/cv')
    client.post('/logout')
    token = make_token(client, 'agent')
    names = {t['name'] for t in rpc(client, 'tools/list', token=token)['result']['tools']}
    assert {'save_papers', 'unsave_papers', 'mark_not_for'} <= names and not {'mark_read', 'set_note'} & names
    assert 'agent:' in call(client, 'list_tags', token)[0]['access']
    # every write goes to an agent: tag, named so automatically
    got, _ = call(client, 'save_papers', token, ids=[vision[1]], tag='video')
    assert got['tag'] == 'agent:video'
    call(client, 'save_papers', token, ids=[vision[2]], tag='agent:video')
    got, _ = call(client, 'unsave_papers', token, ids=[vision[0]], tag='cv')
    assert got['tag'] == 'agent:cv' and got['removed'] == []
    call(client, 'mark_not_for', token, ids=[vision[0]], tag='cv')
    assert db.get_tags('alice') == {'cv': {vision[0]}, 'agent:video': {vision[1], vision[2]}}
    assert db.get_tags('alice', neg=True) == {'agent:cv': {vision[0]}}
    # nothing else: no read marks, notes, or the site's own endpoints
    assert 'full changes' in call(client, 'mark_read', token, ids=[vision[0]])[1]
    assert 'full changes' in call(client, 'set_note', token, id=vision[0], text='x')[1]
    auth = {'Authorization': 'Bearer ' + token}
    assert client.post(f'/sub/{vision[0]}/cv', headers=auth).status_code == 403
    assert client.post(f'/read/{vision[0]}', headers=auth).status_code == 403
    assert db.get_tags('alice')['cv'] == {vision[0]} and db.get_reads('alice') == set()
    # and the OAuth consent page offers the same choice
    cid = client.post('/oauth/register', json={'redirect_uris': [CALLBACK]}).get_json()['client_id']
    login(client)
    from urllib.parse import parse_qs

    verifier, back = grant(client, cid, access='agent')
    req = {
        'grant_type': 'authorization_code',
        'code': parse_qs(back.query)['code'][0],
        'client_id': cid,
        'redirect_uri': CALLBACK,
        'code_verifier': verifier,
    }
    assert client.post('/oauth/token', data=req).get_json()['scope'] == 'agent'


def test_an_app_on_another_machine_gets_the_callback_to_open_there(client, corpus):
    """Claude Code over ssh: its callback listens on that machine's localhost, the browser is elsewhere"""
    import html
    from urllib.parse import parse_qs, urlencode, urlparse

    local = 'http://localhost:43123/callback'
    cid = client.post('/oauth/register', json={'client_name': 'Claude Code', 'redirect_uris': [local]}).get_json()[
        'client_id'
    ]
    login(client)
    verifier, challenge = pkce()
    q = {
        'response_type': 'code',
        'client_id': cid,
        'redirect_uri': local,
        'code_challenge': challenge,
        'code_challenge_method': 'S256',
        'state': 's1',
    }
    assert 'your own' in client.get('/oauth/authorize?' + urlencode(q)).data.decode()
    r = client.post('/oauth/authorize', data=q | {'decision': 'allow', 'access': 'read'})
    page = r.data.decode()
    assert r.status_code == 200 and 'Back to the app' in page and "curl -s '" in page
    callback = html.unescape(re.search(r"curl -s '([^']+)'", page).group(1))
    assert callback.startswith(local + '?')
    back = parse_qs(urlparse(callback).query)
    assert back['state'] == ['s1']
    req = {
        'grant_type': 'authorization_code',
        'code': back['code'][0],
        'client_id': cid,
        'redirect_uri': local,
        'code_verifier': verifier,
    }
    assert client.post('/oauth/token', data=req).get_json()['scope'] == 'read'
