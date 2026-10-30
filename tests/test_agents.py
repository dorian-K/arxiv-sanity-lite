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


def make_token(client, write=False):
    login(client)
    page = client.post('/agents/token', data={'name': 'test agent', **({'write': '1'} if write else {})}).data.decode()
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
    token = make_token(client, write=True)

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
    token = make_token(client, write=True)
    call(client, 'set_note', token, id=vision[0], text='x')
    login(client)
    client.post('/delete_account', data={'password': 'correct horse'})
    assert db.get_notes('alice') == {} and db.list_tokens('alice') == []
    assert client.get('/api/list_tags', headers={'Authorization': 'Bearer ' + token}).status_code == 401
