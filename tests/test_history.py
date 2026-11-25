"""the history of a user's changes (db.action, db.undo, /history), and what an agent's token may not change"""

from conftest import login
from test_agents import call

from aslite import db


def test_every_change_is_recorded_and_undone(corpus):
    a, b, c = sorted(corpus)[:3]
    db.tag_paper('alice', a, 'cv')
    db.tag_paper('alice', b, 'cv')
    db.tag_paper('alice', c, 'cv', neg=True)
    db.set_digest_size('alice', 'cv', 3)
    db.set_read('alice', a)
    db.set_note('alice', a, 'why I saved it')
    texts = [db.describe(e) for e in db.get_history('alice')]
    assert texts == [
        'Wrote a note on 1 paper',
        'Marked 1 paper read',
        'Set the digest size of cv to 3',
        "Marked 1 paper 'not for cv'",
        'Saved 1 paper to cv',
        'Saved 1 paper to cv',
    ]

    # deleting the tag is one entry, and undoing it brings back the saves, the downvote and the size
    db.delete_tag('alice', 'cv')
    e = db.get_history('alice')[0]
    assert db.describe(e) == "Deleted the tag cv (2 papers saved, 1 'not for')"
    assert db.undo('alice', e['id']) == {'restored': 4, 'skipped': 0, 'tags': ['cv']}
    assert db.get_tags('alice') == {'cv': {a, b}} and db.get_tags('alice', neg=True) == {'cv': {c}}
    assert db.get_digest_sizes('alice') == {'cv': 3}
    assert db.get_history('alice')[1]['undone']
    # the undo can be undone (a redo)
    redo = db.get_history('alice')[0]
    assert redo['kind'] == 'undo'
    db.undo('alice', redo['id'])
    assert db.get_tags('alice') == {}

    # a note edited again since isn't overwritten by undoing the first edit
    note = [e for e in db.get_history('alice') if e['kind'] == 'note'][0]
    db.set_note('alice', a, 'changed my mind')
    assert db.undo('alice', note['id'])['skipped'] == 1
    assert db.get_notes('alice') == {a: 'changed my mind'}
    assert db.undo('bob', note['id']) is None  # only one's own
    assert db.export_user('alice')['history']
    db.delete_user('alice')
    assert db.get_history('alice') == []


def test_history_page_undo_and_agent_actions(client, corpus):
    a, b = sorted(corpus)[:2]
    login(client)
    client.post(f'/add/{a}/cv')
    token = db.create_token('alice', 'my claude', 'write')
    res, err = call(client, 'save_papers', token, ids=[a, b], tag='nlp')
    assert err is None
    page = client.get('/ui/history').get_json()
    first = page['entries'][0]
    assert first['by'] == 'my claude' and first['text'] == 'Saved 2 papers to nlp' and first['can_undo']
    assert [p['id'] for p in first['papers']] == [a, b] and first['papers'][0]['title']
    assert page['entries'][1]['text'].startswith('Made the access token "my claude"')
    assert page['entries'][2] == {**page['entries'][2], 'by': '', 'text': 'Saved 1 paper to cv'}
    # one click undoes the agent's whole call
    assert client.post('/history/undo', data={'id': first['id']}).get_json()['restored'] == 2
    assert db.get_tags('alice') == {'cv': {a}}
    assert client.post('/history/undo', data={'id': 999999}).status_code == 404
    # an agent's token can neither read the history nor undo
    auth = {'Authorization': 'Bearer ' + token}
    assert client.get('/ui/history', headers=auth).status_code == 403
    assert client.post('/history/undo', data={'id': first['id']}, headers=auth).status_code == 403


def test_tokens_cant_change_account_settings(client, corpus):
    login(client)
    token = db.create_token('alice', 'agent', 'write')
    auth = {'Authorization': 'Bearer ' + token}
    assert client.post('/sharing', data={'public': '1'}, headers=auth).status_code == 403
    assert not db.get_user('alice')['public']
    assert client.post('/register_email', data={'email': 'evil@example.org'}, headers=auth).status_code == 403
    assert db.get_user('alice')['pending_email'] == ''
    assert client.post('/delete_account', data={'password': 'correct horse'}, headers=auth).status_code == 403
    # deleting a tag is fine (the history undoes it)
    pid = sorted(corpus)[0]
    client.post(f'/add/{pid}/cv')
    assert client.post('/del/cv', headers=auth).status_code == 200
    assert db.get_tags('alice') == {}
    assert db.get_history('alice')[0]['actor'] == 'agent'
    # the user on the site still can, and it is in the history
    assert client.post('/sharing', data={'public': '1'}).get_json() == {'ok': True}
    assert db.describe(db.get_history('alice')[0]) == 'Made the profile public'


def test_old_users_db_gets_the_history_table(tmp_path, monkeypatch):
    import sqlite3

    path = str(tmp_path / 'old_users.sqlite')
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE users (name TEXT PRIMARY KEY, pw_hash TEXT, email TEXT NOT NULL DEFAULT '')")
    con.execute(
        'CREATE TABLE tags (user TEXT, tag TEXT, pid TEXT, neg INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (user, tag, pid))'
    )
    con.commit()
    con.close()
    monkeypatch.setattr(db, 'USERS_DB_FILE', path)
    db.tag_paper('old', '2601.00001', 'cv')
    assert db.describe(db.get_history('old')[0]) == 'Saved 1 paper to cv'
