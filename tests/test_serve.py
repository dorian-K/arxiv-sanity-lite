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

    for model in ['svm', 'logreg', 'lda']:
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
    assert '2502.00002' not in db.get_tags('alice')['reading']  # not fetched: not tagged either


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
    assert all(f'<option value="{d}"' in page for d in ('7', '30', '365', ''))  # the time window
    # the page fetches its results as /?format=json, page 2 is /?page_number=2: both still "For you"
    # (they once fell back to the newest papers, which looked like random recommendations)
    import serve

    pct = [p.get('pct') for p in client.get('/?format=json').get_json()['papers']]
    assert pct and all(pct)
    assert '"rank": "tags"' in client.get('/?page_number=2').data.decode()
    assert serve.FOR_YOU_DAYS == 30
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


def test_for_you_can_let_each_tag_pick_in_turns(client, corpus):
    login(client)
    vision = [p for p in corpus if corpus[p]['tags'][0]['term'] == 'cs.CV']
    language = [p for p in corpus if corpus[p]['tags'][0]['term'] != 'cs.CV']
    for p in vision[:4]:
        client.post(f'/add/{p}/cv')
    for p in language[:4]:
        client.post(f'/add/{p}/nlp')
    # the default: one model on all tags together, no per-tag labels
    papers = client.get('/?rank=tags&tags=all&skip_have=yes&format=json').get_json()['papers']
    assert papers and not any(p.get('for_tag') for p in papers)
    papers = client.get('/?rank=tags&tags=all&skip_have=yes&mix=turns&format=json').get_json()['papers']
    # one model per tag, taking turns: every paper says which tag picked it, and both get a say
    assert [p['for_tag'] for p in papers[:4]] == ['cv', 'nlp', 'cv', 'nlp']
    assert all(corpus[p['id']]['tags'][0]['term'] == 'cs.CV' for p in papers[:6] if p['for_tag'] == 'cv')
    assert len({p['id'] for p in papers}) == len(papers)
    # the C used by default is shown
    from aslite.rank import DEFAULT_C

    page = client.get('/?rank=tags&tags=cv&model=logreg&feats=tfidf').data.decode()
    assert f'placeholder="default: {DEFAULT_C[("logreg", "tfidf")]:g}"' in page


def test_downvotes_are_visible_and_only_hide_for_their_tag(client, corpus):
    login(client)
    vision = [p for p in corpus if corpus[p]['tags'][0]['term'] == 'cs.CV']
    for p in vision[:4]:
        client.post(f'/add/{p}/cv')
        client.post(f'/add/{p}/images')
    bad = vision[5]
    client.post(f'/addneg/{bad}/cv')
    # the tag page lists its downvotes in their own tab
    page = client.get('/?rank=down&tags=cv').data.decode()
    assert 'Not for this tag (1)' in page and f'"id": "{bad}"' in page
    assert f'"id": "{bad}"' not in client.get('/?rank=down&tags=images').data.decode()
    # hidden from the tag it was marked for, but still recommended for the other one
    ids = lambda tag: [
        p['id'] for p in client.get(f'/?rank=tags&tags={tag}&skip_have=yes&format=json').get_json()['papers']
    ]
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
    page = client.get('/?u=bob&rank=saved&tags=cv').data.decode()
    assert "profile isn't public" in page and vision[0] not in page
    db.set_public('bob', True)
    # shared: anyone with the link sees bob's tag, read-only, no login needed
    page = client.get('/?u=bob&rank=saved&tags=cv').data.decode()
    assert 'viewing <b>@bob</b>' in page and f'"id": "{vision[0]}"' in page and 'var viewing = true' in page
    assert client.get('/?u=bob&rank=tags&tags=cv&format=json').get_json()['papers']
    assert client.get('/tag_check?tag=cv&u=bob').status_code == 200
    assert b'@bob&#39;s digest' in client.get('/digest?u=bob').data
    # a logged-in visitor can't vote on it, but can copy it into their own tags
    login(client)  # alice
    assert '"tags": "all"' in client.get('/?u=bob').data.decode()  # bob's "For you"
    r = client.post('/copy_tag', data={'owner': 'bob', 'tag': 'cv', 'as': 'bobs-cv'})
    assert r.status_code == 302 and 'bobs-cv' in r.headers['Location']
    assert db.get_tags('alice')['bobs-cv'] == set(vision[:4])
    assert db.get_tags('alice', neg=True)['bobs-cv'] == {vision[6]}
    assert db.get_tags('bob')['cv'] == set(vision[:4])  # bob's own tag is untouched
    # writes never go to the owner, whatever the url says
    client.post(f'/add/{vision[5]}/cv?u=bob')
    assert vision[5] not in db.get_tags('bob')['cv']
    # sharing off again: links and copying stop at once
    db.set_public('bob', False)
    assert vision[0] not in client.get('/?u=bob&rank=saved&tags=cv').data.decode()
    assert client.post('/copy_tag', data={'owner': 'bob', 'tag': 'cv'}).status_code == 404


def test_search_ignores_the_pages_filters(client, corpus):
    login(client)
    pid = next(iter(corpus))
    client.post(f'/add/{pid}/cv')
    title = corpus[pid]['title']
    # the search box sends the page's filters along (here: those of "For you"); they don't apply
    r = client.get('/?' + 'q=' + title.split()[0] + '&rank=tags&time_filter=1&skip_have=yes')
    assert pid in r.data.decode()
    # unless search is picked explicitly
    r = client.get('/?q=' + title.split()[0] + '&rank=search&skip_have=yes')
    assert f'"id": "{pid}"' not in r.data.decode()


def test_unknown_paper_and_odd_tag_names(client, corpus):
    login(client)
    assert 'no paper with this id' in client.get('/?rank=pid&pid=2999.99999&format=json').get_json()['notice']
    for bad in ('.', '..'):
        assert client.post('/import', data={'text': next(iter(corpus)), 'tag': bad}).status_code == 400


def test_errors_are_pages_of_the_site(client, corpus):
    r = client.get('/inspect?pid=2999.99999')
    assert r.status_code == 404 and b'Unknown paper' in r.data and b'<header' in r.data
    login(client)
    pid = next(iter(corpus))
    client.post(f'/add/{pid}/cv')
    page = client.get('/?rank=saved&tags=cv').data.decode()
    assert 'id="deltag"' in page  # deleting a tag is on its page, not on the chips


def test_newest_and_random_keep_to_the_time_window(client, corpus):
    # paper i came out i hours ago (conftest)
    for rank in ('time', 'random'):
        ids = [p['id'] for p in client.get(f'/?rank={rank}&time_filter=1&format=json').get_json()['papers']]
        assert ids and all(int(p.split('.')[1]) < 24 for p in ids), rank
    ids = [p['id'] for p in client.get('/?rank=time&format=json').get_json()['papers']]
    assert ids == sorted(ids)[:25]  # newest first


def test_static_files_are_versioned_and_cached(client):
    page = client.get('/about').get_data(as_text=True)
    assert '/static/common.js?v=' in page
    assert 'immutable' in client.get('/static/common.js?v=1').headers['Cache-Control']
    assert 'immutable' not in client.get('/static/common.js').headers.get('Cache-Control', '')


def test_published_filter_and_venues(client, corpus):
    from aslite import db

    pids = sorted(corpus)
    with db.get_papers_db() as pdb:
        p = pdb[pids[3]]
        p['arxiv_comment'], p['_version'] = '8 pages. Accepted at NeurIPS 2026', 2
        pdb.upsert([p])
    for url in ('/?rank=time&published=yes', '/?rank=search&q=paper&published=yes'):
        ids = [x['id'] for x in client.get(url + '&format=json').get_json()['papers']]
        assert ids == [pids[3]], url
    page = client.get('/?rank=time&format=json').get_json()['papers']
    assert next(x for x in page if x['id'] == pids[3])['venue'] == 'Accepted at NeurIPS 2026'
    assert 'published_only' in client.get('/?rank=time').get_data(as_text=True)  # the checkbox
    # recommendations too
    login(client)
    for x in pids[:6]:
        client.post(f'/add/{x}/t')
    res = client.get('/?rank=tags&tags=t&published=yes&format=json').get_json()['papers']
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

    assert 'name="public"' not in client.get('/profile').get_data(as_text=True)  # not on the login form
    client.post('/login', data={'username': 'pub', 'password': 'correct horse'})
    assert db.get_user('pub')['public'] == 0  # private until answered
    page = client.get('/').get_data(as_text=True)
    assert 'id="askpublic"' in page and 'name="public" value="1" checked' in page
    r = client.post('/sharing', data={'public': '1', 'next': '/?rank=time'})
    assert r.headers['Location'].endswith('/?rank=time') and db.get_user('pub')['public'] == 1
    assert 'id="askpublic"' not in client.get('/').get_data(as_text=True)  # asked once
    client.post('/logout')
    client.post('/login', data={'username': 'priv', 'password': 'correct horse'})
    r = client.post('/sharing', data={'next': '//evil.example/'})  # unticked; and no open redirect
    assert db.get_user('priv')['public'] == 0 and 'evil' not in r.headers['Location']
    client.post('/logout')
    client.post('/login', data={'username': 'priv', 'password': 'correct horse'})  # a login doesn't ask again
    assert 'id="askpublic"' not in client.get('/').get_data(as_text=True)


def test_search_fixes_typos_unless_asked_not_to(client, corpus, monkeypatch):
    import serve
    from aslite.spell import Speller

    assert serve.get_speller() is not None  # compute.py wrote the words
    pid, p = next(iter(corpus.items()))
    word = max(p['title'].lower().split(), key=len)
    monkeypatch.setattr(serve, 'get_speller', lambda: Speller({word: 100}))
    r = client.get('/?rank=search&q=' + word[0] + word[2:])  # one letter missing
    assert ('"corrected": "%s"' % word).encode() in r.data and pid.encode() in r.data
    r = client.get('/?rank=search&exact=yes&q=' + word[0] + word[2:])
    assert b'"corrected": ""' in r.data


def test_stats_show_page_speed_and_background_work(client, corpus):
    from aslite import db

    login(client)
    vision = [p for p in corpus if corpus[p]['tags'][0]['term'] == 'cs.CV']
    for p in vision[:3]:
        client.post(f'/add/{p}/cv')
    client.get('/?rank=search&q=vision')
    client.get('/?rank=tags&tags=cv&format=json')
    db.put_job('embed', 12.0, papers=8, left=400)
    page = ' '.join(client.get('/stats').data.decode().split())
    for label in ('Search', 'Vote: save', 'Recommendations, the results after the placeholder', 'Training a model'):
        assert label in page
    assert 'Rebuilding the index' in page and 'MB' in page  # conftest's compute.py run
    assert '400 papers still without an embedding, about 50.0 days' in page
