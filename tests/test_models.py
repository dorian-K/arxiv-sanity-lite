import threading
import time

import numpy as np

from conftest import login


def vision_papers(corpus):
    return [p for p in corpus if corpus[p]['tags'][0]['term'] == 'cs.CV']


def counting_fit(monkeypatch, delay=0.0):
    """counts the models serve.py trains (and can make each fit slow)"""
    import serve

    calls = []
    real = serve.fit

    def fit(*args, **kwargs):
        calls.append(1)
        time.sleep(delay)
        return real(*args, **kwargs)

    monkeypatch.setattr(serve, 'fit', fit)
    return calls


def rank_json(client, url):
    return client.get(url + '&format=json').get_json()['papers']


def test_a_model_is_trained_once_per_training_set(client, corpus, monkeypatch):
    import serve

    login(client)
    vision = vision_papers(corpus)
    for p in vision[:4]:
        client.post(f'/add/{p}/cv')
    calls = counting_fit(monkeypatch)
    for days in ('1', '3', ''):  # another time window: the same model
        assert rank_json(client, f'/?rank=tags&tags=cv&skip_have=yes&time_filter={days}')
    assert len(calls) == 1
    # another worker (or a restart) finds it in models.sqlite
    serve._models.clear()
    serve._rank_cache.clear()
    assert rank_json(client, '/?rank=tags&tags=cv&skip_have=yes')
    assert len(calls) == 1
    # neutral votes don't train, so they keep the model
    client.post(f'/seen/{vision[5]}/cv')
    rank_json(client, '/?rank=tags&tags=cv&skip_have=yes')
    assert len(calls) == 1
    # a like is a new training set
    client.post(f'/add/{vision[6]}/cv')
    rank_json(client, '/?rank=tags&tags=cv&skip_have=yes')
    assert len(calls) == 2


def test_a_model_is_only_used_for_the_state_it_was_trained_on(client, corpus, features):
    """a fit that finishes after a vote (stored late) must not answer for the new state"""
    import serve
    from aslite import db

    login(client)
    vision = vision_papers(corpus)
    for p in vision[:4]:
        client.post(f'/add/{p}/cv')
    f = db.load_features()
    old = serve.training_set(db.get_tags('alice'), {}, 'cv')
    client.post(f'/add/{vision[4]}/cv')  # the vote lands while the old fit still runs
    m_old, _ = serve.fit(f, *old)
    db.put_model('alice', 'cv', 'concat|logreg|None', serve.fingerprint(*old), f['_version'], m_old)
    db.put_model('alice', 'cv', 'tfidf|svm|None', serve.fingerprint(*old), f['_version'], m_old)
    with serve.app.test_request_context():
        serve.g.owner = 'alice'
        m, _, n_pos = serve.svm_rank(tags='cv')
    assert n_pos == 5 and m is not m_old
    assert not np.allclose(m['parts'][0]['coef'], m_old['parts'][0]['coef'])


def test_concurrent_requests_train_a_model_once(client, corpus, features, monkeypatch):
    import serve
    from aslite import db

    login(client)
    for p in vision_papers(corpus)[:4]:
        client.post(f'/add/{p}/cv')
    calls = counting_fit(monkeypatch, delay=0.3)
    pos, neg = serve.training_set(db.get_tags('alice'), {}, 'cv')
    got = []
    threads = [
        threading.Thread(target=lambda: got.append(serve.get_model(features, 'alice', 'cv', pos, neg)))
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1 and len(got) == 4 and all(m is got[0] for m in got)
    assert not serve._models_inflight


def test_a_failed_fit_lets_the_next_one_try(client, corpus, features, monkeypatch):
    import serve
    from aslite import db

    login(client)
    for p in vision_papers(corpus)[:4]:
        client.post(f'/add/{p}/cv')
    pos, neg = serve.training_set(db.get_tags('alice'), {}, 'cv')
    real = serve.fit
    monkeypatch.setattr(serve, 'fit', lambda *a, **k: 1 / 0)
    try:
        serve.get_model(features, 'alice', 'cv', pos, neg)
    except ZeroDivisionError:
        pass
    assert not serve._models_inflight  # nobody waits forever on it
    monkeypatch.setattr(serve, 'fit', real)
    assert serve.get_model(features, 'alice', 'cv', pos, neg) is not None


def test_models_are_erased_with_their_tag_and_account(client, corpus):
    from aslite import db

    login(client)
    vision = vision_papers(corpus)
    for p in vision[:3]:
        client.post(f'/add/{p}/cv')
    client.post(f'/add/{vision[3]}/other')
    for tags in ('cv', 'other', 'all'):
        rank_json(client, f'/?rank=tags&tags={tags}&skip_have=yes')

    def stored():
        con = db._models_db()
        try:
            return {r[0] for r in con.execute("SELECT tags FROM models WHERE user='alice'")}
        finally:
            con.close()

    assert stored() == {'cv', 'other', 'all'}
    client.post('/del/cv')
    assert stored() == {'other'}  # 'all' had cv in it
    db.delete_user('alice')
    assert stored() == set()


def test_window_ranking_matches_a_full_sort(client, corpus, features):
    """only the time window gets scored; ranks and "top x%" must be those of the full ranking"""
    import serve
    from aslite import db, rank

    login(client)
    vision = vision_papers(corpus)
    for p in vision[:4]:
        client.post(f'/add/{p}/cv')
    papers = rank_json(client, '/?rank=tags&tags=cv&time_filter=2&skip_have=yes')
    pos, neg = serve.training_set(db.get_tags('alice'), {}, 'cv')
    pids, scores, _ = rank.rank(features, pos, neg)
    times = db.get_papers_db().times()
    expect = [(i, p) for i, p in enumerate(pids) if time.time() - times[p] < 2 * 86400 and p not in pos]
    assert [p['id'] for p in papers] == [p for _, p in expect][:25]
    assert [p['pct'] for p in papers] == [(i + 1) / len(pids) for i, _ in expect][:25]
    assert features['pids'] == sorted(features['pids'], key=times.get)  # compute.py: oldest first


def prefit_on(monkeypatch):
    import serve

    monkeypatch.setattr(serve, 'PREFIT', True)
    monkeypatch.setattr(serve, 'PREFIT_DELAY', 0.2)


def wait_for_prefit(timeout=20):
    """until no model is queued or training (and stays so for a moment)"""
    import serve

    quiet_since, end = None, time.time() + timeout
    while time.time() < end:
        busy = bool(serve._prefit_pending or serve._models_inflight)
        if busy:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = time.time()
        elif time.time() - quiet_since > 0.5:
            return
        time.sleep(0.02)
    raise AssertionError('background training did not finish')


def test_a_vote_trains_the_next_page_s_model_in_the_background(client, corpus, monkeypatch):
    login(client)
    prefit_on(monkeypatch)
    calls = counting_fit(monkeypatch)
    vision = vision_papers(corpus)
    for p in vision[:5]:  # a burst of votes: one fit per page, of the final state
        client.post(f'/add/{p}/cv')
    wait_for_prefit()
    assert len(calls) == 2  # cv and all
    rank_json(client, '/?rank=tags&tags=cv&skip_have=yes')
    rank_json(client, '/?rank=tags&tags=all&skip_have=yes')
    assert len(calls) == 2  # both pages found theirs ready


def test_a_vote_during_a_background_fit_gets_its_own(client, corpus, monkeypatch):
    import serve

    login(client)
    prefit_on(monkeypatch)
    calls = counting_fit(monkeypatch, delay=0.6)
    vision = vision_papers(corpus)
    for p in vision[:3]:
        client.post(f'/add/{p}/cv')
    end = time.time() + 10
    while not calls and time.time() < end:  # the first fit is running
        time.sleep(0.01)
    client.post(f'/add/{vision[3]}/cv')
    wait_for_prefit()
    n = len(calls)
    assert n >= 3  # the old state (cv, all) and the new one again
    monkeypatch.setattr(serve, 'PREFIT', False)
    assert rank_json(client, '/?rank=tags&tags=cv&skip_have=yes')
    assert len(calls) == n  # the model of the state after the second vote was ready


def test_a_page_with_its_models_cached_comes_with_its_results(client, corpus):
    import serve

    login(client)
    vision = vision_papers(corpus)
    for p in vision[:4]:
        client.post(f'/add/{p}/cv')
    client.post(f'/add/{vision[6]}/more')  # two tags: turns has a model per tag
    for mix in ('one', 'turns'):
        url = f'/?rank=tags&tags=all&mix={mix}&skip_have=yes'
        assert b'"pending": true' in client.get(url).data  # nothing trained yet: placeholder
        rank_json(client, url)
        serve._rank_cache.clear()
        serve._models.clear()  # another worker: the model is in models.sqlite
        r = client.get(url)
        assert b'"pending": false' in r.data and b'2601.' in r.data
    client.post(f'/add/{vision[4]}/cv')  # a new training set, not trained yet (PREFIT=0)
    assert b'"pending": true' in client.get('/?rank=tags&tags=all&skip_have=yes').data
