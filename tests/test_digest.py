import os
import subprocess
import sys

from aslite import db, digest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def vision(corpus):
    return [p for p in corpus if corpus[p]['tags'][0]['term'] == 'cs.CV']


def test_digest_recommends_new_untagged_papers(features, corpus):
    v = vision(corpus)
    times = {p: corpus[p]['_time'] for p in corpus}
    sections = digest.compute(features, times, {'cv': set(v[:4])}, {'cv': {v[4]}}, days=7, per_tag=5)
    assert [s['tag'] for s in sections] == ['cv']
    got = [p for p, _ in sections[0]['papers']]
    assert len(got) == 5 and not set(got) & set(v[:5])  # nothing already tagged or downvoted
    assert sum(p in v for p in got) >= 4  # and on topic
    # nothing new in the window: no section
    assert digest.compute(features, times, {'cv': set(v[:4])}, {}, days=7, now=max(times.values()) + 30 * 86400) == []


def test_digest_shows_each_paper_once_under_its_best_tag(features, corpus):
    v = vision(corpus)
    times = {p: corpus[p]['_time'] for p in corpus}
    # two tags about the same topic recommend the same papers; the second is smaller
    tags = {'cv': set(v[:5]), 'cv2': set(v[5:7])}
    sections = digest.compute(features, times, tags, {}, days=7, per_tag=4)
    got = [p for s in sections for p, _ in s['papers']]
    assert len(got) == len(set(got)) == 8  # no duplicates, and each tag still gets its 4
    # a paper that would rank higher in the other tag only lands here if the other tag filled
    # up with papers that rank even higher there (the best fits are placed first)
    from aslite.rank import score

    scores = {t: score(features, tags[t])[0] for t in tags}

    def higher(t, p):
        sc = scores[t]
        return int((sc > sc[features['ptoi'][p]]).sum())

    picked = {s['tag']: [p for p, _ in s['papers']] for s in sections}
    for t, other in (('cv', 'cv2'), ('cv2', 'cv')):
        for p in picked[t]:
            if higher(other, p) < higher(t, p):
                assert all(higher(other, q) <= higher(other, p) for q in picked[other])


def test_digest_page_and_votes(client, corpus):
    from conftest import login

    v = vision(corpus)
    login(client)
    assert b'No digest yet' in client.get('/digest').data
    for p in v[:4]:
        client.post(f'/add/{p}/cv')
    assert client.post('/digest').status_code == 302  # make one now
    d = db.get_digest('alice')
    assert d['days'] == 7 and d['sections'][0]['tag'] == 'cv'
    page = client.get('/digest').data.decode()
    first = d['sections'][0]['papers'][0][0]
    assert first in page and 'var digest' in page


def test_emails_and_unsubscribe(features, corpus, monkeypatch, tmp_path):
    v = vision(corpus)
    for p in v[:4]:
        db.tag_paper('alice', p, 'cv')
    db.request_email('alice', 'alice@example.org')
    db.confirm_email('alice', 'alice@example.org')  # double opt-in done
    out = tmp_path / 'digest.html'
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, 'send_emails.py'), '--dry-run', '1', '--save', str(out)],
        capture_output=True,
        text=True,
        env=dict(os.environ, PUBLIC_URL='https://sanity.example.org'),
    )
    assert r.returncode == 0, r.stderr
    assert 'composed digest for alice' in r.stdout
    body = out.read_text()
    assert 'https://sanity.example.org/digest' in body and '/unsubscribe?token=' in body
    assert 'https://sanity.example.org/profile#digest-sizes' in body  # where to change the papers per tag
    assert db.get_digest('alice')['sections'][0]['tag'] == 'cv'  # stored for the page

    # the unsubscribe link works without logging in, one-click POST included
    import importlib

    import serve

    importlib.reload(serve)
    c = serve.app.test_client()
    path = '/unsubscribe?token=' + body.split('/unsubscribe?token=')[1].split('"')[0]
    assert b'Stop the digest emails' in c.get(path).data
    assert b'Unsubscribed' in c.post(path).data
    assert db.get_user('alice')['email'] == ''
    assert c.post('/unsubscribe?token=forged').status_code == 400
    # an old link can't touch an address confirmed later
    db.request_email('alice', 'new@example.org')
    db.confirm_email('alice', 'new@example.org')
    assert b'Already unsubscribed' in c.post(path).data
    assert db.get_user('alice')['email'] == 'new@example.org'


def test_smtp_message(monkeypatch):
    import smtplib

    import send_emails

    sent = []

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            sent.append((host, port))

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def send_message(self, msg):
            sent.append(msg)

    monkeypatch.setattr(smtplib, 'SMTP', FakeSMTP)
    monkeypatch.setattr(send_emails, 'get_secret_key', lambda: 'k')
    send_emails.send('Alice <a@example.org>', 'subj', 'text', '<p>html</p>', 'alice', dry_run=False)
    (host, port), msg = sent
    assert (host, port) == ('localhost', 25)
    assert msg['To'] == 'Alice <a@example.org>' and msg['List-Unsubscribe-Post'] == 'List-Unsubscribe=One-Click'
    assert {p.get_content_type() for p in msg.iter_parts()} == {'text/plain', 'text/html'}


def test_schedule_is_sunday_8_berlin_time():
    import datetime
    import zoneinfo

    from send_emails import last_slot

    tz = zoneinfo.ZoneInfo('Europe/Berlin')
    at = lambda *a: datetime.datetime(*a, tzinfo=tz).timestamp()  # noqa: E731
    sched = 'sun 08:00 Europe/Berlin'
    # sunday 2026-11-01 (winter time, CET) and sunday 2026-03-29, the day summer time starts
    assert last_slot(at(2026, 11, 1, 8, 0), sched) == at(2026, 11, 1, 8, 0)
    assert last_slot(at(2026, 11, 1, 7, 59), sched) == at(2026, 10, 25, 8, 0)
    assert last_slot(at(2026, 11, 4, 12, 0), sched) == at(2026, 11, 1, 8, 0)  # a wednesday
    assert last_slot(at(2026, 3, 29, 9, 0), sched) == at(2026, 3, 29, 8, 0)  # 08:00 CEST, not 09:00
    assert datetime.datetime.fromtimestamp(last_slot(at(2026, 3, 30, 0, 0), sched), tz).hour == 8


def test_if_due_waits_for_the_next_slot(features, monkeypatch):
    import send_emails

    t0 = 1793000000.0  # a monday morning
    assert not send_emails.due(t0)  # first run: only starts the clock
    assert not send_emails.due(t0 + 3600)  # same week
    assert send_emails.due(t0 + 7 * 86400)  # a sunday slot has passed
    assert send_emails.due(t0 + 7 * 86400 + 600)  # still due until a run records that it went out
    db.set_last_digest(t0 + 7 * 86400 + 600)
    assert not send_emails.due(t0 + 7 * 86400 + 1200)  # then only once


def test_neutral_hides_without_training(client, corpus):
    from conftest import login

    v = vision(corpus)
    login(client)
    for p in v[:4]:
        client.post(f'/add/{p}/cv')
    client.post('/digest')
    first = db.get_digest('alice')['sections'][0]['papers'][0][0]
    assert client.post(f'/seen/{first}/cv').data == b'ok'
    assert db.get_tags('alice', neg=db.SEEN) == {'cv': {first}}
    assert first not in db.get_tags('alice')['cv'] and db.get_tags('alice', neg=True) == {}  # not a like or downvote
    client.post('/digest')  # the next digest leaves it out
    assert first not in [p for s in db.get_digest('alice')['sections'] for p, _ in s['papers']]
    client.post(f'/add/{first}/cv')  # liking replaces neutral
    assert db.get_tags('alice', neg=db.SEEN) == {}
    assert client.post(f'/unseen/{first}/cv').data.startswith(b"paper")


def test_double_opt_in(client, corpus, monkeypatch):
    from conftest import login

    from aslite import mail

    sent = []
    monkeypatch.setattr(mail, 'send', sent.append)
    login(client)
    assert client.post('/register_email', data={'email': 'alice@example.org'}).status_code == 302
    assert db.get_emails() == {}  # nothing goes out before the owner confirms
    assert db.get_user('alice')['pending_email'] == 'alice@example.org'
    (msg,) = sent
    assert msg['To'] == 'alice@example.org'
    link = next(w for w in msg.get_content().split() if '/confirm_email?token=' in w)
    path = link[link.index('/confirm_email') :]
    # opening the link (or a mail scanner prefetching it) only shows a button
    assert b'Yes, send me the digest' in client.get(path).data and db.get_emails() == {}
    assert b'Confirmed' in client.post(path).data
    assert db.get_emails() == {'alice': 'alice@example.org'} and db.get_user('alice')['email_confirmed']
    assert b'Nothing to confirm' in client.post(path).data  # a link works once
    assert client.post('/confirm_email?token=forged').status_code == 400
    # removing the address works at once and needs no confirmation
    client.post('/register_email', data={'email': ''})
    assert db.get_emails() == {} and db.get_user('alice')['email'] == ''
    # the form can't be used to flood someone with confirmation mails
    codes = [client.post('/register_email', data={'email': 'victim@example.org'}).status_code for _ in range(7)]
    assert codes.count(429) >= 1 and len(sent) <= 6


def test_export_and_delete_account(client, corpus):
    import json

    from conftest import login

    pid = next(iter(corpus))
    login(client)
    client.post(f'/add/{pid}/cv')
    r = client.get('/export')
    data = json.loads(r.data)
    assert 'attachment' in r.headers['Content-Disposition']
    assert data['username'] == 'alice' and data['tags'] == {'cv': {'liked': [pid]}}
    assert 'pw_hash' not in r.data.decode() and 'scrypt' not in r.data.decode()
    assert client.post('/delete_account', data={'password': 'wrong one'}).status_code == 403
    assert db.get_user('alice') is not None
    assert b'Account deleted' in client.post('/delete_account', data={'password': 'correct horse'}).data
    assert db.get_user('alice') is None and db.get_tags('alice') == {} and db.get_digest('alice') is None
    assert client.post(f'/add/{pid}/cv').data.startswith(b'error, not logged in')  # logged out too


def test_legal_pages_and_no_third_parties(client):
    for url in ['/privacy', '/imprint']:
        assert client.get(url).status_code == 200
    page = client.get('/').data.decode()
    assert 'cdn' not in page and 'jsdelivr' not in page and '/static/vendor/' in page
    csp = client.get('/').headers['Content-Security-Policy']
    assert "default-src 'self'" in csp and 'http' not in csp
    assert '/privacy' in page and '/imprint' in page  # linked from every page


def test_papers_per_tag_in_the_digest(client, corpus):
    from conftest import login

    v = vision(corpus)
    lang = [p for p in corpus if p not in v]
    login(client)
    for p in v[:4]:
        client.post(f'/add/{p}/cv')
    for p in lang[:4]:
        client.post(f'/add/{p}/nlp')
    assert b'Papers per tag' in client.get('/profile').data
    r = client.post('/digest_sizes', data={'tag': ['cv', 'nlp', 'nope'], 'n': ['5', '0', '5']})
    assert r.status_code == 302 and db.get_digest_sizes('alice') == {'cv': 5, 'nlp': 0}  # unknown tags ignored
    client.post('/digest')
    d = db.get_digest('alice')
    assert [(s['tag'], len(s['papers'])) for s in d['sections']] == [('cv', 5)]  # 0 leaves nlp out
    # back to the default number stores nothing; bounds are enforced
    client.post('/digest_sizes', data={'tag': ['cv', 'nlp'], 'n': [str(digest.PER_TAG), '999']})
    assert db.get_digest_sizes('alice') == {'nlp': digest.MAX_PER_TAG}
    assert db.export_user('alice')['digest_papers_per_tag'] == {'nlp': digest.MAX_PER_TAG}
    client.post('/del/nlp')
    assert db.get_digest_sizes('alice') == {}


def test_digest_hides_a_downvote_only_for_its_tag(features, corpus):
    v = vision(corpus)
    times = {p: corpus[p]['_time'] for p in corpus}
    tags = {'cv': set(v[:4]), 'images': set(v[:4])}
    first = digest.compute(features, times, tags, {}, per_tag=3)[0]['papers'][0][0]
    sections = {s['tag']: [p for p, _ in s['papers']] for s in digest.compute(features, times, tags, {'cv': {first}})}
    assert first not in sections['cv'] and first in sections['images']
