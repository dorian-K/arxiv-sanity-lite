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
    db.set_email('alice', 'alice@example.org')
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
    assert 'https://sanity.example.org/digest' in body and '/unsubscribe/' in body
    assert db.get_digest('alice')['sections'][0]['tag'] == 'cv'  # stored for the page

    # the unsubscribe link works without logging in, one-click POST included
    import importlib

    import serve

    importlib.reload(serve)
    c = serve.app.test_client()
    token = body.split('/unsubscribe/')[1].split('"')[0]
    assert b'Stop the digest emails' in c.get(f'/unsubscribe/{token}').data
    assert c.post(f'/unsubscribe/{token}').status_code == 200
    assert db.get_user('alice')['email'] == ''
    assert c.post('/unsubscribe/forged').status_code == 400


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
