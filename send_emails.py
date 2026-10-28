"""
Sends every user with an email address their digest: for each of their tags, the best
papers of the last few days (aslite/digest.py). The digest is also stored, and the
email links to the /digest page that shows it with thumbs up/down.

Mail goes out over plain SMTP, configured with environment variables:
    SMTP_HOST      default localhost (in docker compose: the `mail` service)
    SMTP_PORT      default 25
    SMTP_TLS       none | starttls | ssl (default none, for a server on the same host/network)
    SMTP_USER, SMTP_PASSWORD   if the server wants a login
    MAIL_FROM      e.g. "arxiv-sanity <digest@example.org>"
    PUBLIC_URL     where the site is reachable, e.g. https://sanity.example.org

    uv run python send_emails.py --dry-run 1 -u alice --save digest.html   # preview
    uv run python send_emails.py --if-due   # what the docker updater runs every few minutes

--if-due sends only when the weekly slot in DIGEST_SCHEDULE (default "sun 08:00
Europe/Berlin", local time, so summer time included) has passed since the last scheduled
send, kept in data/last_digest. The first --if-due run only starts that clock: enabling the
digest never mails everyone on the spot. Manual runs (without --if-due) don't touch it.
"""

import os
import time
import html
import argparse
import smtplib
import datetime
import zoneinfo
from email.message import EmailMessage
from email.utils import make_msgid, formataddr, parseaddr

from itsdangerous import URLSafeSerializer

from aslite import db, digest
from aslite.db import load_features, get_papers_db, get_emails, get_secret_key

DIGEST_SCHEDULE = os.environ.get('DIGEST_SCHEDULE', 'sun 08:00 Europe/Berlin')
LAST_DIGEST_FILE = os.path.join(db.DATA_DIR, 'last_digest')
DAYS = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']


def last_slot(now, schedule=DIGEST_SCHEDULE):
    """unix time of the latest weekly slot at or before now, e.g. 'sun 08:00 Europe/Berlin'"""
    day, hhmm, tz = schedule.split()
    hour, minute = map(int, hhmm.split(':'))
    local = datetime.datetime.fromtimestamp(now, zoneinfo.ZoneInfo(tz))
    slot = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    slot -= datetime.timedelta(days=(local.weekday() - DAYS.index(day.lower()[:3])) % 7)
    if slot > local:
        slot -= datetime.timedelta(days=7)
    # rebuild from the wall-clock time, so a slot across a summer-time switch stays at 08:00
    slot = datetime.datetime(slot.year, slot.month, slot.day, hour, minute, tzinfo=zoneinfo.ZoneInfo(tz))
    return slot.timestamp()


def due(now):
    """for --if-due: whether a scheduled digest is due now; records it if so"""
    if not os.path.isfile(LAST_DIGEST_FILE):
        with open(LAST_DIGEST_FILE, 'w') as f:
            f.write(str(now))
        print(f"digest schedule started ({DIGEST_SCHEDULE}), the first one goes out at the next slot")
        return False
    with open(LAST_DIGEST_FILE) as f:
        last = float(f.read().strip() or 0)
    if last >= last_slot(now):
        return False
    with open(LAST_DIGEST_FILE, 'w') as f:
        f.write(str(now))
    return True


PUBLIC_URL = os.environ.get('PUBLIC_URL', 'http://localhost:5000').rstrip('/')
MAIL_FROM = os.environ.get('MAIL_FROM', 'arxiv-sanity <digest@localhost>')


def unsubscribe_url(user):
    token = URLSafeSerializer(get_secret_key(), salt='unsubscribe').dumps(user)
    return f'{PUBLIC_URL}/unsubscribe/{token}'


def render(user, sections, papers, days):
    """(subject, plain text, html) of one digest email"""
    n = sum(len(s['papers']) for s in sections)
    subject = f"{n} new papers for you, {time.strftime('%b %d')}"
    digest_url = f'{PUBLIC_URL}/digest'
    unsub = unsubscribe_url(user)
    e = html.escape

    text = [
        f"Hi {user}, the best papers of the last {days} days for each of your tags.",
        f"Rate them: {digest_url}",
        '',
    ]
    parts = []
    for s in sections:
        text.append(f"== {s['tag']}")
        parts.append(f'<h2 style="font-size:17px;margin:28px 0 8px;color:#b3261e">{e(s["tag"])}</h2>')
        for pid, _ in s['papers']:
            p = papers[pid]
            authors = ', '.join(a['name'] for a in p['authors'][:6]) + (' et al.' if len(p['authors']) > 6 else '')
            summary = ' '.join(p['summary'].split())
            summary = summary[:400] + ('…' if len(summary) > 400 else '')
            title = ' '.join(p['title'].split())
            text += [f"- {title}", f"  {authors}", f"  https://arxiv.org/abs/{pid}", '']
            parts.append(
                '<div style="margin:0 0 16px">'
                f'<a href="https://arxiv.org/abs/{pid}" style="font-size:15px;font-weight:600;color:#1a1a1a">{e(title)}</a>'
                f'<div style="font-size:13px;color:#666;margin:2px 0">{e(authors)} · {e(p["_time_str"])}</div>'
                f'<div style="font-size:14px;color:#333">{e(summary)}</div>'
                f'<div style="font-size:13px;margin-top:4px"><a href="{PUBLIC_URL}/?rank=pid&pid={pid}">similar papers</a>'
                f' · <a href="https://arxiv.org/pdf/{pid}">pdf</a></div>'
                '</div>'
            )
    text += ['', f"Unsubscribe: {unsub}"]
    body = f"""<!DOCTYPE html>
<html><body style="margin:0;padding:24px 16px;background:#f6f5f2;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif">
<div style="max-width:640px;margin:0 auto;background:#fff;border-radius:10px;padding:24px">
<p style="margin-top:0;font-size:15px">Hi {e(user)}, here are the best papers of the last {days} days for each of your tags.</p>
<p><a href="{digest_url}" style="display:inline-block;padding:10px 16px;background:#b3261e;color:#fff;border-radius:8px;text-decoration:none;font-weight:600">Rate them 👍 👎</a></p>
<p style="font-size:13px;color:#666">Each rating teaches your tag what you want, so next week's list gets better.</p>
{''.join(parts)}
<p style="font-size:12px;color:#888;margin-top:32px">You get this because you added your email at {e(PUBLIC_URL)}.
<a href="{unsub}" style="color:#888">Unsubscribe</a></p>
</div></body></html>"""
    return subject, '\n'.join(text), body


def send(to, subject, text, body, user, dry_run):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = MAIL_FROM
    msg['To'] = to
    msg['Message-ID'] = make_msgid(domain=parseaddr(MAIL_FROM)[1].split('@')[-1] or None)
    # one-click unsubscribe (RFC 8058), which Gmail and others look for in bulk mail
    msg['List-Unsubscribe'] = f'<{unsubscribe_url(user)}>'
    msg['List-Unsubscribe-Post'] = 'List-Unsubscribe=One-Click'
    msg.set_content(text)
    msg.add_alternative(body, subtype='html')
    if dry_run:
        return
    host, port = os.environ.get('SMTP_HOST', 'localhost'), int(os.environ.get('SMTP_PORT', '25'))
    tls = os.environ.get('SMTP_TLS', 'none')
    cls = smtplib.SMTP_SSL if tls == 'ssl' else smtplib.SMTP
    with cls(host, port, timeout=60) as s:
        if tls == 'starttls':
            s.starttls()
        if os.environ.get('SMTP_USER'):
            s.login(os.environ['SMTP_USER'], os.environ.get('SMTP_PASSWORD', ''))
        s.send_message(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sends the digest emails')
    parser.add_argument('-n', '--per-tag', type=int, default=8, help='papers per tag')
    parser.add_argument('-t', '--time-delta', type=int, default=7, help='papers from the last this many days')
    parser.add_argument('-d', '--dry-run', type=int, default=0, help='if 1, compose (and store) but do not send')
    parser.add_argument('-u', '--user', type=str, default='', help='only this user (for testing)')
    parser.add_argument('--save', type=str, default='', help='also write the html of the (last) email to this file')
    parser.add_argument('--feats', type=str, default='auto', help='auto|tfidf|emb|concat|fusion (see aslite/rank.py)')
    parser.add_argument('--model', type=str, default='auto', help='auto|svm|logreg (see aslite/rank.py)')
    parser.add_argument('--if-due', action='store_true', help='only send when the DIGEST_SCHEDULE slot has passed')
    args = parser.parse_args()
    if args.if_due and not due(time.time()):
        raise SystemExit(0)
    print(args)

    emails = get_emails()
    if args.user:
        emails = {u: e for u, e in emails.items() if u == args.user}
    if not emails or not db.features_exist():
        print("nobody to send to" if not emails else "no features yet, run compute.py")
        raise SystemExit(0)
    features = load_features()
    pdb = get_papers_db()
    times = pdb.times()

    num_sent = 0
    for user, email in sorted(emails.items()):
        sections = digest.compute_for(
            user, features, times, args.time_delta, args.per_tag, feats=args.feats, model=args.model
        )
        if not sections:
            print(f"skipping {user}: no recommendations (no tags, or nothing new)")
            continue
        db.save_digest(user, sections, args.time_delta)
        papers = pdb.get_many(p for s in sections for p, _ in s['papers'])
        subject, text, body = render(user, sections, papers, args.time_delta)
        if args.save:
            with open(args.save, 'w') as f:
                f.write(body)
        try:
            send(formataddr((user, email)), subject, text, body, user, args.dry_run)
        except (smtplib.SMTPException, OSError) as e:
            print(f"failed to send to {user}: {e}")
            continue
        num_sent += 1
        print(f"{'composed' if args.dry_run else 'sent'} digest for {user}: {subject}")
        time.sleep(0.5)  # no need to burst

    print(f"done, {num_sent} {'composed' if args.dry_run else 'sent'}")
