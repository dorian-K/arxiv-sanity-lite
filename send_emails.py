"""
Sends every user with an email address their digest: for each of their tags, the best
papers of the last few days (aslite/digest.py). The digest is also stored, and the
email links to the /digest page that shows it with thumbs up/down.

Only confirmed addresses get it (double opt-in, see serve.py). Mail goes out over plain
SMTP, configured as described in aslite/mail.py, plus
    PUBLIC_URL     where the site is reachable, e.g. https://sanity.example.org

    uv run python send_emails.py --dry-run 1 -u alice --save digest.html   # preview
    uv run python send_emails.py --if-due   # what the docker updater runs every few minutes

--if-due sends only when the weekly slot in DIGEST_SCHEDULE (default "sun 08:00
Europe/Berlin", local time, so summer time included) has passed since the last scheduled
send, kept in data/last_digest. The first --if-due run only starts that clock: enabling the
digest never mails everyone on the spot. Manual runs (without --if-due) don't touch it.

With AI summaries on (SUMMARIES_ENABLED=1, aslite/summary.py), every paper in a mail has one.
--prepare-if-due, run as often as --if-due, makes and stores the digests DIGEST_PREPARE_HOURS
(default 12) before the slot, so enrich_daemon.py has the time to summarize their papers from the
full text (one arxiv.org request per 15 s or more); --if-due then sends those digests, and any paper
still without a summary gets one from its abstract right before the mails go out.
"""

import os
import time
import html
import argparse
import smtplib
from concurrent.futures import ThreadPoolExecutor
import datetime
import zoneinfo
from email.utils import formataddr

from itsdangerous import URLSafeSerializer

from aslite import db, digest, mail, summary
from aslite.db import load_features, get_papers_db, get_emails, get_secret_key

DIGEST_SCHEDULE = os.environ.get('DIGEST_SCHEDULE', 'sun 08:00 Europe/Berlin')
PREPARE = float(os.environ.get('DIGEST_PREPARE_HOURS') or 12) * 3600
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
    """
    for --if-due: whether a scheduled digest is due now. the caller records the send with
    db.set_last_digest once it went out, so a run that fails (SMTP down) is tried again
    """
    last = db.get_last_digest()
    if last is None:
        db.set_last_digest(now)
        print(f"digest schedule started ({DIGEST_SCHEDULE}), the first one goes out at the next slot")
        return False
    return last < last_slot(now)


def next_slot(now, schedule=DIGEST_SCHEDULE):
    """unix time of the first weekly slot after now"""
    return last_slot(now + 7 * 86400, schedule)


def prepared(d, now):
    """whether the stored digest d was made for the slot that is due now (by --prepare-if-due, or by hand since)"""
    return bool(d) and d['time'] >= last_slot(now) - PREPARE - 3600 and d['time'] > (db.get_last_digest() or 0)


def _summarize(pid):
    """a summary from the paper's abstract (enrich_daemon.py didn't get to its text in time)"""
    with get_papers_db() as pdb, db.get_refs_db() as rdb:
        d = pdb.get_many([pid]).get(pid)
        if d is None:
            return
        try:
            fields = summary.summarize(' '.join(d['title'].split()), ' '.join(d['summary'].split()), '')
        except Exception as e:
            print(f"no summary for {pid}: {type(e).__name__}: {e}")
            rdb.put_summary(pid, 'error')
            return
        rdb.put_summary(pid, 'ok', summary.MODEL, 'abstract', fields)


def ensure_summaries(pids):
    """{pid: summary} for the papers of the mails; those without one get it now, all at once (not capped)"""
    if not summary.enabled() or not pids:
        return {}
    with db.get_refs_db() as rdb:
        missing = [p for p in pids if (rdb.summaries_of([p]).get(p) or {}).get('status') != 'ok']
    if missing:
        print(f"summarizing {len(missing)} papers from their abstracts")
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(_summarize, missing))
    with db.get_refs_db() as rdb:
        return {p: s for p, s in rdb.summaries_of(pids).items() if s['status'] == 'ok'}


def prepare(features, times, emails, now, per_tag=digest.PER_TAG, **kw):
    """makes the digests of the coming slot ahead of it, once; returns how many"""
    slot = next_slot(now)
    if now < slot - PREPARE:
        return 0
    n = 0
    for user in sorted(emails):
        d = db.get_digest(user)
        if d and d['time'] >= slot - PREPARE:
            continue  # done already
        days = db.get_digest_days(user)
        db.save_digest(user, digest.compute_for(user, features, times, days, per_tag, **kw), days)
        n += 1
    return n


# `or`: docker compose passes unset variables as empty strings
PUBLIC_URL = (os.environ.get('PUBLIC_URL') or 'http://localhost:5000').rstrip('/')


def unsubscribe_url(user):
    """a link that works until the user unsubscribes or confirms another address (see serve.py)"""
    confirmed = (db.get_user(user) or {}).get('email_confirmed')
    token = URLSafeSerializer(get_secret_key(), salt='unsubscribe').dumps([user, confirmed])
    return f'{PUBLIC_URL}/unsubscribe?token={token}'


AI_LABELS = (('problem', 'Problem'), ('approach', 'Approach'), ('result', 'Result'), ('caveat', 'Caveat'))


def render(user, sections, papers, days, ais=None):
    """(subject, plain text, html) of one digest email; ais: {pid: AI summary}, shown instead of the abstract"""
    ais = ais or {}
    n = sum(len(s['papers']) for s in sections)
    subject = f"{n} new papers for you, {time.strftime('%b %d')}"
    digest_url = f'{PUBLIC_URL}/digest'
    settings_url = f'{PUBLIC_URL}/profile#digest-sizes'
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
            abstract = ' '.join(p['summary'].split())
            abstract = abstract[:400] + ('…' if len(abstract) > 400 else '')
            title = ' '.join(p['title'].split())
            ai = ais.get(pid)
            lines = [(label, ai[k]) for k, label in AI_LABELS if ai and ai.get(k)]
            text += [f"- {title}", f"  {authors}", *(f"  {label}: {v}" for label, v in lines)]
            text += [f"  https://arxiv.org/abs/{pid}", '']
            if lines:
                about = (
                    '<ul style="margin:4px 0;padding-left:18px;font-size:14px;color:#333">'
                    + ''.join(f'<li style="margin:2px 0"><b>{label}:</b> {e(v)}</li>' for label, v in lines)
                    + '</ul><div style="font-size:11px;color:#999">AI summary from the paper\'s '
                    + ('abstract' if ai.get('source') == 'abstract' else 'full text')
                    + ', may contain errors</div>'
                )
            else:
                about = f'<div style="font-size:14px;color:#333">{e(abstract)}</div>'
            parts.append(
                '<div style="margin:0 0 16px">'
                f'<a href="https://arxiv.org/abs/{pid}" style="font-size:15px;font-weight:600;color:#1a1a1a">{e(title)}</a>'
                f'<div style="font-size:13px;color:#666;margin:2px 0">{e(authors)} · {e(p["_time_str"])}</div>'
                + about
                + f'<div style="font-size:13px;margin-top:4px"><a href="{PUBLIC_URL}/?rank=pid&pid={pid}">similar papers</a>'
                f' · <a href="https://arxiv.org/pdf/{pid}">pdf</a></div>'
                '</div>'
            )
    text += ['', f"Papers per tag can be changed in your settings: {settings_url}", f"Unsubscribe: {unsub}"]
    body = f"""<!DOCTYPE html>
<html><body style="margin:0;padding:24px 16px;background:#f6f5f2;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif">
<div style="max-width:640px;margin:0 auto;background:#fff;border-radius:10px;padding:24px">
<p style="margin-top:0;font-size:15px">Hi {e(user)}, here are the best papers of the last {days} days for each of your tags.</p>
<p><a href="{digest_url}" style="display:inline-block;padding:10px 16px;background:#b3261e;color:#fff;border-radius:8px;text-decoration:none;font-weight:600">Rate them 👍 👎</a></p>
<p style="font-size:13px;color:#666">Each rating teaches your tag what you want, so next week's list gets better.</p>
{''.join(parts)}
<p style="font-size:12px;color:#888;margin-top:32px">How many papers each tag gets can be changed
in your <a href="{settings_url}" style="color:#888">settings</a>, down to none.<br>
You get this because you added your email at {e(PUBLIC_URL)}.
<a href="{unsub}" style="color:#888">Unsubscribe</a></p>
</div></body></html>"""
    return subject, '\n'.join(text), body


def send(to, subject, text, body, user, dry_run):
    # one-click unsubscribe (RFC 8058), which Gmail and others look for in bulk mail
    headers = {'List-Unsubscribe': f'<{unsubscribe_url(user)}>', 'List-Unsubscribe-Post': 'List-Unsubscribe=One-Click'}
    msg = mail.message(to, subject, text, body, headers)
    if not dry_run:
        mail.send(msg)
    return msg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sends the digest emails')
    parser.add_argument(
        '-n', '--per-tag', type=int, default=digest.PER_TAG, help="papers per tag, where the user didn't pick a number"
    )
    parser.add_argument(
        '-t',
        '--time-delta',
        type=int,
        default=0,
        help="papers from the last this many days (default: each user's pick)",
    )
    parser.add_argument('-d', '--dry-run', type=int, default=0, help='if 1, compose (and store) but do not send')
    parser.add_argument('-u', '--user', type=str, default='', help='only this user (for testing)')
    parser.add_argument('--save', type=str, default='', help='also write the html of the (last) email to this file')
    parser.add_argument('--feats', type=str, default='auto', help='auto|tfidf|emb|concat|fusion (see aslite/rank.py)')
    parser.add_argument('--model', type=str, default='auto', help='auto|svm|logreg (see aslite/rank.py)')
    parser.add_argument('--if-due', action='store_true', help='only send when the DIGEST_SCHEDULE slot has passed')
    parser.add_argument(
        '--prepare-if-due',
        action='store_true',
        help='with AI summaries on: make the digests DIGEST_PREPARE_HOURS before the slot (send nothing)',
    )
    args = parser.parse_args()
    started = time.time()
    if args.prepare_if_due:
        if not summary.enabled() or db.get_last_digest() is None or not db.features_exist():
            raise SystemExit(0)  # nothing to summarize, or the schedule hasn't started
        if started >= next_slot(started) - PREPARE:
            n = prepare(load_features(), get_papers_db().times(), get_emails(), started, args.per_tag)
            if n:
                print(f"prepared {n} digests for the next slot, so their papers get summaries in time")
        raise SystemExit(0)
    if args.if_due and not due(started):
        raise SystemExit(0)
    print(args)
    if not args.dry_run and not os.environ.get('PUBLIC_URL'):
        # every link in the mail would point at localhost
        print("PUBLIC_URL is not set, not sending (set it to the site's address, e.g. https://sanity.example.org)")
        raise SystemExit(1)

    emails = get_emails()
    if args.user:
        emails = {u: e for u, e in emails.items() if u == args.user}
    if not emails:
        print("nobody to send to")
        if args.if_due:
            db.set_last_digest(started)
        raise SystemExit(0)
    if not db.features_exist():
        print("no features yet, run compute.py")  # due again on the next try
        raise SystemExit(0)
    features = load_features()
    pdb = get_papers_db()
    times = pdb.times()

    todo = []
    for user, email in sorted(emails.items()):
        d = db.get_digest(user)
        if args.if_due and summary.enabled() and prepared(d, started):
            sections, days = d['sections'], d['days']  # made ahead, so its papers have their summaries
        else:
            days = args.time_delta or db.get_digest_days(user)
            sections = digest.compute_for(user, features, times, days, args.per_tag, feats=args.feats, model=args.model)
            if sections:
                db.save_digest(user, sections, days)
        if not sections:
            print(f"skipping {user}: no recommendations (no tags, or nothing new)")
            continue
        todo.append((user, email, sections, days))
    ais = ensure_summaries(list(dict.fromkeys(p for _, _, ss, _ in todo for s in ss for p, _ in s['papers'])))

    num_sent = num_failed = 0
    for user, email, sections, days in todo:
        papers = pdb.get_many(p for s in sections for p, _ in s['papers'])
        subject, text, body = render(user, sections, papers, days, ais)
        if args.save:
            with open(args.save, 'w') as f:
                f.write(body)
        try:
            send(formataddr((user, email)), subject, text, body, user, args.dry_run)
        except (smtplib.SMTPException, OSError) as e:
            print(f"failed to send to {user}: {e}")
            num_failed += 1
            continue
        num_sent += 1
        print(f"{'composed' if args.dry_run else 'sent'} digest for {user}: {subject}")
        time.sleep(0.5)  # no need to burst

    print(f"done, {num_sent} {'composed' if args.dry_run else 'sent'}, {num_failed} failed")
    if args.if_due:
        if num_sent or not num_failed:
            db.set_last_digest(started)  # this slot is done (those that failed amid successes wait a week)
        else:
            print("every send failed: trying again on the next run")
            raise SystemExit(1)
