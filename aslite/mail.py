"""
Sending mail over plain SMTP, for the digest (send_emails.py) and the confirmation mails
of the double opt-in (serve.py). Configured with environment variables:
    SMTP_HOST      default localhost (in docker compose: the `mail` service)
    SMTP_PORT      default 25
    SMTP_TLS       none | starttls | ssl (default none, for a server on the same host/network)
    SMTP_USER, SMTP_PASSWORD   if the server wants a login
    MAIL_FROM      e.g. "Arxiv Resanity <digest@example.org>"
"""

import os
import smtplib
from email.message import EmailMessage
from email.utils import make_msgid, parseaddr

MAIL_FROM = os.environ.get('MAIL_FROM') or 'Arxiv Resanity <digest@localhost>'  # compose passes unset as ''


def message(to, subject, text, html=None, headers=None):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = MAIL_FROM
    msg['To'] = to
    msg['Message-ID'] = make_msgid(domain=parseaddr(MAIL_FROM)[1].split('@')[-1] or None)
    for k, v in (headers or {}).items():
        msg[k] = v
    msg.set_content(text)
    if html:
        msg.add_alternative(html, subtype='html')
    return msg


def send(msg):
    host, port = os.environ.get('SMTP_HOST', 'localhost'), int(os.environ.get('SMTP_PORT', '25'))
    tls = os.environ.get('SMTP_TLS', 'none')
    cls = smtplib.SMTP_SSL if tls == 'ssl' else smtplib.SMTP
    with cls(host, port, timeout=60) as s:
        if tls == 'starttls':
            s.starttls()
        if os.environ.get('SMTP_USER'):
            s.login(os.environ['SMTP_USER'], os.environ.get('SMTP_PASSWORD', ''))
        s.send_message(msg)
