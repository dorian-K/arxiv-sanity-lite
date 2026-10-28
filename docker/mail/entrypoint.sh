#!/bin/sh
# configures postfix + opendkim from the environment, then runs both in the foreground
#   MAIL_DOMAIN    the domain mail comes from (MAIL_FROM's), e.g. example.org
#   MAIL_HOSTNAME  this server's name, must match the VPS's reverse DNS, e.g. vps.example.org
#   DKIM_SELECTOR  default "asl"
#   RELAYHOST      optional: hand everything to this smarthost instead, e.g. [smtp.example.org]:587
set -eu
: "${MAIL_DOMAIN:?set MAIL_DOMAIN}"
MAIL_HOSTNAME=${MAIL_HOSTNAME:-$MAIL_DOMAIN}
SEL=${DKIM_SELECTOR:-asl}
KEYDIR=/dkim/$MAIL_DOMAIN

# the dkim key lives in a volume, so it survives rebuilds (the dns record must not change)
if [ ! -f "$KEYDIR/$SEL.private" ]; then
    mkdir -p "$KEYDIR"
    opendkim-genkey -b 2048 -r -s "$SEL" -d "$MAIL_DOMAIN" -D "$KEYDIR"
fi
chown -R opendkim:opendkim /dkim
chmod 600 "$KEYDIR/$SEL.private"

cat > /etc/opendkim.conf <<CONF
Syslog                  no
LogWhy                  yes
Mode                    s
Domain                  $MAIL_DOMAIN
Selector                $SEL
KeyFile                 $KEYDIR/$SEL.private
Canonicalization        relaxed/simple
SignatureAlgorithm      rsa-sha256
OversignHeaders         From
Socket                  inet:8891@127.0.0.1
# sign what the app containers hand in (by default only mail from 127.0.0.1 is signed)
InternalHosts           127.0.0.1, 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
UserID                  opendkim
PidFile                 /run/opendkim/opendkim.pid
CONF
mkdir -p /run/opendkim && chown opendkim:opendkim /run/opendkim

postconf -e \
    "myhostname = $MAIL_HOSTNAME" \
    "mydomain = $MAIL_DOMAIN" \
    "myorigin = $MAIL_DOMAIN" \
    "mydestination =" \
    "inet_interfaces = all" \
    "inet_protocols = ${INET_PROTOCOLS:-ipv4}" \
    "mynetworks = 127.0.0.0/8 10.0.0.0/8 172.16.0.0/12 192.168.0.0/16" \
    "smtpd_relay_restrictions = permit_mynetworks, reject" \
    "smtpd_recipient_restrictions = permit_mynetworks, reject" \
    "relayhost = ${RELAYHOST:-}" \
    "smtp_tls_security_level = may" \
    "smtp_tls_CAfile = /etc/ssl/certs/ca-certificates.crt" \
    "smtp_tls_loglevel = 1" \
    "smtpd_milters = inet:127.0.0.1:8891" \
    "non_smtpd_milters = inet:127.0.0.1:8891" \
    "milter_default_action = tempfail" \
    "maillog_file = /dev/stdout" \
    "message_size_limit = 10240000"
# tempfail above: if opendkim is down, mail waits in the queue instead of going out unsigned.
# no chroot: the chroot would need copies of resolv.conf and friends inside the spool
postconf -F '*/*/chroot = n'

echo "=============================================================================="
echo " DNS records for $MAIL_DOMAIN (set once; the key is kept in the dkim volume):"
echo "   $SEL._domainkey.$MAIL_DOMAIN  TXT  \"$(grep -o '"[^"]*"' "$KEYDIR/$SEL.txt" | tr -d '"\n')\""
echo "   $MAIL_DOMAIN  TXT  \"v=spf1 a:$MAIL_HOSTNAME -all\""
echo "   _dmarc.$MAIL_DOMAIN  TXT  \"v=DMARC1; p=none\""
echo "   and reverse DNS (PTR) of this server's IP -> $MAIL_HOSTNAME, in your VPS provider's panel"
echo "=============================================================================="

opendkim -x /etc/opendkim.conf
exec postfix start-fg
