#!/bin/sh
# The updater service: fetch new papers from arXiv, then recompute the features the web
# service ranks with (it picks up the new data/features by itself). That happens every
# UPDATE_INTERVAL seconds. arxiv_daemon.py pauses 3-6s between its requests and stops
# as soon as it reaches papers it already has, so a daily run is a handful of requests.
#
# In between, papers users added through the web page (/import) get their features
# within COMPUTE_MIN_INTERVAL seconds: serve.py leaves data/compute_requested behind.
set -u

compute() {
    echo "== $(date -u) computing features ${EMBED_MODEL:+with $EMBED_MODEL}"
    # word-splitting COMPUTE_ARGS is intended
    # shellcheck disable=SC2086
    python compute.py ${EMBED_MODEL:+--embed_model "$EMBED_MODEL"} ${COMPUTE_ARGS:-} \
        || echo "compute.py failed, keeping the previous features"
    last_compute=$(date +%s)
}

last_daily=0
last_compute=0

# the weekly digest emails, when DIGEST_ENABLED=1: send_emails.py --if-due sends at the
# DIGEST_SCHEDULE slot (default sunday 08:00 Europe/Berlin) and otherwise exits right away.
# its first run only starts the clock, so turning the digest on never mails everyone on the
# spot (send a test with `docker compose exec arxiv-resanity-updater python send_emails.py -u <you>`)
digest() {
    [ "${DIGEST_ENABLED:-0}" = 1 ] || return 0
    python send_emails.py --if-due --time-delta "${DIGEST_DAYS:-7}" || echo "send_emails.py failed"
}
while true; do
    now=$(date +%s)
    if [ $((now - last_daily)) -ge "${UPDATE_INTERVAL:-86400}" ]; then
        last_daily=$now
        echo "== $(date -u) fetching new papers"
        requested=0
        if [ -f data/compute_requested ]; then requested=1; fi
        rm -f data/compute_requested  # this run covers those papers too
        # one copy of the accounts/tags/votes database, refreshed daily, in data/backups/
        python -c "from aslite.db import backup_users; print('backed up users to', backup_users())" \
            || echo "backup failed"
        if python arxiv_daemon.py --num "${DAEMON_NUM:-2000}" || [ "$requested" = 1 ]; then
            compute
        else
            echo "no new papers (or arXiv unreachable)"
        fi
    elif [ -f data/compute_requested ] && [ $((now - last_compute)) -ge "${COMPUTE_MIN_INTERVAL:-3600}" ]; then
        echo "== $(date -u) users added papers"
        rm -f data/compute_requested  # before computing, so papers added meanwhile trigger another run
        compute
    fi
    digest
    sleep 300
done
