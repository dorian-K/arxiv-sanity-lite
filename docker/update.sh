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
while true; do
    now=$(date +%s)
    if [ $((now - last_daily)) -ge "${UPDATE_INTERVAL:-86400}" ]; then
        last_daily=$now
        echo "== $(date -u) fetching new papers"
        requested=0
        if [ -f data/compute_requested ]; then requested=1; fi
        rm -f data/compute_requested  # this run covers those papers too
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
    sleep 300
done
