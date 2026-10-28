#!/bin/sh
# The updater service: fetch new papers from arXiv, then recompute the features the web
# service ranks with (it picks up the new data/features by itself). Repeats every
# UPDATE_INTERVAL seconds. arxiv_daemon.py pauses 3-6s between its requests and stops
# as soon as it reaches papers it already has, so a daily run is a handful of requests.
set -u

while true; do
    echo "== $(date -u) fetching new papers"
    if python arxiv_daemon.py --num "${DAEMON_NUM:-2000}"; then
        echo "== $(date -u) computing features ${EMBED_MODEL:+with $EMBED_MODEL}"
        # word-splitting COMPUTE_ARGS is intended
        # shellcheck disable=SC2086
        python compute.py ${EMBED_MODEL:+--embed_model "$EMBED_MODEL"} ${COMPUTE_ARGS:-} \
            || echo "compute.py failed, keeping the previous features"
    else
        echo "no new papers (or arXiv unreachable), nothing to compute"
    fi
    echo "== $(date -u) done, next run in ${UPDATE_INTERVAL:-86400}s"
    sleep "${UPDATE_INTERVAL:-86400}"
done
