"""
Fills the embedding cache (data/embeddings.sqlite) slowly in the background, so new papers
and a big backlog (e.g. after import_snapshot.py --years 40) never keep the server busy for
long. It embeds a few papers, then sleeps, so that it works about --duty of the time
(EMBED_DUTY, default 0.1), at the lowest CPU priority, on --threads cores (EMBED_THREADS,
default 2). At 10% of one CPU like a Ryzen 7 1700 that is several thousand papers a day:
the ~1000 new ones, and the rest goes to the backlog.

Order: papers someone tagged or downvoted (they train the models), then the papers of the
last --new_days days, newest first, then everything else, newest first. When none of the
first two is left after it embedded some, it asks for a compute.py run (db.request_compute,
which the docker updater picks up within COMPUTE_MIN_INTERVAL), so new papers show up in
recommendations the same day. compute.py itself only uses the cache (EMBED_BUDGET=0 in
docker); papers without an embedding are left out of recommendations until they have one.

Embeddings computed elsewhere (gpu_embed.sh) are merged into the cache, never replace it.

    EMBED_MODEL=st:google/embeddinggemma-2 uv run python embed_daemon.py
    uv run python embed_daemon.py --duty 1 --once   # everything missing, flat out, then exit
"""

import os
import time
import argparse

from aslite import db
from aslite.db import EmbeddingCache, get_papers_db
from aslite.embed import get_backend, paper_text

REFRESH = 600  # seconds between looking for new papers and tags


def queue(cache, new_days, now=None):
    """(missing pids, best first; how many of them are urgent: tagged, or from the last new_days days)"""
    now = now or time.time()
    with get_papers_db() as pdb:
        times = pdb.times()
    have = cache.pids()
    tagged = set()
    for neg in (False, True):
        for tags in db.get_all_tags(neg=neg).values():
            tagged = tagged.union(*tags.values())
    missing = [p for p in times if p not in have]
    new = now - new_days * 86400

    def rank(p):
        t = times[p]
        return (0 if p in tagged else 1 if t > new else 2, -t)

    missing.sort(key=rank)
    urgent = sum(1 for p in missing if p in tagged or times[p] > new)
    return missing, urgent


def main():
    parser = argparse.ArgumentParser(description='embeds papers slowly in the background')
    parser.add_argument('--model', default=os.environ.get('EMBED_MODEL', ''), help='defaults to $EMBED_MODEL')
    parser.add_argument('--duty', type=float, default=float(os.environ.get('EMBED_DUTY') or 0.1),
                        help='share of the time spent embedding, 0-1 (default $EMBED_DUTY or 0.1)')
    parser.add_argument('--threads', type=int, default=int(os.environ.get('EMBED_THREADS') or 2),
                        help='CPU threads for the model (default $EMBED_THREADS or 2)')
    parser.add_argument('--batch', type=int, default=8, help='papers per step')
    parser.add_argument('--new_days', type=float, default=30, help='papers this new come before the backlog')
    parser.add_argument('--once', action='store_true', help='exit when nothing is missing')
    args = parser.parse_args()
    if not args.model:
        raise SystemExit('no model: set EMBED_MODEL or pass --model')
    duty = min(1.0, max(0.01, args.duty))

    try:
        os.nice(19)  # the web server always comes first
    except OSError:
        pass
    import torch

    torch.set_num_threads(max(1, args.threads))
    backend = get_backend(args.model)
    print(f"embedding with {args.model} on {args.threads} threads, {duty:.0%} of the time", flush=True)

    with EmbeddingCache(args.model) as cache:
        missing, urgent, refreshed = [], 0, 0.0
        urgent_done = done = 0
        t_start = time.time()
        while True:
            if not missing or time.time() - refreshed > REFRESH:
                missing, urgent = queue(cache, args.new_days)
                refreshed = time.time()
                print(f"{len(missing)} papers without an embedding, {urgent} of them tagged or new", flush=True)
            if urgent == 0 and urgent_done:
                db.request_compute()  # the new papers are ready to be recommended
                print(f"embedded {urgent_done} tagged or new papers, asked for a compute.py run", flush=True)
                urgent_done = 0
            if not missing:
                if args.once:
                    break
                time.sleep(REFRESH)
                continue
            batch, missing = missing[: args.batch], missing[args.batch :]
            t0 = time.time()
            try:
                with get_papers_db() as pdb:
                    papers = pdb.get_many(batch)
                batch = [p for p in batch if p in papers]
                if batch:
                    cache.put(batch, backend.encode([paper_text(papers[p], backend.text_format) for p in batch]))
            except Exception as e:  # e.g. the database mid-replace: try again later
                print(f"embedding failed: {e}", flush=True)
                time.sleep(60)
                continue
            n_urgent = min(len(batch), urgent)
            urgent -= n_urgent
            urgent_done += n_urgent
            done += len(batch)
            took = time.time() - t0
            if done % 500 < len(batch):
                rate = done / (time.time() - t_start) * 86400
                print(f"  {done} embedded, {len(missing)} left, about {rate:.0f} a day", flush=True)
            time.sleep(min(600.0, took * (1 / duty - 1)))


if __name__ == '__main__':
    main()
