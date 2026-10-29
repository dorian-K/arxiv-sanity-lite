"""
Fills the embedding cache (data/embeddings.sqlite) slowly in the background, so new papers
and a big backlog (e.g. after import_snapshot.py --years 40) never keep the server busy for
long. A worker process loads the model, embeds for about --session seconds (EMBED_SESSION,
default 120) and exits, which gives all of its memory back (the model holds ~1.5 GB, and
deleting it inside a process frees only part of that); then this process, which never loads
torch, sleeps so that the work takes about --duty of the time (EMBED_DUTY, default 0.1). All
at the lowest CPU priority, on --threads cores (EMBED_THREADS, default 2). At 10% of a CPU
like a Ryzen 7 1700 that is several thousand papers a day: the ~1000 new ones, and the rest
goes to the backlog. The model is in memory about --duty of the time, and once the backlog is
done only while new papers come in.

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
import multiprocessing

from aslite import db
from aslite.db import EmbeddingCache, get_papers_db
from aslite.embed import get_backend, paper_text

REFRESH = 600  # seconds between looking for new papers and tags when nothing is missing


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


def session(model, threads, pids, seconds, batch):
    """runs in a worker process: embeds pids in order for about `seconds`, returns how many"""
    import torch

    torch.set_num_threads(max(1, threads))
    backend = get_backend(model)
    t0, done = time.time(), 0
    with EmbeddingCache(model) as cache:
        for i in range(0, len(pids), batch):
            if time.time() - t0 > seconds:
                break
            with get_papers_db() as pdb:
                papers = pdb.get_many(pids[i : i + batch])
            got = [p for p in pids[i : i + batch] if p in papers]
            if got:
                cache.put(got, backend.encode([paper_text(papers[p], backend.text_format) for p in got]))
            done += len(pids[i : i + batch])
    return done


def main():
    parser = argparse.ArgumentParser(description='embeds papers slowly in the background')
    parser.add_argument('--model', default=os.environ.get('EMBED_MODEL', ''), help='defaults to $EMBED_MODEL')
    parser.add_argument(
        '--duty',
        type=float,
        default=float(os.environ.get('EMBED_DUTY') or 0.1),
        help='share of the time spent embedding, 0-1 (default $EMBED_DUTY or 0.1)',
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=int(os.environ.get('EMBED_THREADS') or 2),
        help='CPU threads for the model (default $EMBED_THREADS or 2)',
    )
    parser.add_argument(
        '--session',
        type=float,
        default=float(os.environ.get('EMBED_SESSION') or 120),
        help='seconds a worker embeds before it exits and frees the model (default $EMBED_SESSION or 120)',
    )
    parser.add_argument('--batch', type=int, default=8, help='papers per step')
    parser.add_argument('--new_days', type=float, default=30, help='papers this new come before the backlog')
    parser.add_argument('--once', action='store_true', help='exit when nothing is missing')
    args = parser.parse_args()
    if not args.model:
        raise SystemExit('no model: set EMBED_MODEL or pass --model')
    duty = min(1.0, max(0.01, args.duty))

    try:
        os.nice(19)  # the web server always comes first; workers inherit it
    except OSError:
        pass
    print(f"embedding with {args.model} on {args.threads} threads, {duty:.0%} of the time", flush=True)
    spawn = multiprocessing.get_context('spawn')  # a fresh process without our memory

    with EmbeddingCache(args.model) as cache:
        urgent_done = 0
        while True:
            missing, urgent = queue(cache, args.new_days)
            if urgent == 0 and urgent_done:
                db.request_compute()  # the new papers are ready to be recommended
                print(f"embedded {urgent_done} tagged or new papers, asked for a compute.py run", flush=True)
                urgent_done = 0
            if not missing:
                if args.once:
                    break
                time.sleep(REFRESH)
                continue
            t0 = time.time()
            try:
                # a few papers more than a session can do, at ~1 paper/s
                todo = missing[: int(args.session * 10) + args.batch]
                with spawn.Pool(1) as pool:
                    n = pool.apply(session, (args.model, args.threads, todo, args.session, args.batch))
            except Exception as e:  # e.g. the database mid-replace: try again later
                print(f"embedding failed: {e}", flush=True)
                time.sleep(60)
                continue
            took = time.time() - t0  # loading the model counts too
            urgent_done += min(n, urgent)
            rate = n / took * duty * 86400
            print(f"  {n} embedded in {took:.0f}s, {len(missing) - n} left, about {rate:.0f} a day", flush=True)
            time.sleep(min(3600.0, took * (1 / duty - 1)))


if __name__ == '__main__':
    main()
