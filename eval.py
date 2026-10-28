"""
Compares ranking variants (feature sets x models, see aslite/rank.py) by holding
out some liked papers and checking where they end up in the ranking.

Two sources of "interest profiles" (a set of liked papers, optionally downvoted ones):

1. your own tags (each user/tag is a profile, downvotes for that tag are its negatives):

    python eval.py run --source tags

2. the public Scholar Inbox ratings (https://github.com/avg-dev/scholar_inbox_datasets),
   774k thumbs up/down from ~14k researchers. First pick some users and fetch their papers:

    python eval.py prepare-si --ratings path/to/rated_papers.csv --users 60
    python fetch_papers.py --file data/si_ids.txt
    python compute.py --embed_model st:BAAI/bge-small-en-v1.5
    python eval.py run --source si

Metrics, averaged over profiles (each profile weighs the same), with k-fold
cross-validation over each profile's liked and downvoted papers:

    auc_rand  P(held-out liked paper scores above a random unlabeled paper).
              How well your interests are separated from everything else.
    auc_hard  P(held-out liked paper scores above a held-out downvoted paper).
              How well near-misses are rejected; only for profiles with downvotes.
    top1pct   share of held-out liked papers that land in the top 1% of all papers,
              roughly "would this have made it into the digest".
"""

import os
import csv
import json
import time
import random
import argparse
from collections import defaultdict

import pickle

import numpy as np

from aslite.db import load_features, get_tags_db, get_neg_tags_db, DATA_DIR
from aslite.rank import rank

SI_PROFILES = os.path.join(DATA_DIR, 'si_profiles.json')
SI_IDS = os.path.join(DATA_DIR, 'si_ids.txt')

# -----------------------------------------------------------------------------

def prepare_si(args):
    """ samples Scholar Inbox users and writes their profiles + the ids to fetch """
    pos, neg = defaultdict(set), defaultdict(set)
    with open(args.ratings) as f:
        for row in csv.DictReader(f):
            if '/' in row['arxiv_id']:
                continue # old-style ids (hep-th/0408218): aslite.arxiv stores them under a mangled id
            (pos if row['rating'] == '1' else neg)[row['user_id']].add(row['arxiv_id'])
    users = [u for u in pos if len(pos[u]) >= args.min_pos and len(neg[u]) >= args.min_neg]
    print(f"{len(users)} users have >= {args.min_pos} likes and >= {args.min_neg} downvotes")
    rng = random.Random(args.seed)
    users = rng.sample(sorted(users), min(args.users, len(users)))
    profiles, ids = {}, set()
    for u in users:
        p = sorted(pos[u])
        n = sorted(neg[u] - pos[u])
        # cap very active users so the corpus (and the fetch) stays manageable
        p = rng.sample(p, min(len(p), args.max_per_user))
        n = rng.sample(n, min(len(n), args.max_per_user))
        profiles['si:' + u] = {'pos': p, 'neg': n}
        ids.update(p)
        ids.update(n)
    with open(SI_PROFILES, 'w') as f:
        json.dump(profiles, f)
    with open(SI_IDS, 'w') as f:
        f.write('\n'.join(sorted(ids)) + '\n')
    print(f"wrote {len(profiles)} profiles to {SI_PROFILES} and {len(ids)} ids to {SI_IDS}")

# -----------------------------------------------------------------------------

def load_profiles(source):
    if source == 'si':
        with open(SI_PROFILES) as f:
            return {k: (set(v['pos']), set(v['neg'])) for k, v in json.load(f).items()}
    profiles = {}
    with get_tags_db() as tdb, get_neg_tags_db() as ndb:
        for user, tags in tdb.items():
            negs = ndb[user] if user in ndb else {}
            for tag, pids in tags.items():
                profiles[f"{user}/{tag}"] = (set(pids), set(negs.get(tag, set())))
    return profiles


def auc(a, b):
    """ P(x > y) for x in a, y in b, ties count half """
    a, b = np.asarray(a), np.sort(np.asarray(b))
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    lo = np.searchsorted(b, a, side='left')
    hi = np.searchsorted(b, a, side='right')
    return float(np.mean((lo + hi) / 2) / len(b))


def restrict_to(features, keep):
    """
    features limited to the papers in keep. used for --pool rated: background papers fetched
    with arxiv_daemon.py are newer than any rated paper, so models can push them down by their
    date-specific vocabulary alone, which inflates auc_rand and top1pct. in the app every
    candidate is new, so paper age carries no signal there; restricting the pool to rated papers
    (other users' likes and downvotes from the same years) removes it here too
    """
    ix = [i for i, p in enumerate(features['pids']) if p in keep]
    out = dict(features)
    out['pids'] = [features['pids'][i] for i in ix]
    out['x'] = features['x'][ix]
    if features.get('x_emb') is not None:
        out['x_emb'] = features['x_emb'][ix]
    return out


def run(args):
    if args.features:
        with open(args.features, 'rb') as f:
            features = pickle.load(f)
    else:
        features = load_features()
    profiles = load_profiles(args.source)
    if args.pool == 'rated':
        features = restrict_to(features, set().union(*(p | n for p, n in profiles.values())))
    known = set(features['pids'])
    # only keep what we have features for, and profiles with enough likes to split
    profiles = {k: (p & known, n & known) for k, (p, n) in profiles.items()}
    profiles = {k: v for k, v in profiles.items() if len(v[0]) >= max(args.min_pos, args.folds)}
    if args.max_profiles and len(profiles) > args.max_profiles:
        keys = random.Random(0).sample(sorted(profiles), args.max_profiles)
        profiles = {k: profiles[k] for k in keys}
    print(f"evaluating on {len(profiles)} profiles, {len(known)} papers, {args.folds}-fold")
    if not profiles:
        return

    variants = []
    for v in args.variants.split(','):
        parts = v.split(':') # feats:model[:C]
        variants.append((parts[0], parts[1], float(parts[2]) if len(parts) > 2 else None))

    n_papers = len(features['pids'])
    k_top = max(1, n_papers // 100)
    results = {v: defaultdict(list) for v in variants}
    times = defaultdict(float)
    rng = random.Random(1)

    for pi, (name, (pos, neg)) in enumerate(sorted(profiles.items())):
        pos_l, neg_l = sorted(pos), sorted(neg)
        rng.shuffle(pos_l)
        rng.shuffle(neg_l)
        for v in variants:
            m = defaultdict(list)
            for fold in range(args.folds):
                test_p = set(pos_l[fold::args.folds])
                test_n = set(neg_l[fold::args.folds])
                train_p, train_n = pos - test_p, neg - test_n
                t0 = time.time()
                pids, scores, _ = rank(features, train_p, train_n, feats=v[0], model=v[1], C=v[2],
                                       alpha=args.alpha, fusion_w=args.fusion_w)
                times[v] += time.time() - t0
                score = dict(zip(pids, scores))
                rank_of = {p: i for i, p in enumerate(pids)}
                sp = [score[p] for p in test_p]
                sn = [score[p] for p in test_n]
                labeled = pos | neg
                srand = [s for p, s in score.items() if p not in labeled]
                m['auc_rand'].append(auc(sp, srand))
                if sn:
                    m['auc_hard'].append(auc(sp, sn))
                m['top1pct'].append(np.mean([rank_of[p] < k_top for p in test_p]))
            for key, vals in m.items():
                results[v][key].append(float(np.nanmean(vals)))
        if (pi + 1) % 10 == 0 or pi + 1 == len(profiles):
            print(f"  {pi + 1}/{len(profiles)} profiles done")

    print()
    header = f"{'variant':<24} {'auc_rand':>9} {'auc_hard':>9} {'top1pct':>8} {'sec/fit':>8}"
    print(header)
    print('-' * len(header))
    rows = []
    n_fits = len(profiles) * args.folds
    for v in variants:
        r = results[v]
        label = f"{v[0]}:{v[1]}" + (f":{v[2]:g}" if v[2] is not None else '')
        def fmt(key):
            vals = r.get(key, [])
            if not vals:
                return float('nan'), float('nan')
            return float(np.mean(vals)), float(np.std(vals) / np.sqrt(len(vals)))
        ar, ah, tp = fmt('auc_rand'), fmt('auc_hard'), fmt('top1pct')
        print(f"{label:<24} {ar[0]:>9.4f} {ah[0]:>9.4f} {tp[0]:>8.3f} {times[v] / n_fits:>8.2f}")
        rows.append({'variant': label, 'auc_rand': ar[0], 'auc_rand_se': ar[1], 'auc_hard': ah[0],
                     'auc_hard_se': ah[1], 'top1pct': tp[0], 'top1pct_se': tp[1],
                     'sec_per_fit': times[v] / n_fits, 'profiles': len(profiles)})
    if args.out:
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}")

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate ranking variants')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('prepare-si', help='sample Scholar Inbox users into eval profiles')
    p.add_argument('--ratings', required=True, help='path to rated_papers.csv')
    p.add_argument('--users', type=int, default=60)
    p.add_argument('--min_pos', type=int, default=20)
    p.add_argument('--min_neg', type=int, default=20)
    p.add_argument('--max_per_user', type=int, default=150, help='cap likes and downvotes per user')
    p.add_argument('--seed', type=int, default=0)

    r = sub.add_parser('run', help='run the comparison')
    r.add_argument('--source', choices=['tags', 'si'], default='tags')
    r.add_argument('--variants', default='tfidf:svm,emb:svm,concat:svm,fusion:svm,tfidf:logreg,emb:logreg,fusion:logreg',
                   help='comma-separated feats:model[:C]')
    r.add_argument('--folds', type=int, default=3)
    r.add_argument('--min_pos', type=int, default=5)
    r.add_argument('--max_profiles', type=int, default=0, help='subsample profiles (0 = all)')
    r.add_argument('--alpha', type=float, default=None, help='concat: embedding block scale')
    r.add_argument('--fusion_w', type=float, default=None, help='fusion: weight of the embedding score')
    r.add_argument('--out', default='', help='also write results to this csv')
    r.add_argument('--pool', choices=['all', 'rated'], default='all',
                   help="papers to rank: everything, or only papers some profile rated "
                        "(use 'rated' with --source si, see restrict_to)")
    r.add_argument('--features', default='', help='features pickle to use instead of data/features.p, '
                   'e.g. a copy made with another --embed_model, to compare embedding models')

    args = parser.parse_args()
    {'prepare-si': prepare_si, 'run': run}[args.cmd](args)
