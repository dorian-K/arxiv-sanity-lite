"""
Checks the features for problems that make rankings look random, and compares a user's
digest across feature sets. Read-only.

    uv run python diagnose.py                 # the features
    uv run python diagnose.py -u alice        # also alice's digest per feature set
    docker compose exec arxiv_resanity_web python diagnose.py -u alice
"""

import time
import argparse

import numpy as np

from aslite import db, digest


def rows_report(name, x, ages):
    """broken rows of a feature matrix, overall and among the newest papers"""
    if hasattr(x, 'toarray'):  # sparse tfidf: empty rows
        nnz = np.diff(x.indptr)
        bad = nnz == 0
        what = 'empty'
    else:
        x = np.asarray(x, dtype=np.float32)
        finite = np.isfinite(x).all(axis=1)
        norm = np.where(finite, np.linalg.norm(np.nan_to_num(x), axis=1), 0)
        bad = ~finite | (norm < 1e-6)
        what = 'NaN/inf or zero'
        print(f"  {name}: norms min {norm[finite].min():.3f} median {np.median(norm[finite]):.3f}")
    for label, mask in (
        ('all papers', np.ones(len(ages), bool)),
        ('last 7 days', ages < 7),
        ('last 30 days', ages < 30),
    ):
        n = int(mask.sum())
        print(f"  {name}: {int(bad[mask].sum())} of {n} {what} rows ({label})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-u', '--user', default='', help='also build this user digest with each feature set')
    args = parser.parse_args()

    features = db.load_features()
    times = db.get_papers_db().times()
    pids = features['pids']
    now = time.time()
    ages = np.array([(now - times.get(p, 0)) / 86400 for p in pids])
    print(f"features: {len(pids)} papers, newest {ages.min():.1f} days old, embeddings: {features.get('emb_model')}")
    missing = len(set(times) - set(features['ptoi']))
    print(f"  papers in the database but not in the features: {missing}")
    rows_report('tfidf', features['x'], ages)
    if features.get('x_emb') is not None:
        rows_report('emb', features['x_emb'], ages)

    if args.user:
        tags = db.get_tags(args.user)
        print(f"\n{args.user}: tags {', '.join(f'{t} ({len(p)})' for t, p in tags.items())}")
        for t, pos in tags.items():
            n_in = sum(p in features['ptoi'] for p in pos)
            print(f"  {t}: {n_in} of {len(pos)} papers in the features")
        for feats in ['tfidf'] + (['emb', 'concat'] if features.get('x_emb') is not None else []):
            sections = digest.compute_for(args.user, features, times, feats=feats)
            tops = {s['tag']: [p for p, _ in s['papers']] for s in sections}
            same = len({tuple(v) for v in tops.values()}) == 1 and len(tops) > 1
            print(f"\n  digest with feats={feats}{'  <-- every tag the same!' if same else ''}")
            for t, ps in tops.items():
                print(f"    {t}: {' '.join(ps[:5])}")
