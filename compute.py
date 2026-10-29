"""
Extracts tfidf features (and optionally dense embeddings) from all paper
abstracts and saves them to disk.
"""

import os
import time
import resource
import argparse
import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from aslite import spell
from aslite.db import get_papers_db, hold_heavy_lock, put_job, save_features, save_terms
from aslite.embed import embed_papers

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arxiv Computor')
    parser.add_argument('-n', '--num', type=int, default=20000, help='number of tfidf features')
    parser.add_argument('--min_df', type=int, default=5, help='min df')
    parser.add_argument('--max_df', type=float, default=0.1, help='max df')
    parser.add_argument(
        '--max_docs',
        type=int,
        default=100000,
        help='learn the vocabulary and idf from a random sample of this many papers (-1: all). '
        'counting every word pair of all 657k papers takes 5 GB of RAM, a 100k sample a fraction',
    )
    parser.add_argument(
        '--embed_model',
        type=str,
        default=os.environ.get('EMBED_MODEL', ''),
        help="also compute dense embeddings, e.g. 'st:BAAI/bge-small-en-v1.5' or 'api:text-embedding-3-small' (see aslite/embed.py). empty = tfidf only. defaults to $EMBED_MODEL",
    )
    parser.add_argument(
        '--emb_dims', type=int, default=256, help='reduce embeddings to this many dims with PCA (0 = keep all)'
    )
    parser.add_argument(
        '--embed_batch', type=int, default=32, help='papers per embedding batch; 32 suits a CPU, use 512+ on a big GPU'
    )
    parser.add_argument(
        '--embed_budget',
        type=float,
        default=float(os.environ['EMBED_BUDGET']) if os.environ.get('EMBED_BUDGET') else None,
        help='seconds to spend embedding missing papers (default: all of them; 0: none, only what '
        'embed_daemon.py cached). papers without an embedding are left out of recommendations. '
        'defaults to $EMBED_BUDGET',
    )
    args = parser.parse_args()
    print(args)
    # never at the same time as an embed_daemon.py session: each takes ~2 GB of RAM
    heavy_lock = hold_heavy_lock()
    t_start = time.time()
    phases = {}  # seconds per step, for /stats

    v = TfidfVectorizer(
        input='content',
        encoding='utf-8',
        decode_error='replace',
        strip_accents='unicode',
        lowercase=True,
        analyzer='word',
        stop_words='english',
        token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b',
        ngram_range=(1, 2),
        max_features=args.num,
        norm='l2',
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
        max_df=args.max_df,
        min_df=args.min_df,
    )

    pdb = get_papers_db()
    pids = pdb.keys()

    def make_corpus(training: bool):
        assert isinstance(training, bool)

        # determine which papers we will use to build tfidf
        if training and args.max_docs > 0 and args.max_docs < len(pids):
            # crop to a random subset of papers
            keys = list(pids)
            random.Random(0).shuffle(keys)  # the same sample every run, so the vocabulary is stable
            keep = set(keys[: args.max_docs])
        else:
            keep = None

        # yield the abstracts of the papers, streamed in the same order as pids
        for pid, d in pdb.items():
            if keep is not None and pid not in keep:
                continue
            author_str = ' '.join([a['name'] for a in d['authors']])
            yield ' '.join([d['title'], d['summary'], author_str])

    t0 = time.time()
    print("training tfidf vectors...")
    v.fit(make_corpus(training=True))

    print("running inference...")
    x = v.transform(make_corpus(training=False)).astype(np.float32)
    assert x.shape[0] == len(pids), "papers were added while computing, run compute.py again"
    print(x.shape)

    # rows oldest first, so the papers of the last N days are one slice at the end, which
    # serve.py scores without copying anything (see _model_ranking)
    times = pdb.times()
    order = np.argsort([times[p] for p in pids], kind='stable')
    pids = [pids[i] for i in order]
    x = x[order]
    phases['tfidf'] = time.time() - t0

    features = {
        'pids': pids,
        'x': x,
        'vocab': v.vocabulary_,
        'idf': v.idf_,
    }

    n_emb = 0
    if args.embed_model:
        t0 = time.time()
        print("computing embeddings...")
        x_emb, have = embed_papers(
            pids, pdb, args.embed_model, batch_size=args.embed_batch, budget=args.embed_budget, dims=args.emb_dims
        )
        if x_emb is None or have.sum() <= args.emb_dims:
            print("too few papers have embeddings yet, tfidf only for now")
        else:
            features['x_emb'] = x_emb
            features['emb_model'] = args.embed_model
            if not have.all():
                features['emb_mask'] = have  # see rank.emb_mask
                print(f"{int((~have).sum())} papers have no embedding yet: not recommended until they do")
            print(x_emb.shape)
            n_emb = int(have.sum())
        phases['embeddings'] = time.time() - t0

    t0 = time.time()
    print("saving to features to disk...")
    save_features(features)
    phases['save'] = time.time() - t0

    # the words search may correct a typo to (aslite/spell.py)
    terms = pdb.terms(spell.MIN_DOCS)
    save_terms(terms)
    print(f"{len(terms)} words for search typo correction")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB (Linux reports KB)
    put_job('compute', time.time() - t_start, papers=len(pids), embedded=n_emb, peak_mb=round(peak), phases=phases)
