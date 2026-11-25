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
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from aslite import spell
from aslite.db import get_papers_db, get_refs_db, hold_heavy_lock, load_features, put_job, save_features, save_terms
from aslite.refs import merge_twins
from aslite.embed import embed_papers

REFIT_DAYS = 30  # the tfidf vocabulary and idf are learned again after this many days
REFIT_GROWTH = 0.1  # ... or once there are this much more papers than when they were learned

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
        '--refit',
        action='store_true',
        help=f'learn the tfidf vocabulary and idf again now (else every {REFIT_DAYS} days, '
        'reusing the rows of unchanged papers in between)',
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

    params = dict(
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
        dtype=np.float32,
    )
    fit_params = {**params, 'max_docs': args.max_docs, 'dtype': 'float32'}

    pdb = get_papers_db()
    with get_refs_db() as rdb:  # a paper users added from another site may have reached arXiv since
        for old, new in merge_twins(pdb, rdb):
            print(f"{old} is on arXiv too, as {new}: merged into it")
    times = pdb.times()
    # rows oldest first, so the papers of the last N days are one slice at the end, which
    # serve.py scores without copying anything (see _model_ranking). the ids from the same read as
    # the times: a paper imported in between was in keys() but not in times (a KeyError)
    pids = sorted(times, key=lambda p: (times[p], p))

    def text(d):
        return ' '.join([d['title'], d['summary'], ' '.join(a['name'] for a in d['authors'])])

    def sample():
        """the papers the vocabulary and idf are learned from: the same random sample every run"""
        if 0 < args.max_docs < len(pids):
            keys = sorted(pids)
            random.Random(0).shuffle(keys)
            keep = set(keys[: args.max_docs])
            for pid, d in pdb.items():
                if pid in keep:
                    yield text(d)
        else:
            for _, d in pdb.items():
                yield text(d)

    def stream(want):
        """the texts of these papers, in this order (read in chunks, so memory stays small)"""
        for i in range(0, len(want), 5000):
            got = pdb.get_many(want[i : i + 5000])
            yield from (text(got[p]) for p in want[i : i + 5000])

    t0 = time.time()
    # the vocabulary and idf are learned again only now and then (--refit, every REFIT_DAYS, or once the
    # corpus grew by REFIT_GROWTH): in between, a paper's row depends on its text alone, so the rows of
    # the papers that didn't change are taken over from the last features and only new ones are made
    # (learning them takes ~1 min, making every row ~4: 721k papers, one core)
    prev = None
    if not args.refit:
        try:
            prev = load_features()
        except Exception as e:  # none yet, or unreadable: everything from scratch
            print(f"no earlier features to build on ({type(e).__name__})")
        if prev is not None:
            why = (
                'the settings changed'
                if prev.get('tfidf_params') != fit_params
                else 'they are older than the row times'
                if prev.get('x_time') is None
                else f'the vocabulary is over {REFIT_DAYS} days old'
                if time.time() - prev.get('tfidf_fit_time', 0) > REFIT_DAYS * 86400
                else 'the corpus grew a lot'
                if len(pids) > prev.get('tfidf_fit_n', 0) * (1 + REFIT_GROWTH)
                else ''
            )
            if why:
                print(f"learning the vocabulary again: {why}")
                prev = None
    if prev is None:
        print("learning the tfidf vocabulary and idf...")
        v = TfidfVectorizer(**params).fit(sample())
        fit_time, fit_n = time.time(), len(pids)
        todo = pids
    else:
        v = TfidfVectorizer(**{**params, 'vocabulary': prev['vocab']})
        v.idf_ = prev['idf']
        fit_time, fit_n = prev['tfidf_fit_time'], prev['tfidf_fit_n']
        ptoi, ptime = prev['ptoi'], prev['x_time']
        todo = [p for p in pids if p not in ptoi or ptime[ptoi[p]] != times[p]]  # new, or a new version
    print(f"tfidf rows of {len(todo)} papers (of {len(pids)})...")
    new_x = v.transform(stream(todo)) if todo else sp.csr_matrix((0, len(v.vocabulary_)), dtype=np.float32)
    if prev is None:
        x = new_x
    else:
        # the rows taken over and the new ones, put together in pids' order (one copy of the
        # matrix besides the result: the old one is memory-mapped)
        n_old, new_row = prev['x'].shape[0], {p: i for i, p in enumerate(todo)}
        sel = np.array([n_old + new_row[p] if p in new_row else ptoi[p] for p in pids], dtype=np.int64)
        both = sp.vstack([prev['x'], new_x], format='csr')
        x = both[sel]
        del both
    x = x.astype(np.float32, copy=False)
    print(x.shape)
    phases['tfidf'] = time.time() - t0

    features = {
        'pids': pids,
        'x': x,
        'x_time': np.array([times[p] for p in pids], dtype=np.float64),  # which version a row was made from
        'vocab': v.vocabulary_,
        'idf': np.asarray(v.idf_),
        'tfidf_params': fit_params,
        'tfidf_fit_time': fit_time,
        'tfidf_fit_n': fit_n,
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
