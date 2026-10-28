"""
Extracts tfidf features (and optionally dense embeddings) from all paper
abstracts and saves them to disk.
"""

import os
import argparse
from random import shuffle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from aslite.db import get_papers_db, save_features
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
        default=-1,
        help='maximum number of documents to use when training tfidf, or -1 to disable',
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
    args = parser.parse_args()
    print(args)

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

    pdb = get_papers_db(flag='r')

    def make_corpus(training: bool):
        assert isinstance(training, bool)

        # determine which papers we will use to build tfidf
        if training and args.max_docs > 0 and args.max_docs < len(pdb):
            # crop to a random subset of papers
            keys = list(pdb.keys())
            shuffle(keys)
            keys = keys[: args.max_docs]
        else:
            keys = pdb.keys()

        # yield the abstracts of the papers
        for p in keys:
            d = pdb[p]
            author_str = ' '.join([a['name'] for a in d['authors']])
            yield ' '.join([d['title'], d['summary'], author_str])

    print("training tfidf vectors...")
    v.fit(make_corpus(training=True))

    print("running inference...")
    x = v.transform(make_corpus(training=False)).astype(np.float32)
    print(x.shape)

    pids = list(pdb.keys())
    features = {
        'pids': pids,
        'x': x,
        'vocab': v.vocabulary_,
        'idf': v.idf_,
    }

    if args.embed_model:
        print("computing embeddings...")
        x_emb = embed_papers(pids, pdb, args.embed_model)
        if 0 < args.emb_dims < x_emb.shape[1]:
            # Scholar Inbox found 256 PCA dims lose ~nothing for recommendation,
            # while making the per-request classifier faster and the file smaller
            pca = PCA(n_components=args.emb_dims, random_state=0)
            x_emb = pca.fit_transform(x_emb).astype(np.float32)
            x_emb /= np.linalg.norm(x_emb, axis=1, keepdims=True) + 1e-12
            print(f"PCA to {args.emb_dims} dims keeps {pca.explained_variance_ratio_.sum():.1%} of variance")
        features['x_emb'] = x_emb
        features['emb_model'] = args.embed_model
        print(x_emb.shape)

    print("saving to features to disk...")
    save_features(features)
