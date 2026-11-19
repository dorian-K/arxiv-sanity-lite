import os
import numpy as np
import pytest

from aslite.rank import rank, resolve_feats, resolve_model, FEATS, MODELS


def with_fake_embeddings(features):
    """adds an x_emb block that separates the two topics, as a real embedding would"""
    f = dict(features)
    rng = np.random.default_rng(0)
    is_vision = np.array([int(p.split('.')[1]) % 6 == 0 for p in f['pids']])
    emb = rng.normal(size=(len(f['pids']), 16)).astype(np.float32)
    emb[:, 0] += np.where(is_vision, 3.0, -3.0)
    f['x_emb'] = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return f


def vision_pids(features):
    return [p for p in features['pids'] if int(p.split('.')[1]) % 6 == 0]


def test_auto_is_the_original_without_embeddings(features):
    # without embeddings, auto must stay plain tfidf + svm, as in the original arxiv-sanity-lite
    assert resolve_feats(features, 'auto') == 'tfidf'
    assert resolve_model(features, 'auto') == 'svm'
    with pytest.raises(ValueError):
        resolve_feats(features, 'emb')


def test_auto_with_embeddings(features):
    f = with_fake_embeddings(features)
    assert resolve_feats(f, 'auto') == 'concat'
    assert resolve_model(f, 'auto') == 'logreg'


@pytest.mark.parametrize('model', [m for m in MODELS if m != 'auto'])
@pytest.mark.parametrize('feats', [f for f in FEATS if f != 'auto'])
def test_liked_topic_ranks_first(features, feats, model):
    f = with_fake_embeddings(features)
    vision = vision_pids(f)
    pids, scores, words = rank(f, pos_pids=vision[:5], feats=feats, model=model)

    assert len(pids) == len(f['pids'])
    assert scores == sorted(scores, reverse=True)
    # unlabeled vision papers should rank above the language papers
    pos = set(vision[:5])
    rank_of = {p: i for i, p in enumerate(pids)}
    v = [rank_of[p] for p in vision if p not in pos]
    l = [rank_of[p] for p in pids if p not in vision]
    auc = np.mean([[a < b for b in l] for a in v])
    assert auc >= 0.9, auc
    if feats == 'emb':
        assert words == []
    else:
        assert words


def test_downvotes_push_papers_down(features):
    vision = vision_pids(features)
    pos, neg = vision[:5], vision[5:8]
    base, _, _ = rank(features, pos)
    down, _, _ = rank(features, pos, neg_pids=neg)
    for p in neg:
        assert down.index(p) > base.index(p)


def test_no_positives_returns_empty(features):
    assert rank(features, pos_pids=[]) == ([], [], [])


def test_svm_with_sampled_negatives(features):
    # a cap below the corpus size makes the svm train on a sample of the untagged papers;
    # the ranking still has to work, and be the same on every reload
    vision = vision_pids(features)
    pos = vision[:5]
    pids, scores, _ = rank(features, pos, model='svm', svm_max_neg=30)
    again, _, _ = rank(features, pos, model='svm', svm_max_neg=30)
    assert pids == again
    rank_of = {p: i for i, p in enumerate(pids)}
    v = [rank_of[p] for p in vision if p not in pos]
    l = [rank_of[p] for p in pids if p not in vision]
    assert np.mean([[a < b for b in l] for a in v]) >= 0.9


def test_compute_reuses_the_rows_of_unchanged_papers(corpus, features):
    """a second run learns no new vocabulary and makes rows only for new or updated papers, the same ones a full run would"""
    import random
    import subprocess
    import sys
    import time

    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    from aslite.db import get_papers_db, load_features
    from conftest import REPO, make_paper

    with get_papers_db() as pdb:
        new = make_paper(500, 'vision', random.Random(1))
        new['_time'] = time.time() + 60  # the newest
        changed = dict(corpus[sorted(corpus)[3]], _version=9, summary='a new version about quantum chromodynamics')
        changed['_time'], changed['_idv'] = time.time() - 10, changed['_id'] + 'v9'
        pdb.upsert([new, changed])
    run = lambda *a: (
        subprocess.run(
            [sys.executable, os.path.join(REPO, 'compute.py'), '--min_df', '1', '--max_df', '1.0', *a],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    out = run()
    assert 'tfidf rows of 2 papers' in out and 'learning' not in out
    f = load_features()
    assert f['tfidf_fit_time'] == features['tfidf_fit_time'] and f['pids'][-1] == new['_id']
    params = {k: v for k, v in f['tfidf_params'].items() if k != 'max_docs'} | {'dtype': np.float32}
    v = TfidfVectorizer(**params, vocabulary=f['vocab'])
    v.idf_ = f['idf']
    with get_papers_db() as pdb:
        got = pdb.get_many(f['pids'])
    texts = [
        ' '.join([got[p]['title'], got[p]['summary'], ' '.join(a['name'] for a in got[p]['authors'])])
        for p in f['pids']
    ]
    assert abs(v.transform(texts) - f['x']).max() == 0
    assert 'learning the tfidf vocabulary' in run('--refit')
