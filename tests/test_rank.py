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
