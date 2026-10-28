"""
Ranks all papers for a set of positive (tagged) and optional negative
(downvoted) papers by training a small linear model on the spot.

Used by both serve.py and send_emails.py, and by eval.py to compare variants.

Feature sets ("feats"):
    tfidf   the original sparse tfidf vectors
    emb     dense embeddings from compute.py --embed_model
    concat  tfidf and embeddings side by side in one model; alpha scales the embedding block
    fusion  one model per feature set, scores z-normalized and mixed: w*emb + (1-w)*tfidf
    auto    concat if embeddings were computed, else tfidf (best in eval.py, see README)

Models ("model"):
    svm     the original arxiv-sanity-lite LinearSVC: every untagged paper is a negative,
            classes balanced. Downvoted papers count as negatives with extra weight.
    logreg  the Scholar Inbox recipe (Flicke et al. 2025, arXiv:2504.08385): logistic
            regression on positives, downvotes and a random sample of other papers,
            with class weights set by S (total negative weight) and V (share of the
            negative weight that goes to downvotes rather than random papers).
            Its training set doesn't grow with the corpus, so it stays fast on 100k+ papers.
    auto    logreg if embeddings were computed, else svm (so without embeddings nothing
            changes from the original arxiv-sanity-lite)
"""

import zlib

import numpy as np
import scipy.sparse as sp
from sklearn import svm
from sklearn.linear_model import LogisticRegression

FEATS = ('auto', 'tfidf', 'emb', 'concat', 'fusion')
MODELS = ('auto', 'svm', 'logreg')

# default regularization per (model, feature set); C is "inverse regularization"
# so lower = smoother. the tfidf svm value is the original arxiv-sanity-lite one; the
# others were swept with eval.py on 60 Scholar Inbox users (--pool rated, bge-small),
# also checked with only 5 training likes (--train_pos 5). logreg's weights normalize
# the loss mass, which is why its best C is so much larger than svm's.
DEFAULT_C = {
    ('svm', 'tfidf'): 0.01,
    ('svm', 'emb'): 0.03,
    ('svm', 'concat'): 0.01,
    ('logreg', 'tfidf'): 100.0,
    ('logreg', 'emb'): 100.0,
    ('logreg', 'concat'): 100.0,
}

DEFAULTS = dict(
    alpha=1.0,  # concat: scale of the embedding block relative to tfidf
    fusion_w=0.5,  # fusion: weight of the embedding model's score
    svm_neg_weight=5.0,  # svm: sample weight of downvoted papers (other negatives have 1)
    n_random=5000,  # logreg: number of random negatives
    S=5.0,  # logreg: total weight of negatives relative to positives
    V=0.8,  # logreg: share of negative weight on downvotes vs random papers
)

_concat_cache = {}


def has_embeddings(features):
    return features.get('x_emb') is not None


def resolve_feats(features, feats):
    if feats in (None, '', 'auto'):
        return 'concat' if has_embeddings(features) else 'tfidf'
    if feats not in FEATS:
        raise ValueError(f"unknown feats {feats!r}, options: {FEATS}")
    if feats != 'tfidf' and not has_embeddings(features):
        raise ValueError(f"feats={feats!r} needs embeddings: run compute.py with --embed_model")
    return feats


def resolve_model(features, model):
    if model in (None, '', 'auto'):
        return 'logreg' if has_embeddings(features) else 'svm'
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}, options: {MODELS}")
    return model


def _matrix(features, feats, alpha):
    if feats == 'tfidf':
        return features['x']
    if feats == 'emb':
        return features['x_emb']
    if feats == 'concat':
        key = (id(features), alpha)
        if key not in _concat_cache:
            _concat_cache.clear()  # only keep the current features file around
            emb = sp.csr_matrix(features['x_emb'] * alpha)
            _concat_cache[key] = sp.hstack([features['x'], emb], format='csr')
        return _concat_cache[key]
    raise ValueError(feats)


def _fit_score(x, pos, neg, model, C, opts, seed):
    """fits one linear model on rows of x, returns (scores for all rows, coef vector)"""
    n = x.shape[0]

    if model == 'svm':
        y = np.zeros(n, dtype=np.float32)
        y[pos] = 1.0
        w = np.ones(n, dtype=np.float32)
        if len(neg):
            w[neg] = opts['svm_neg_weight']
        clf = svm.LinearSVC(class_weight='balanced', max_iter=10000, tol=1e-6, C=C, dual=True)
        clf.fit(x, y, sample_weight=w)
        return clf.decision_function(x), clf.coef_[0]

    if model == 'logreg':
        rng = np.random.default_rng(seed)
        taken = np.zeros(n, dtype=bool)
        taken[pos] = True
        taken[neg] = True
        others = np.flatnonzero(~taken)
        rand = rng.choice(others, size=min(opts['n_random'], len(others)), replace=False)

        nP, nN, nR = len(pos), len(neg), len(rand)
        S, V = opts['S'], opts['V']
        if nN == 0:
            V = 0.0  # nothing to put the downvote share on
        denom = V * nN + (1 - V) * nR
        wP, wN, wR = 1.0 / nP, S * V / denom, S * (1 - V) / denom

        rows = np.concatenate([pos, neg, rand])
        y = np.concatenate([np.ones(nP), np.zeros(nN), np.zeros(nR)])
        w = np.concatenate([np.full(nP, wP), np.full(nN, wN), np.full(nR, wR)])
        # sklearn sums the weighted losses, so with these weights the total loss mass is
        # S+1 regardless of how many papers someone rated, which keeps C comparable
        # across users (the same point the paper makes about its nT correction)
        clf = LogisticRegression(C=C, solver='liblinear', max_iter=1000)
        clf.fit(x[rows], y, sample_weight=w)
        return clf.decision_function(x), clf.coef_[0]

    raise ValueError(f"unknown model {model!r}, options: {MODELS}")


def _zscore(s):
    return (s - s.mean()) / (s.std() + 1e-12)


def _top_words(features, coef_tfidf):
    ivocab = {v: k for k, v in features['vocab'].items()}
    sortix = np.argsort(-coef_tfidf)
    return [{'word': ivocab[ix], 'weight': float(coef_tfidf[ix])} for ix in list(sortix[:40]) + list(sortix[-20:])]


def rank(features, pos_pids, neg_pids=(), feats='auto', model='auto', C=None, **kwargs):
    """
    Returns (pids, scores, words): all papers sorted by score, and the tfidf words
    with the most positive / negative weight (empty if no tfidf model was trained).
    """
    opts = dict(DEFAULTS)
    opts.update({k: v for k, v in kwargs.items() if v is not None})
    feats = resolve_feats(features, feats)
    model = resolve_model(features, model)

    pids = features['pids']
    ptoi = {p: i for i, p in enumerate(pids)}
    pos = np.array(sorted({ptoi[p] for p in pos_pids if p in ptoi}), dtype=np.int64)
    neg = np.array(sorted({ptoi[p] for p in neg_pids if p in ptoi} - set(pos.tolist())), dtype=np.int64)
    if len(pos) == 0:
        return [], [], []

    # deterministic random negatives for a given set of positives, so reloading
    # the page does not reshuffle the ranking
    seed = zlib.crc32(pos.tobytes())

    def C_for(f):
        return C if C is not None else DEFAULT_C[(model, f)]

    if feats == 'fusion':
        s_t, coef_t = _fit_score(features['x'], pos, neg, model, C_for('tfidf'), opts, seed)
        s_e, _ = _fit_score(features['x_emb'], pos, neg, model, C_for('emb'), opts, seed)
        w = opts['fusion_w']
        s = w * _zscore(s_e) + (1 - w) * _zscore(s_t)
        words = _top_words(features, coef_t)
    else:
        x = _matrix(features, feats, opts['alpha'])
        s, coef = _fit_score(x, pos, neg, model, C_for(feats), opts, seed)
        s = 100 * s if model == 'svm' else s  # keep the original svm score scale
        n_vocab = features['x'].shape[1]
        words = _top_words(features, coef[:n_vocab]) if feats in ('tfidf', 'concat') else []

    sortix = np.argsort(-s)
    return [pids[ix] for ix in sortix], [float(s[ix]) for ix in sortix], words
