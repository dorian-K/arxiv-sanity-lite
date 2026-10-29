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
            On corpora bigger than svm_max_neg untagged papers, a random sample of them
            stands in for the rest, so training time stops growing with the corpus.
    logreg  the Scholar Inbox recipe (Flicke et al. 2025, arXiv:2504.08385): logistic
            regression on positives, downvotes and a random sample of other papers,
            with class weights set by S (total negative weight) and V (share of the
            negative weight that goes to downvotes rather than random papers).
            Its training set doesn't grow with the corpus, so it stays fast on 100k+ papers.
    lda     linear discriminant analysis in closed form: the mean of the liked papers minus a
            reference (all papers, moved a little towards the downvotes), scaled by how much
            all papers vary. no iterations, so it trains in ~5 ms instead of ~0.3 s on 465k
            papers, but ranks a little worse than logreg (README), most of all for small tags
    auto    logreg if embeddings were computed, else svm (so without embeddings nothing
            changes from the original arxiv-sanity-lite)
"""

import zlib
import random
import hashlib

import numpy as np
import scipy.sparse as sp
from sklearn import svm
from sklearn.linear_model import LogisticRegression

FEATS = ('auto', 'tfidf', 'emb', 'concat', 'fusion')
MODELS = ('auto', 'svm', 'logreg', 'lda')

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
    ('lda', 'tfidf'): 0.1,
    ('lda', 'emb'): 0.3,
    ('lda', 'concat'): 0.3,
}

DEFAULTS = dict(
    alpha=1.0,  # concat: scale of the embedding block relative to tfidf
    fusion_w=0.5,  # fusion: weight of the embedding model's score
    svm_neg_weight=5.0,  # svm: sample weight of downvoted papers (other negatives have 1)
    svm_max_neg=20000,  # svm: train on at most this many untagged papers (0 = all of them)
    n_random=5000,  # logreg: number of random negatives
    S=5.0,  # logreg: total weight of negatives relative to positives
    V=0.8,  # logreg: share of negative weight on downvotes vs random papers
    lda_neg=0.25,  # lda: weight of the downvotes' mean in the reference point (the rest: all papers)
    lda_w=0.7,  # lda with concat: weight of the embedding block (each block's score normalized first)
    lda_sample=20000,  # lda: papers the background statistics are estimated from
)


def has_embeddings(features):
    return features.get('x_emb') is not None


def usable(features, feats):
    """
    which papers a model on these feats can score: those with an embedding, while
    embed_daemon.py is still filling them in (compute.py's emb_mask); None means all. the
    others have zero rows, so they must neither be ranked nor stand in as random negatives
    """
    mask = features.get('emb_mask')
    return None if mask is None or feats == 'tfidf' else np.asarray(mask, dtype=bool)


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


def _rows(features, feats, alpha, rows):
    """the training matrix: rows of the chosen feature set (rows=None: all papers)"""
    take = (lambda m: m) if rows is None else (lambda m: m[rows])
    if feats == 'tfidf':
        return take(features['x'])
    if feats == 'emb':
        return np.asarray(take(features['x_emb']))
    if feats == 'concat':
        emb = sp.csr_matrix(np.asarray(take(features['x_emb'])) * alpha)
        return sp.hstack([take(features['x']), emb], format='csr')
    raise ValueError(feats)


def _sel(m, rows):
    """rows of a feature matrix: all (None), a slice (a view, no copy) or an index array"""
    if isinstance(rows, slice) and sp.issparse(m):
        # scipy copies a sliced csr matrix (0.17 s for a year of papers); this one shares the
        # memory-mapped arrays and only shifts the row pointers
        a, b = rows.start, rows.stop
        lo, hi = m.indptr[a], m.indptr[b]
        return sp.csr_matrix((m.data[lo:hi], m.indices[lo:hi], m.indptr[a : b + 1] - lo), shape=(b - a, m.shape[1]))
    return m if rows is None else m[rows]


def _decision(features, feats, alpha, coef, intercept, rows=None):
    """scores of the papers in rows, without materializing the concat matrix (it would be a full copy)"""
    # in the features' own float32: a float64 coef makes numpy upcast the whole matrix first,
    # which took 10x longer than the product itself (0.4 s on 465k papers)
    coef = coef.astype(np.float32)
    if feats == 'tfidf':
        return _sel(features['x'], rows) @ coef + intercept
    if feats == 'emb':
        return _sel(features['x_emb'], rows) @ coef + intercept
    n_vocab = features['x'].shape[1]
    x, emb = _sel(features['x'], rows), _sel(features['x_emb'], rows)
    return x @ coef[:n_vocab] + alpha * (emb @ coef[n_vocab:]) + intercept


def _bg_stats(features, n_sample):
    """
    lda: mean and spread of all papers, per block, from a fixed random sample. computed once
    per features (and process), so each fit only needs the means of its own papers
    """
    cache = features.setdefault('_bg', {})
    key = (id(features['x']), n_sample)  # restrict_to (eval.py) copies the dict, not this cache
    if key in cache:
        return cache[key]
    ok = usable(features, 'emb')
    pool = np.arange(len(features['pids'])) if ok is None else np.flatnonzero(ok)
    rows = np.sort(np.random.default_rng(0).choice(pool, size=min(len(pool), n_sample), replace=False))
    xs = features['x'][rows]
    mu = np.asarray(xs.mean(axis=0)).ravel()
    var = np.asarray(xs.multiply(xs).mean(axis=0)).ravel() - mu**2
    out = {'xs': xs, 't': (mu, var)}
    if has_embeddings(features):
        e = np.asarray(features['x_emb'][rows], dtype=np.float64)
        out['es'] = e.astype(np.float32)
        out['e'] = (e.mean(axis=0), np.cov(e, rowvar=False))
    cache[key] = out
    return out


def _fit_lda(features, feats, alpha, pos, neg, C, opts):
    """
    linear discriminant analysis, closed form: w = S^-1 (mean of the liked papers - reference),
    where the reference is the mean of all papers, moved towards the downvotes' mean, and S the
    spread of all papers (full covariance for the embeddings, per-word variance for tfidf), plus
    1/C times its average on the diagonal. no iterations, so a fit takes milliseconds
    """
    st = _bg_stats(features, opts['lda_sample'])
    lneg = opts['lda_neg'] if len(neg) else 0.0

    def block(m, mu, mean_of):
        ref = (1 - lneg) * mu + (lneg * mean_of(neg) if lneg else 0)
        return mean_of(pos) - ref, ref

    def mean_rows(m):
        return lambda r: np.asarray(m[r].mean(axis=0), dtype=np.float64).ravel()

    ws = {}
    if feats in ('tfidf', 'concat'):
        mu, var = st['t']
        d, ref = block(features['x'], mu, mean_rows(features['x']))
        ws['t'] = d / (var + var.mean() / C)
    if feats in ('emb', 'concat'):
        mu, cov = st['e']
        d, ref = block(features['x_emb'], mu, mean_rows(features['x_emb']))
        ws['e'] = np.linalg.solve(cov + np.trace(cov) / len(cov) / C * np.eye(len(cov)), d)
    if feats == 'tfidf':
        return ws['t'].astype(np.float32), 0.0
    if feats == 'emb':
        return ws['e'].astype(np.float32), 0.0
    # concat: each block's score scaled to unit spread over the sample, then mixed by lda_w
    w = opts['lda_w']
    sd_t = float(np.std(st['xs'] @ ws['t'].astype(np.float32))) + 1e-12
    sd_e = float(np.std(st['es'] @ ws['e'].astype(np.float32))) + 1e-12
    coef = np.concatenate([(1 - w) * ws['t'] / sd_t, w * ws['e'] / sd_e / alpha])  # _decision multiplies by alpha
    return coef.astype(np.float32), 0.0


def _fit(features, feats, alpha, pos, neg, model, C, opts, seed):
    """fits one linear model, returns (coef vector as float32, intercept)"""
    n = len(features['pids'])
    if model == 'lda':
        return _fit_lda(features, feats, alpha, pos, neg, C, opts)
    ok = usable(features, feats)

    if model == 'svm':
        y = np.zeros(n, dtype=np.float32)
        y[pos] = 1.0
        w = np.ones(n, dtype=np.float32)
        if len(neg):
            w[neg] = opts['svm_neg_weight']
        rows = None
        cap = opts['svm_max_neg']
        labeled = np.zeros(n, dtype=bool)
        labeled[pos] = True
        labeled[neg] = True
        pool = ~labeled if ok is None else ~labeled & ok  # the untagged papers that can stand in
        if cap and pool.sum() > cap:
            # big corpus: train on a random sample of the untagged papers instead of all of
            # them. scaling the weights by (papers) / sample size keeps the loss an unbiased
            # estimate of the full one, so the same C works at any corpus size
            rng = np.random.default_rng(seed)
            rand = rng.choice(np.flatnonzero(pool), size=cap, replace=False)
            rows = np.sort(np.concatenate([np.flatnonzero(labeled), rand]))
            w = w[rows] * ((labeled.sum() + pool.sum()) / len(rows))
            y = y[rows]
        elif ok is not None:
            rows = np.flatnonzero(labeled | pool)
            w, y = w[rows], y[rows]
        clf = svm.LinearSVC(class_weight='balanced', max_iter=10000, tol=1e-6, C=C, dual=True)
        clf.fit(_rows(features, feats, alpha, rows), y, sample_weight=w)

    elif model == 'logreg':
        rng = np.random.default_rng(seed)
        taken = np.zeros(n, dtype=bool)
        taken[pos] = True
        taken[neg] = True
        others = np.flatnonzero(~taken if ok is None else ~taken & ok)
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
        clf.fit(_rows(features, feats, alpha, rows), y, sample_weight=w)

    else:
        raise ValueError(f"unknown model {model!r}, options: {MODELS}")

    return clf.coef_[0].astype(np.float32), float(clf.intercept_[0])


def _top_words(features, coef_tfidf):
    ivocab = features.get('_ivocab')
    if ivocab is None:
        ivocab = features['_ivocab'] = {v: k for k, v in features['vocab'].items()}
    sortix = np.argsort(-coef_tfidf)
    return [{'word': ivocab[ix], 'weight': float(coef_tfidf[ix])} for ix in list(sortix[:40]) + list(sortix[-20:])]


SKETCH_TOP = 5000  # the scores of this many best papers are kept exactly, for "top 0.01%"
SKETCH_QUANTILES = 1001  # and this many quantiles of all scores for the rest


def fingerprint(pos_pids, neg_pids=()):
    """names a training set, the same in every process (unlike hash()): the key of a cached model"""
    pos = sorted(set(pos_pids))
    neg = sorted(set(neg_pids) - set(pos))
    return hashlib.sha1(('\n'.join(pos) + '\n--\n' + '\n'.join(neg)).encode()).hexdigest()


def fit(features, pos_pids, neg_pids=(), feats='auto', model='auto', C=None, sketch=True, **kwargs):
    """
    Trains the model for these positives (and downvotes) and returns (m, scores): m is a
    plain dict that apply() scores any papers with, so it can be cached and stored, and
    scores are those of all papers in the order of features['pids']. (None, None) without
    positives. m holds the tfidf words with the most positive / negative weight ('words',
    empty if no tfidf model was trained) and, with sketch, what pct() needs for "top x%"
    """
    opts = dict(DEFAULTS)
    opts.update({k: v for k, v in kwargs.items() if v is not None})
    feats = resolve_feats(features, feats)
    model = resolve_model(features, model)

    ptoi = features.get('ptoi') or {p: i for i, p in enumerate(features['pids'])}
    pos = np.array(sorted({ptoi[p] for p in pos_pids if p in ptoi}), dtype=np.int64)
    neg = np.array(sorted({ptoi[p] for p in neg_pids if p in ptoi} - set(pos.tolist())), dtype=np.int64)
    if len(pos) == 0:
        return None, None

    # deterministic random negatives for a given set of positives, so reloading
    # the page does not reshuffle the ranking
    seed = zlib.crc32(pos.tobytes())

    def C_for(f):
        return C if C is not None else DEFAULT_C[(model, f)]

    # the score is sum(w * (raw - mu) / sd) over the parts: one part scaled by w, or for
    # fusion each model's score z-normalized over all papers and mixed
    if feats == 'fusion':
        w = opts['fusion_w']
        parts = []
        for f, wf in (('emb', w), ('tfidf', 1 - w)):
            coef, b = _fit(features, f, 1.0, pos, neg, model, C_for(f), opts, seed)
            raw = _decision(features, f, 1.0, coef, np.float64(b))
            if usable(features, 'emb') is not None:
                raw = raw[usable(features, 'emb')]  # z-scores over the papers that get ranked
            parts.append(
                {'feats': f, 'coef': coef, 'b': b, 'w': wf, 'mu': float(raw.mean()), 'sd': float(raw.std() + 1e-12)}
            )
        words = _top_words(features, parts[1]['coef'])
    else:
        coef, b = _fit(features, feats, opts['alpha'], pos, neg, model, C_for(feats), opts, seed)
        # svm: keep the original score scale
        parts = [{'feats': feats, 'coef': coef, 'b': b, 'w': 100.0 if model == 'svm' else 1.0, 'mu': 0.0, 'sd': 1.0}]
        n_vocab = features['x'].shape[1]
        words = _top_words(features, coef[:n_vocab]) if feats in ('tfidf', 'concat') else []
    m = {'feats': feats, 'model': model, 'alpha': opts['alpha'], 'parts': parts, 'words': words}
    s = apply(features, m)
    if sketch:
        ok = usable(features, feats)
        ranked = s if ok is None else s[ok]  # "top x%" among the papers that get ranked
        k = min(SKETCH_TOP, len(ranked))
        top = ranked[np.argpartition(-ranked, k - 1)[:k]]
        m['top'] = np.sort(top)[::-1].copy()  # best first
        m['quantiles'] = np.quantile(ranked, np.linspace(0, 1, SKETCH_QUANTILES))  # ascending
        m['n'] = len(ranked)
    return m, s


def apply(features, m, rows=None):
    """the scores of model m (see fit) for the papers in rows: all (None), a slice or an index array"""
    s = None
    for p in m['parts']:
        # the intercept as a float64 scalar, which makes the scores float64 as they always were
        raw = _decision(features, p['feats'], m['alpha'], p['coef'], np.float64(p['b']), rows)
        v = p['w'] * ((raw - p['mu']) / p['sd'])
        s = v if s is None else s + v
    return s


def pct(m, scores):
    """
    where each of these scores lands among all papers, as a fraction (0.003 = top 0.3%), from
    m's sketch: exact among the best SKETCH_TOP papers, interpolated from quantiles below them
    """
    scores = np.asarray(scores, dtype=np.float64)
    top, n = m['top'], m['n']
    higher = np.searchsorted(-top, -scores, side='left').astype(np.float64)  # best papers above each
    below = scores < top[-1]
    if below.any():
        frac = np.interp(scores[below], m['quantiles'], np.linspace(0, 1, len(m['quantiles'])))
        higher[below] = np.maximum(len(top), np.round((1 - frac) * n))
    return ((higher + 1) / n).tolist()


def score(features, pos_pids, neg_pids=(), feats='auto', model='auto', C=None, **kwargs):
    """
    Returns (scores, words): the score of every paper, as an array in the order of
    features['pids'] (None without positives), and the tfidf words with the most
    positive / negative weight (empty if no tfidf model was trained). Cheaper than rank()
    when only some papers are wanted: sorting 465k pids into a list takes longer than the fit
    """
    m, s = fit(features, pos_pids, neg_pids, feats, model, C, sketch=False, **kwargs)
    return (None, []) if m is None else (s, m['words'])


def rank(features, pos_pids, neg_pids=(), feats='auto', model='auto', C=None, **kwargs):
    """
    Returns (pids, scores, words): all papers sorted by score, and the tfidf words
    with the most positive / negative weight (empty if no tfidf model was trained).
    """
    s, words = score(features, pos_pids, neg_pids, feats, model, C, **kwargs)
    if s is None:
        return [], [], []
    pids = features['pids']
    sortix = np.argsort(-s)
    return [pids[ix] for ix in sortix], s[sortix].tolist(), words


def holdout_ranks(features, pos_pids, neg_pids=(), folds=5, **kwargs):
    """
    how well each liked paper fits its tag: {pid: rank fraction}, where 0.004 means the
    paper lands in the top 0.4% of all papers when ranked by a model trained on the
    tag's *other* papers (k-fold, so `folds` fits instead of one per paper). papers the
    rest of the tag doesn't predict are mis-tagged or a side topic that a tag of its own
    would serve better. needs at least `folds` papers
    """
    pos = sorted({p for p in pos_pids if p in features['ptoi']})
    if len(pos) < max(folds, 2):
        return {}
    order = random.Random(zlib.crc32(','.join(pos).encode())).sample(pos, len(pos))
    ptoi, n, out = features['ptoi'], len(features['pids']), {}
    for k in range(folds):
        test = set(order[k::folds])
        s, _ = score(features, [p for p in pos if p not in test], neg_pids, **kwargs)
        for p in test:
            out[p] = (int((s > s[ptoi[p]]).sum()) + 1) / n  # its place in the ranking, without sorting
    return out
