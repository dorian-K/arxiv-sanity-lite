"""papers without an embedding yet (embed_daemon.py fills them in slowly): masked, never ranked"""

import os
import subprocess
import sys

import numpy as np
import pytest

from aslite import db, rank
from test_rank import vision_pids, with_fake_embeddings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def with_backlog(features, every=3):
    """fake embeddings, but every `every`th paper has none yet (a zero row, masked out)"""
    f = with_fake_embeddings(features)
    mask = np.arange(len(f['pids'])) % every != 0
    f['x_emb'] = f['x_emb'] * mask[:, None]
    f['emb_mask'] = mask
    f['ptoi'] = features['ptoi']
    return f


@pytest.mark.parametrize('model', ['svm', 'logreg', 'lda'])
def test_papers_without_embeddings_are_not_negatives_or_ranked(features, model, monkeypatch):
    f = with_backlog(features)
    used = []
    real = rank._rows

    def rows(features, feats, alpha, r):
        used.append(r)
        return real(features, feats, alpha, r)

    monkeypatch.setattr(rank, '_rows', rows)
    m, s = rank.fit(f, vision_pids(f)[:6], feats='concat', model=model)
    for r in used:  # what the model trained on: no paper without an embedding
        assert r is not None and f['emb_mask'][r].all()
    assert m['n'] == int(f['emb_mask'].sum())  # "top x%" among the papers that get ranked
    assert rank.usable(f, 'tfidf') is None  # tfidf alone can rank every paper


def test_cache_partial_matrix_and_merge(corpus, tmp_path):
    with db.EmbeddingCache('m') as c:
        c.put(['a', 'b'], np.eye(2, dtype=np.float32))
        x, have = c.matrix(['a', 'x', 'b'], partial=True)
        assert have.tolist() == [True, False, True] and not x[1].any()
        with pytest.raises(KeyError):
            c.matrix(['a', 'x'])
    other = str(tmp_path / 'gpu.sqlite')
    with db.EmbeddingCache('m', path=other) as c:
        c.put(['a', 'x'], np.full((2, 2), 7, dtype=np.float32))
    with db.EmbeddingCache('m') as c:
        assert c.merge(other) == 1  # only x: a was computed here meanwhile and is kept
        x, have = c.matrix(['a', 'x'], partial=True)
        assert have.all() and x[0].tolist() == [1, 0] and x[1].tolist() == [7, 7]


def test_compute_uses_only_cached_embeddings_with_budget_0(corpus):
    pids = sorted(corpus)
    rng = np.random.default_rng(0)
    with db.EmbeddingCache('api:fake') as c:  # a backend that can't run here: budget 0 never loads it
        c.put(pids[:40], rng.normal(size=(40, 8)).astype(np.float32))
    r = subprocess.run(
        [
            sys.executable,
            os.path.join(REPO, 'compute.py'),
            '--min_df',
            '1',
            '--max_df',
            '1.0',
            '--embed_model',
            'api:fake',
            '--embed_budget',
            '0',
            '--emb_dims',
            '4',
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    f = db.load_features()
    mask = np.asarray(f['emb_mask'])
    assert mask.sum() == 40 and {f['pids'][i] for i in np.flatnonzero(mask)} == set(pids[:40])
    assert not np.asarray(f['x_emb'])[~mask].any()


def test_backlog_order_tagged_then_new_then_old(corpus):
    import embed_daemon

    pids = sorted(corpus, key=lambda p: corpus[p]['_time'])  # oldest first
    db.tag_paper('alice', pids[0], 't')  # the oldest, but tagged
    now = corpus[pids[-1]]['_time'] + 60
    with db.EmbeddingCache('m') as c:
        c.put([pids[-1]], np.ones((1, 2), dtype=np.float32))  # the newest is done
        missing, urgent = embed_daemon.queue(c, new_days=1, now=now)
    assert missing[0] == pids[0] and pids[-1] not in missing
    assert missing[1] == pids[-2]  # then newest first
    new = [p for p in pids[1:-1] if corpus[p]['_time'] > now - 86400]
    assert urgent == 1 + len(new) and missing[1 : 1 + len(new)] == new[::-1]
