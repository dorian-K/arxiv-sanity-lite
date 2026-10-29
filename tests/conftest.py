"""
Shared fixtures. Every script in this repo reads and writes data/ relative to the
current directory, so tests build a small synthetic corpus in a temp dir and chdir
into it instead of touching the real data/.
"""

import importlib
import os
import random
import subprocess
import sys
import time

import pytest

# no background model training: tests call serve.prefit() when they want it
os.environ.setdefault('PREFIT', '0')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# two clearly separable "fields" so ranking results are easy to check
TOPICS = {
    'vision': 'image pixel convolution segmentation detection camera depth optical'.split(),
    'language': 'token sentence grammar translation parsing lexicon dialogue syntax'.split(),
}


def make_paper(i, topic, rng):
    words = rng.choices(TOPICS[topic], k=40)
    t = time.time() - i * 3600
    return {
        '_id': f'2601.{i:05d}',
        '_idv': f'2601.{i:05d}v1',
        '_version': 1,
        '_time': t,
        '_time_str': time.strftime('%b %d %Y', time.localtime(t)),
        'title': f'{topic} paper {i} ' + ' '.join(words[:3]),
        'summary': ' '.join(words),
        'authors': [{'name': f'Author {i % 7}'}],
        'tags': [{'term': 'cs.CV' if topic == 'vision' else 'cs.CL'}],
    }


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """
    temp working dir with data/papers.sqlite holding 120 papers: every 6th is about vision, the
    rest about language. a minority topic, like a real tag, because every unlabeled paper is a
    training negative and with half the corpus on-topic a weakly regularized model just
    learns to push the unlabeled on-topic papers down
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'data').mkdir()
    from aslite.db import get_papers_db

    rng = random.Random(0)
    papers = {}
    with get_papers_db() as pdb:
        for i in range(120):
            p = make_paper(i, 'vision' if i % 6 == 0 else 'language', rng)
            papers[p['_id']] = p
        pdb.upsert(papers.values())
    return papers


@pytest.fixture
def features(corpus):
    """runs compute.py on the corpus, like the cron job would, and returns the features"""
    subprocess.run(
        [sys.executable, os.path.join(REPO, 'compute.py'), '--min_df', '1', '--max_df', '1.0'],
        check=True,
        capture_output=True,
    )
    from aslite.db import load_features

    return load_features()


@pytest.fixture
def client(features):
    import serve

    importlib.reload(serve)  # module-level state must not leak between temp dirs
    serve.app.config['TESTING'] = True
    with serve.app.test_client() as c:
        yield c


def login(client, user='alice', password='correct horse'):
    return client.post('/login', data={'username': user, 'password': password})
