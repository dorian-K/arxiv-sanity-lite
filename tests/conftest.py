"""
Shared fixtures. Every script in this repo reads and writes data/ relative to the
current directory, so tests build a small synthetic corpus in a temp dir and chdir
into it instead of touching the real data/.
"""

import os
import random
import subprocess
import sys
import time

import pytest

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
    """ temp working dir with data/papers.db holding 60 papers, 30 per topic """
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'data').mkdir()
    from aslite.db import get_papers_db, get_metas_db
    rng = random.Random(0)
    papers = {}
    with get_papers_db(flag='c') as pdb, get_metas_db(flag='c') as mdb:
        for i in range(60):
            p = make_paper(i, 'vision' if i % 2 == 0 else 'language', rng)
            pdb[p['_id']] = p
            mdb[p['_id']] = {'_time': p['_time']}
            papers[p['_id']] = p
    return papers


@pytest.fixture
def features(corpus):
    """ runs compute.py on the corpus, like the cron job would, and returns data/features.p """
    subprocess.run([sys.executable, os.path.join(REPO, 'compute.py'), '--min_df', '1', '--max_df', '1.0'],
                   check=True, capture_output=True)
    from aslite.db import load_features
    return load_features()
