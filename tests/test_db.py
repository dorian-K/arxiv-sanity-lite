import mmap
import os

from aslite import db


def test_upsert_keeps_the_newest_version(corpus):
    pid = next(iter(corpus))
    p = dict(corpus[pid])
    with db.get_papers_db() as pdb:
        assert len(pdb) == len(corpus)
        assert pdb[pid]['title'] == p['title']
        # an older version is ignored, a newer one replaces it, and search follows along
        assert pdb.upsert([dict(p, title='stale', _time=p['_time'] - 10)]) == 0
        assert pdb.upsert([dict(p, title='zebra crossing', _version=2, _time=p['_time'] + 10)]) == 1
        assert pdb[pid]['_idv'] == pid + 'v2'
        assert pdb.search('zebra')[0] == [pid]
        assert pid not in pdb.search('stale')[0]
        assert pdb.get_many([pid, 'nope']).keys() == {pid}
        assert pdb.times()[pid] == p['_time'] + 10


def test_tags_are_exclusive(corpus):
    pid = next(iter(corpus))
    db.tag_paper('alice', pid, 'cv')
    db.tag_paper('alice', pid, 'cv', neg=True)  # downvoting replaces the like
    assert db.get_tags('alice') == {}
    assert db.get_tags('alice', neg=True) == {'cv': {pid}}
    assert not db.untag_paper('alice', pid, 'cv')
    assert db.untag_paper('alice', pid, 'cv', neg=True)
    db.tag_paper('alice', pid, 'cv')
    db.tag_paper('bob', pid, 'nlp')
    assert db.get_all_tags() == {'alice': {'cv': {pid}}, 'bob': {'nlp': {pid}}}
    assert db.delete_tag('alice', 'cv') and not db.delete_tag('alice', 'cv')


def is_mapped(a):
    while a is not None and not isinstance(a, mmap.mmap):
        a = getattr(a, 'base', None)
    return a is not None


def test_features_are_memory_mapped_and_versioned(features):
    # mapped, not copied: web workers then share one copy through the page cache
    assert is_mapped(features['x'].data) and is_mapped(features['x'].indices)
    first = os.path.realpath(os.path.join(db.FEATURES_DIR, 'current'))
    db.save_features({k: v for k, v in features.items() if k != 'ptoi'})
    again = db.load_features()
    assert again is not features  # a new version gets picked up
    assert (again['x'] != features['x']).nnz == 0
    assert not os.path.exists(first)  # old versions are cleaned up


def test_backup_users(corpus):
    import sqlite3
    import time

    db.tag_paper('alice', next(iter(corpus)), 'cv')
    db.log_fetch('alice', 'old')
    db._run('UPDATE fetches SET time = ?', (time.time() - 3 * 86400,))
    db.log_fetch('alice', 'new')
    path = db.backup_users()
    assert path.endswith('backups/users.sqlite')
    con = sqlite3.connect(path)
    assert con.execute('SELECT user, tag FROM tags').fetchall() == [('alice', 'cv')]
    assert [r[0] for r in con.execute('SELECT pid FROM fetches')] == ['new']  # old log rows pruned
    db.backup_users()  # replaces the one copy
