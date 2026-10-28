"""
Database support functions.
The idea is that none of the individual scripts deal directly with the file system.
Any of the file system I/O and the associated settings are in this single file.
"""

import os
import re
import sqlite3, zlib, pickle, tempfile
from sqlitedict import SqliteDict
from contextlib import contextmanager

# -----------------------------------------------------------------------------
# global configuration

# override to keep several corpora apart, e.g. ARXIV_SANITY_DATA=data/5y for a bulk import
DATA_DIR = os.environ.get('ARXIV_SANITY_DATA', 'data')

# -----------------------------------------------------------------------------
# utilities for safe writing of a pickle file


# Context managers for atomic writes courtesy of
# http://stackoverflow.com/questions/2333872/atomic-writing-to-file-with-python
@contextmanager
def _tempfile(*args, **kws):
    """Context for temporary file.
    Will find a free temporary filename upon entering
    and will try to delete the file on leaving
    Parameters
    ----------
    suffix : string
        optional file suffix
    """

    fd, name = tempfile.mkstemp(*args, **kws)
    os.close(fd)
    try:
        yield name
    finally:
        try:
            os.remove(name)
        except OSError as e:
            if e.errno == 2:
                pass
            else:
                raise e


@contextmanager
def open_atomic(filepath, *args, **kwargs):
    """Open temporary file object that atomically moves to destination upon
    exiting.
    Allows reading and writing to and from the same filename.
    Parameters
    ----------
    filepath : string
        the file path to be opened
    fsync : bool
        whether to force write the file to disk
    kwargs : mixed
        Any valid keyword arguments for :code:`open`
    """
    fsync = kwargs.pop('fsync', False)

    with _tempfile(dir=os.path.dirname(filepath)) as tmppath:
        with open(tmppath, *args, **kwargs) as f:
            yield f
            if fsync:
                f.flush()
                os.fsync(f.fileno())
        os.rename(tmppath, filepath)


def safe_pickle_dump(obj, fname):
    """
    prevents a case where one process could be writing a pickle file
    while another process is reading it, causing a crash. the solution
    is to write the pickle file to a temporary file and then move it.
    """
    with open_atomic(fname, 'wb') as f:
        pickle.dump(obj, f, -1)  # -1 specifies highest binary protocol


# -----------------------------------------------------------------------------


class CompressedSqliteDict(SqliteDict):
    """overrides the encode/decode methods to use zlib, so we get compressed storage"""

    def __init__(self, *args, **kwargs):

        def encode(obj):
            return sqlite3.Binary(zlib.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)))

        def decode(obj):
            return pickle.loads(zlib.decompress(bytes(obj)))

        super().__init__(*args, **kwargs, encode=encode, decode=decode)


# -----------------------------------------------------------------------------
"""
some docs to self:
flag='c': default mode, open for read/write, and creating the db/table if necessary
flag='r': open for read-only
"""


def _flag(filename, tablename, flag):
    """
    sqlitedict refuses to open a missing file or table read-only. tables only get
    created on first write, so on a fresh install (or for a user who never tagged
    anything) reads would crash; open those with 'c' instead, which creates them empty
    """
    if flag == 'r' and (not os.path.isfile(filename) or tablename not in SqliteDict.get_tablenames(filename)):
        return 'c'
    return flag


# stores info about papers, and also their lighter-weight metadata
PAPERS_DB_FILE = os.path.join(DATA_DIR, 'papers.db')
# stores account-relevant info, like which tags exist for which papers
DICT_DB_FILE = os.path.join(DATA_DIR, 'dict.db')


def get_papers_db(flag='r', autocommit=True):
    assert flag in ['r', 'c']
    pdb = CompressedSqliteDict(
        PAPERS_DB_FILE, tablename='papers', flag=_flag(PAPERS_DB_FILE, 'papers', flag), autocommit=autocommit
    )
    return pdb


def get_metas_db(flag='r', autocommit=True):
    assert flag in ['r', 'c']
    mdb = SqliteDict(
        PAPERS_DB_FILE, tablename='metas', flag=_flag(PAPERS_DB_FILE, 'metas', flag), autocommit=autocommit
    )
    return mdb


def get_tags_db(flag='r', autocommit=True):
    assert flag in ['r', 'c']
    tdb = CompressedSqliteDict(
        DICT_DB_FILE, tablename='tags', flag=_flag(DICT_DB_FILE, 'tags', flag), autocommit=autocommit
    )
    return tdb


def get_neg_tags_db(flag='r', autocommit=True):
    """like tags, but for explicit downvotes: user -> {tag: set(pids)}"""
    assert flag in ['r', 'c']
    ntdb = CompressedSqliteDict(
        DICT_DB_FILE, tablename='neg_tags', flag=_flag(DICT_DB_FILE, 'neg_tags', flag), autocommit=autocommit
    )
    return ntdb


def get_last_active_db(flag='r', autocommit=True):
    assert flag in ['r', 'c']
    ladb = SqliteDict(
        DICT_DB_FILE, tablename='last_active', flag=_flag(DICT_DB_FILE, 'last_active', flag), autocommit=autocommit
    )
    return ladb


def get_email_db(flag='r', autocommit=True):
    assert flag in ['r', 'c']
    edb = SqliteDict(DICT_DB_FILE, tablename='email', flag=_flag(DICT_DB_FILE, 'email', flag), autocommit=autocommit)
    return edb


# caches dense embeddings of papers, one table per embedding model
EMBEDDINGS_DB_FILE = os.path.join(DATA_DIR, 'embeddings.db')


def get_embeddings_db(model, flag='r', autocommit=True):
    """pid -> float16 embedding vector, for the given model string"""
    assert flag in ['r', 'c']
    tablename = 'emb_' + re.sub(r'[^A-Za-z0-9]+', '_', model)
    edb = SqliteDict(
        EMBEDDINGS_DB_FILE, tablename=tablename, flag=_flag(EMBEDDINGS_DB_FILE, tablename, flag), autocommit=autocommit
    )
    return edb


# -----------------------------------------------------------------------------
"""
our "feature store" is currently just a pickle file, may want to consider hdf5 in the future
"""

# stores tfidf features a bunch of other metadata
FEATURES_FILE = os.path.join(DATA_DIR, 'features.p')


def save_features(features):
    """takes the features dict and save it to disk in a simple pickle file"""
    safe_pickle_dump(features, FEATURES_FILE)


_features_cache = {'mtime': None, 'features': None}


def load_features():
    """
    loads the features dict from disk. the result is cached in memory and only
    re-read when compute.py has written a new file, so serving a request does not
    re-unpickle the whole feature matrix every time.
    """
    mtime = os.path.getmtime(FEATURES_FILE)
    if _features_cache['mtime'] != mtime:
        with open(FEATURES_FILE, 'rb') as f:
            _features_cache['features'] = pickle.load(f)
        _features_cache['mtime'] = mtime
    return _features_cache['features']
