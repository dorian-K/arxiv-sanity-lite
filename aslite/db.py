"""
Database support functions.
The idea is that none of the individual scripts deal directly with the file system.
Any of the file system I/O and the associated settings are in this single file.

Storage, all in DATA_DIR:
    papers.sqlite      papers, one row each, plus an FTS5 index for search that triggers
                       keep in sync. written by the fetch/import scripts, read by everything
    users.sqlite       accounts, tags and downvotes, written by serve.py
    embeddings.sqlite  per-paper embedding cache of compute.py, per model
    features/          what compute.py produces for ranking: .npy arrays that every reader
                       memory-maps, so web workers share one copy in the page cache
All databases run in WAL mode, so the web server keeps reading while the updater writes.
"""

import os
import re
import json
import time
import pickle
import fcntl
import shutil
import sqlite3
import secrets
import threading
from contextlib import contextmanager

import numpy as np
import scipy.sparse as sp

# -----------------------------------------------------------------------------
# global configuration

# override to keep several corpora apart, e.g. ARXIV_SANITY_DATA=data/5y for a bulk import
DATA_DIR = os.environ.get('ARXIV_SANITY_DATA', 'data')

PAPERS_DB_FILE = os.path.join(DATA_DIR, 'papers.sqlite')
USERS_DB_FILE = os.path.join(DATA_DIR, 'users.sqlite')
EMBEDDINGS_DB_FILE = os.path.join(DATA_DIR, 'embeddings.sqlite')
FEATURES_DIR = os.path.join(DATA_DIR, 'features')

# -----------------------------------------------------------------------------
# sqlite plumbing

_schema_done = set()
_schema_lock = threading.Lock()


def connect(path, schema, page_size=None):
    """
    a connection with the schema in place (created once per process and file). page_size
    only applies to a new file: it has to be set before WAL mode writes the first page
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    con = sqlite3.connect(path, timeout=60, check_same_thread=False)
    con.execute('PRAGMA busy_timeout = 60000')
    con.execute('PRAGMA synchronous = NORMAL')  # safe with WAL, and much faster for bulk writes
    key = os.path.abspath(path)
    if key not in _schema_done:
        with _schema_lock:
            if page_size:
                con.execute(f'PRAGMA page_size = {int(page_size)}')
            con.execute('PRAGMA journal_mode = WAL')  # stored in the file, so only needed once
            con.executescript(schema)
            _schema_done.add(key)
    return con


def snapshot(src, dst):
    """consistent copy of a live database (e.g. to send it to another machine)"""
    s, d = sqlite3.connect(src), sqlite3.connect(dst)
    with d:
        s.backup(d)
    s.close()
    d.close()


def db_mtime(path):
    """changes whenever anything is written to the database, WAL included"""
    return tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in (path, path + '-wal'))


# -----------------------------------------------------------------------------
# papers

PAPERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    n INTEGER PRIMARY KEY,      -- stable rowid, which the search index refers to
    id TEXT NOT NULL UNIQUE,    -- 2601.00001
    version INTEGER NOT NULL,
    time REAL NOT NULL,         -- when the latest version came out (unix time)
    time_str TEXT NOT NULL,
    published TEXT NOT NULL,    -- first version, 2026-01-01T00:00:00Z
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    authors TEXT NOT NULL,      -- json list of names
    categories TEXT NOT NULL,   -- json list, primary category first
    comment TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS papers_time ON papers(time);

-- full-text search over the papers table itself (external content: nothing is stored twice)
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    title, authors, summary, content='papers', content_rowid='n',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS papers_ai AFTER INSERT ON papers BEGIN
    INSERT INTO papers_fts(rowid, title, authors, summary) VALUES (new.n, new.title, new.authors, new.summary);
END;
CREATE TRIGGER IF NOT EXISTS papers_ad AFTER DELETE ON papers BEGIN
    INSERT INTO papers_fts(papers_fts, rowid, title, authors, summary)
        VALUES ('delete', old.n, old.title, old.authors, old.summary);
END;
CREATE TRIGGER IF NOT EXISTS papers_au AFTER UPDATE ON papers BEGIN
    INSERT INTO papers_fts(papers_fts, rowid, title, authors, summary)
        VALUES ('delete', old.n, old.title, old.authors, old.summary);
    INSERT INTO papers_fts(rowid, title, authors, summary) VALUES (new.n, new.title, new.authors, new.summary);
END;
"""
_COLS = 'id, version, time, time_str, published, title, summary, authors, categories, comment'
# bm25 weights per column: a word in the title counts for more than one in the abstract
SEARCH_WEIGHTS = (20.0, 10.0, 1.0)


def _row_to_paper(r):
    """a row as the paper dict the rest of the code uses (the shape arxiv_daemon.py always stored)"""
    pid, version, t, time_str, published, title, summary, authors, cats, comment = r
    return {
        '_id': pid,
        '_idv': f'{pid}v{version}',
        '_version': version,
        '_time': t,
        '_time_str': time_str,
        'link': f'https://arxiv.org/abs/{pid}v{version}',
        'published': published,
        'title': title,
        'summary': summary,
        'authors': [{'name': a} for a in json.loads(authors)],
        'tags': [{'term': c} for c in json.loads(cats)],
        'arxiv_comment': comment,
    }


def _paper_to_row(p):
    return (
        p['_id'],
        int(p['_version']),
        float(p['_time']),
        p['_time_str'],
        p.get('published', ''),
        p['title'],
        p['summary'],
        json.dumps([a['name'] for a in p['authors']], ensure_ascii=False),
        json.dumps([t['term'] for t in p['tags']]),
        p.get('arxiv_comment') or '',
    )


class PapersDB:
    """
    the papers. reads like a dict of pid -> paper dict (`pid in db`, `db[pid]`, `len(db)`,
    `db.keys()`, `db.items()`), writes go through upsert. use as a context manager or close()
    """

    def __init__(self, path=None):
        self.path = path or PAPERS_DB_FILE
        self.con = connect(self.path, PAPERS_SCHEMA)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.con.close()

    def __len__(self):
        return self.con.execute('SELECT count(*) FROM papers').fetchone()[0]

    def __contains__(self, pid):
        return self.con.execute('SELECT 1 FROM papers WHERE id = ?', (pid,)).fetchone() is not None

    def __getitem__(self, pid):
        r = self.con.execute(f'SELECT {_COLS} FROM papers WHERE id = ?', (pid,)).fetchone()
        if r is None:
            raise KeyError(pid)
        return _row_to_paper(r)

    def get_many(self, pids):
        """{pid: paper} for the pids that exist, in one query per 500"""
        pids, out = list(pids), {}
        for i in range(0, len(pids), 500):
            chunk = pids[i : i + 500]
            q = f"SELECT {_COLS} FROM papers WHERE id IN ({','.join('?' * len(chunk))})"
            for r in self.con.execute(q, chunk):
                out[r[0]] = _row_to_paper(r)
        return out

    def keys(self):
        return [r[0] for r in self.con.execute('SELECT id FROM papers ORDER BY n')]

    def items(self):
        """(pid, paper) for all papers, in insertion order, streamed"""
        for r in self.con.execute(f'SELECT {_COLS} FROM papers ORDER BY n'):
            yield r[0], _row_to_paper(r)

    def times(self):
        """{pid: time of the latest version}"""
        return dict(self.con.execute('SELECT id, time FROM papers'))

    def upsert(self, papers):
        """
        stores papers that are new, or newer versions of ones we have, in one transaction.
        returns how many were written. "newer" goes by version number, not time: times
        stored before calendar.timegm was used can be off by the fetching machine's utc offset
        """
        with self.con:
            cur = self.con.executemany(
                f"INSERT INTO papers ({_COLS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                + ', '.join(f'{c} = excluded.{c}' for c in _COLS.split(', ')[1:])
                + " WHERE excluded.version > papers.version",
                (_paper_to_row(p) for p in papers),
            )
        return cur.rowcount  # counts papers only, not the search index rows the triggers write

    def search(self, q, limit=2000):
        """
        best matching (pids, scores) for a free-text query. any word may match, words of 3+
        letters also as prefixes, so 'diffus' finds diffusion
        """
        words = re.findall(r'\w+', q.lower())
        if not words:
            return [], []
        match = ' OR '.join(f'"{w}"*' if len(w) >= 3 else f'"{w}"' for w in words)
        bm25 = 'bm25(papers_fts, %s)' % ', '.join(str(w) for w in SEARCH_WEIGHTS)
        rows = self.con.execute(
            f"SELECT p.id, -{bm25} FROM papers_fts JOIN papers p ON p.n = papers_fts.rowid "
            f"WHERE papers_fts MATCH ? ORDER BY {bm25} LIMIT ?",
            (match, limit),
        ).fetchall()
        return [r[0] for r in rows], [float(r[1]) for r in rows]


def get_papers_db(path=None):
    return PapersDB(path)


# -----------------------------------------------------------------------------
# users: accounts, tags, downvotes

USERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    name TEXT PRIMARY KEY,
    pw_hash TEXT,                  -- NULL for accounts from before passwords existed
    email TEXT NOT NULL DEFAULT '',
    last_active INTEGER NOT NULL DEFAULT 0
) WITHOUT ROWID;
-- one row per (user, tag, paper): liked (neg = 0), downvoted (neg = 1) or seen without an
-- opinion (neg = 2, the digest's neutral button: not shown again, not used for training)
CREATE TABLE IF NOT EXISTS tags (
    user TEXT NOT NULL,
    tag TEXT NOT NULL,
    pid TEXT NOT NULL,
    neg INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user, tag, pid)
) WITHOUT ROWID;
-- papers fetched from arXiv on a user's request, for rate limiting
CREATE TABLE IF NOT EXISTS fetches (
    time REAL NOT NULL,
    user TEXT NOT NULL,
    pid TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS fetches_time ON fetches(time);
-- the latest digest per user (see aslite/digest.py), json sections of [pid, score]
CREATE TABLE IF NOT EXISTS digests (
    user TEXT PRIMARY KEY,
    time REAL NOT NULL,
    days INTEGER NOT NULL,
    sections TEXT NOT NULL
) WITHOUT ROWID;
"""


def users_db():
    return connect(USERS_DB_FILE, USERS_SCHEMA)


def _run(sql, args=()):
    """one statement in its own transaction; returns the number of changed rows"""
    con = users_db()
    try:
        with con:
            return con.execute(sql, args).rowcount
    finally:
        con.close()


def _query(sql, args=()):
    con = users_db()
    try:
        return con.execute(sql, args).fetchall()
    finally:
        con.close()


def get_user(name):
    """{'name', 'pw_hash', 'email', 'last_active'} or None"""
    r = _query('SELECT name, pw_hash, email, last_active FROM users WHERE name = ?', (name,))
    return dict(zip(('name', 'pw_hash', 'email', 'last_active'), r[0], strict=True)) if r else None


def set_password(name, pw_hash):
    _run(
        'INSERT INTO users (name, pw_hash) VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET pw_hash = excluded.pw_hash',
        (name, pw_hash),
    )


def set_email(name, email):
    _run(
        'INSERT INTO users (name, email) VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET email = excluded.email',
        (name, email),
    )


def get_emails():
    """{user: email} for users who registered one"""
    return dict(_query("SELECT name, email FROM users WHERE email != ''"))


def touch_user(name, every=300):
    """records activity; writes at most once per `every` seconds per user"""
    now = int(time.time())
    _run('UPDATE users SET last_active = ? WHERE name = ? AND last_active < ?', (now, name, now - every))


LIKED, DOWNVOTED, SEEN = 0, 1, 2


def get_tags(user, neg=False):
    """{tag: set(pids)} of the user's likes (downvotes with neg=True, neutral ones with neg=SEEN)"""
    out = {}
    for tag, pid in _query('SELECT tag, pid FROM tags WHERE user = ? AND neg = ?', (user, int(neg))):
        out.setdefault(tag, set()).add(pid)
    return out


def get_all_tags(neg=False):
    """{user: {tag: set(pids)}} for everyone"""
    out = {}
    for user, tag, pid in _query('SELECT user, tag, pid FROM tags WHERE neg = ?', (int(neg),)):
        out.setdefault(user, {}).setdefault(tag, set()).add(pid)
    return out


def tag_paper(user, pid, tag, neg=False):
    """likes (downvotes, marks as seen) pid for tag, replacing whichever of those it was"""
    _run(
        'INSERT INTO tags (user, tag, pid, neg) VALUES (?, ?, ?, ?) ON CONFLICT DO UPDATE SET neg = excluded.neg',
        (user, tag, pid, int(neg)),
    )


def untag_paper(user, pid, tag, neg=False):
    """removes a like (or downvote); returns whether there was one"""
    return _run('DELETE FROM tags WHERE user = ? AND tag = ? AND pid = ? AND neg = ?', (user, tag, pid, int(neg))) > 0


def delete_tag(user, tag):
    """removes a tag with all its likes and downvotes; returns whether it existed"""
    return _run('DELETE FROM tags WHERE user = ? AND tag = ?', (user, tag)) > 0


def save_digest(user, sections, days):
    _run(
        'INSERT OR REPLACE INTO digests (user, time, days, sections) VALUES (?, ?, ?, ?)',
        (user, time.time(), days, json.dumps(sections)),
    )


def get_digest(user):
    """{'time', 'days', 'sections'} of the user's latest digest, or None"""
    r = _query('SELECT time, days, sections FROM digests WHERE user = ?', (user,))
    return {'time': r[0][0], 'days': r[0][1], 'sections': json.loads(r[0][2])} if r else None


def log_fetch(user, pid):
    _run('INSERT INTO fetches (time, user, pid) VALUES (?, ?, ?)', (time.time(), user, pid))


def backup_users(dest=None):
    """
    one consistent copy of users.sqlite (accounts, tags, votes: the only data that can't
    be rebuilt), replaced atomically each time. also drops fetch log rows older than the
    day the rate limits look at, the one table that would otherwise keep growing
    """
    _run('DELETE FROM fetches WHERE time < ?', (time.time() - 2 * 86400,))
    dest = dest or os.path.join(DATA_DIR, 'backups', 'users.sqlite')
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    snapshot(USERS_DB_FILE, dest + '.tmp')
    os.replace(dest + '.tmp', dest)
    return dest


def count_fetches(since, user=None):
    if user is None:
        return _query('SELECT count(*) FROM fetches WHERE time > ?', (since,))[0][0]
    return _query('SELECT count(*) FROM fetches WHERE time > ? AND user = ?', (since, user))[0][0]


ARXIV_LOCK_FILE = os.path.join(DATA_DIR, 'arxiv.lock')
# set by serve.py when users added papers; the docker updater then recomputes features
COMPUTE_FLAG_FILE = os.path.join(DATA_DIR, 'compute_requested')


@contextmanager
def arxiv_slot(pause=3.5):
    """
    waits for our turn to call the arXiv API: at most one call per `pause` seconds across
    all processes on this machine (web workers, scripts), as arXiv asks
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(ARXIV_LOCK_FILE, 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            last = float(f.read() or 0)
            time.sleep(max(0.0, last + pause - time.time()))
            yield
        finally:
            f.seek(0)
            f.truncate()
            f.write(str(time.time()))
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)


def request_compute():
    with open(COMPUTE_FLAG_FILE, 'w') as f:
        f.write(str(time.time()))


def get_secret_key():
    """
    the key that signs session cookies: $SECRET_KEY, else secret_key.txt in the repo root
    (the original location), else DATA_DIR/secret_key.txt, generated on first use so
    it survives restarts and container rebuilds along with the rest of the data
    """
    if os.environ.get('SECRET_KEY'):
        return os.environ['SECRET_KEY']
    for path in ('secret_key.txt', os.path.join(DATA_DIR, 'secret_key.txt')):
        if os.path.isfile(path):
            return open(path).read().strip()
    os.makedirs(DATA_DIR, exist_ok=True)
    key = secrets.token_urlsafe(32)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:  # another worker process just created it
        time.sleep(0.5)
        return open(path).read().strip()
    with os.fdopen(fd, 'w') as f:
        f.write(key)
    return key


# -----------------------------------------------------------------------------
# embedding cache

# a rowid table on 8 KB pages: a 768-dim float16 vector is 1.5 KB, which a WITHOUT ROWID table
# (the first version) pushes onto its own overflow page, 4.7 KB per row on disk; this way
# it's 1.7 KB (measured on 50k EmbeddingGemma vectors), and reading them all is faster too
EMBEDDINGS_PAGE_SIZE = 8192
EMBEDDINGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    model TEXT NOT NULL,
    pid TEXT NOT NULL,
    vec BLOB NOT NULL,  -- float16
    PRIMARY KEY (model, pid)
);
"""


def _compact_embeddings(con):
    """one-time rebuild of a cache made with the first, WITHOUT ROWID layout (~2.8x smaller)"""
    sql = con.execute("SELECT sql FROM sqlite_master WHERE name = 'embeddings'").fetchone()
    if not sql or 'WITHOUT ROWID' not in sql[0].upper():
        return
    print("rebuilding the embedding cache in its smaller layout (once, may take a minute)...")
    with con:
        con.execute(
            "CREATE TABLE embeddings_new (model TEXT NOT NULL, pid TEXT NOT NULL, vec BLOB NOT NULL, "
            "PRIMARY KEY (model, pid))"
        )
        con.execute("INSERT INTO embeddings_new SELECT model, pid, vec FROM embeddings")
        con.execute("DROP TABLE embeddings")
        con.execute("ALTER TABLE embeddings_new RENAME TO embeddings")
    # the page size can only change outside WAL mode, through a VACUUM
    con.execute('PRAGMA journal_mode = DELETE')
    con.execute(f'PRAGMA page_size = {EMBEDDINGS_PAGE_SIZE}')
    con.execute('VACUUM')
    con.execute('PRAGMA journal_mode = WAL')


class EmbeddingCache:
    """pid -> float16 vector for one model string"""

    def __init__(self, model, path=None):
        self.model = model
        self.con = connect(path or EMBEDDINGS_DB_FILE, EMBEDDINGS_SCHEMA, page_size=EMBEDDINGS_PAGE_SIZE)
        _compact_embeddings(self.con)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.con.close()

    def pids(self):
        return {r[0] for r in self.con.execute('SELECT pid FROM embeddings WHERE model = ?', (self.model,))}

    def put(self, pids, x):
        with self.con:
            self.con.executemany(
                'INSERT OR REPLACE INTO embeddings (model, pid, vec) VALUES (?, ?, ?)',
                ((self.model, p, np.asarray(v, dtype=np.float16).tobytes()) for p, v in zip(pids, x, strict=True)),
            )

    def matrix(self, pids):
        """float32 matrix with the rows of pids, which must all be cached"""
        ix = {p: i for i, p in enumerate(pids)}
        x, seen = None, 0
        for pid, vec in self.con.execute('SELECT pid, vec FROM embeddings WHERE model = ?', (self.model,)):
            i = ix.get(pid)
            if i is None:
                continue
            v = np.frombuffer(vec, dtype=np.float16)
            if x is None:
                x = np.zeros((len(pids), len(v)), dtype=np.float32)
            x[i] = v
            seen += 1
        if seen != len(pids):
            raise KeyError(f"{len(pids) - seen} papers have no cached {self.model} embedding")
        return x


# -----------------------------------------------------------------------------
# features: what compute.py writes for ranking
#
# every write goes to a new directory features/<version>/; features/current is a symlink
# to the newest one and is swapped atomically, so readers never see half a write. the
# arrays are memory-mapped: all processes share them through the page cache, and a
# process that is still using an old version keeps its mapping after it is deleted.

_ARRAYS = ('x_data', 'x_indices', 'x_indptr', 'x_emb')


def features_exist(path=None):
    path = path or FEATURES_DIR
    if path.endswith('.p'):
        return os.path.isfile(path)
    return os.path.exists(os.path.join(path, 'current'))


def save_features(features):
    os.makedirs(FEATURES_DIR, exist_ok=True)
    version = str(time.time_ns())
    d = os.path.join(FEATURES_DIR, version)
    os.makedirs(d)
    x = features['x'].tocsr()
    arrays = {'x_data': x.data, 'x_indices': x.indices, 'x_indptr': x.indptr}
    if features.get('x_emb') is not None:
        arrays['x_emb'] = np.ascontiguousarray(features['x_emb'], dtype=np.float32)
    for k, v in arrays.items():
        np.save(os.path.join(d, k + '.npy'), v)
    meta = {k: v for k, v in features.items() if k not in ('x', 'x_emb', 'ptoi')}
    meta['x_shape'] = x.shape
    with open(os.path.join(d, 'meta.p'), 'wb') as f:
        pickle.dump(meta, f, -1)
    link = os.path.join(FEATURES_DIR, 'current')
    tmp = link + '.tmp'
    if os.path.lexists(tmp):
        os.remove(tmp)
    os.symlink(version, tmp)
    os.replace(tmp, link)
    for old in os.listdir(FEATURES_DIR):
        if old not in (version, 'current') and old.isdigit():
            shutil.rmtree(os.path.join(FEATURES_DIR, old), ignore_errors=True)


def _load(d):
    with open(os.path.join(d, 'meta.p'), 'rb') as f:
        features = pickle.load(f)
    a = {}
    for k in _ARRAYS:
        p = os.path.join(d, k + '.npy')
        if os.path.isfile(p):
            a[k] = np.load(p, mmap_mode='r')
    shape = features.pop('x_shape')
    features['x'] = sp.csr_matrix((a['x_data'], a['x_indices'], a['x_indptr']), shape=shape, copy=False)
    features['x_emb'] = a.get('x_emb')
    return features


_features_cache = {}
_features_lock = threading.Lock()


def load_features(path=None):
    """
    the features dict: pids, x (sparse tfidf), vocab, idf, and x_emb / emb_model with
    embeddings, plus ptoi (pid -> row). cached per process until compute.py writes a new
    version. path can also be an old-style features .p pickle (e.g. for eval.py)
    """
    path = path or FEATURES_DIR
    with _features_lock:
        if path.endswith('.p'):
            key = (path, os.path.getmtime(path))
        else:
            key = (path, os.path.realpath(os.path.join(path, 'current')))
        if _features_cache.get('key') != key:
            if path.endswith('.p'):
                with open(path, 'rb') as f:
                    features = pickle.load(f)
            else:
                features = _load(key[1])
            features['ptoi'] = {p: i for i, p in enumerate(features['pids'])}
            _features_cache.update(key=key, features=features)
        return _features_cache['features']
