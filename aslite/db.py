"""
Database support functions.
The idea is that none of the individual scripts deal directly with the file system.
Any of the file system I/O and the associated settings are in this single file.

Storage, all in DATA_DIR:
    papers.sqlite      papers, one row each, plus an FTS5 index for search that triggers
                       keep in sync. written by the fetch/import scripts, read by everything
    users.sqlite       accounts, tags and downvotes, written by serve.py
    embeddings.sqlite  per-paper embedding cache of compute.py, per model
    models.sqlite      the trained ranking model per user and tag(s), a cache that serve.py
                       fills and that can be deleted any time
    features/          what compute.py produces for ranking: .npy arrays that every reader
                       memory-maps, so web workers share one copy in the page cache
All databases run in WAL mode, so the web server keeps reading while the updater writes.
"""

import io
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

from aslite.arxiv import venue

# -----------------------------------------------------------------------------
# global configuration

# override to keep several corpora apart, e.g. ARXIV_SANITY_DATA=data/5y for a bulk import
DATA_DIR = os.environ.get('ARXIV_SANITY_DATA', 'data')

PAPERS_DB_FILE = os.path.join(DATA_DIR, 'papers.sqlite')
USERS_DB_FILE = os.path.join(DATA_DIR, 'users.sqlite')
EMBEDDINGS_DB_FILE = os.path.join(DATA_DIR, 'embeddings.sqlite')
FEATURES_DIR = os.path.join(DATA_DIR, 'features')
MODELS_DB_FILE = os.path.join(DATA_DIR, 'models.sqlite')
SEARCH_TERMS_FILE = os.path.join(DATA_DIR, 'search_terms.json')  # aslite/spell.py
METRICS_DB_FILE = os.path.join(DATA_DIR, 'metrics.sqlite')  # timings for /stats, safe to delete
REFS_DB_FILE = os.path.join(DATA_DIR, 'refs.sqlite')  # references of papers (aslite/refs.py)
THUMBS_DIR = os.path.join(DATA_DIR, 'thumbs')  # aslite/thumbs.py: <pid>.figure.jpg, <pid>.page.jpg

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
    comment TEXT NOT NULL DEFAULT '',
    journal_ref TEXT NOT NULL DEFAULT '',  -- where it was published, if the authors said so
    venue TEXT NOT NULL DEFAULT ''         -- aslite.arxiv.venue() of comment and journal_ref
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
CREATE TRIGGER IF NOT EXISTS papers_au AFTER UPDATE OF title, authors, summary ON papers BEGIN
    INSERT INTO papers_fts(papers_fts, rowid, title, authors, summary)
        VALUES ('delete', old.n, old.title, old.authors, old.summary);
    INSERT INTO papers_fts(rowid, title, authors, summary) VALUES (new.n, new.title, new.authors, new.summary);
END;
"""
_COLS = 'id, version, time, time_str, published, title, summary, authors, categories, comment, journal_ref, venue'
# fewer papers that have every word of a query: also show those with any of them
SEARCH_MIN_ALL = 50

# bm25 weights per column: a word in the title counts for more than one in the abstract
SEARCH_WEIGHTS = (20.0, 10.0, 1.0)


def _row_to_paper(r):
    """a row as the paper dict the rest of the code uses (the shape arxiv_daemon.py always stored)"""
    pid, version, t, time_str, published, title, summary, authors, cats, comment, journal_ref, venue_ = r
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
        'arxiv_journal_ref': journal_ref,
        'venue': venue_,
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
        p.get('arxiv_journal_ref') or '',
        venue(p.get('arxiv_comment') or '', p.get('arxiv_journal_ref') or ''),
    )


_papers_columns_done = set()


class PapersDB:
    """
    the papers. reads like a dict of pid -> paper dict (`pid in db`, `db[pid]`, `len(db)`,
    `db.keys()`, `db.items()`), writes go through upsert. use as a context manager or close()
    """

    def __init__(self, path=None):
        self.path = path or PAPERS_DB_FILE
        self.con = connect(self.path, PAPERS_SCHEMA)
        key = os.path.abspath(self.path)
        if key not in _papers_columns_done:
            self._add_columns()
            _papers_columns_done.add(key)

    def _add_columns(self):
        """
        columns added after the table first shipped (CREATE TABLE IF NOT EXISTS won't add them).
        under the write lock, so two processes starting at once don't both add them
        """
        if {'journal_ref', 'venue'} <= {r[1] for r in self.con.execute('PRAGMA table_info(papers)')}:
            return
        self.con.execute('BEGIN IMMEDIATE')
        try:
            cols = {r[1] for r in self.con.execute('PRAGMA table_info(papers)')}
            if 'journal_ref' not in cols:
                self.con.execute("ALTER TABLE papers ADD COLUMN journal_ref TEXT NOT NULL DEFAULT ''")
            if 'venue' not in cols:
                self.con.execute("ALTER TABLE papers ADD COLUMN venue TEXT NOT NULL DEFAULT ''")
                # the search index only needs updating when its own columns change, not for this
                self.con.execute('DROP TRIGGER IF EXISTS papers_au')
                self.con.execute(PAPERS_SCHEMA[PAPERS_SCHEMA.index('CREATE TRIGGER IF NOT EXISTS papers_au') :].strip())
                # one-time backfill, about 20 s for 465k papers
                self.con.create_function('venue', 2, venue, deterministic=True)
                self.con.execute('UPDATE papers SET venue = venue(comment, journal_ref)')
            self.con.execute('COMMIT')
        except BaseException:
            self.con.execute('ROLLBACK')
            raise

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

    def venues(self):
        """{pid: venue} of the papers that say where they were accepted or published"""
        return dict(self.con.execute("SELECT id, venue FROM papers WHERE venue != ''"))

    def recompute_venues(self):
        """fills the venue column again, after aslite.arxiv.venue() changed (a few seconds per 100k papers)"""
        self.con.create_function('venue', 2, venue, deterministic=True)
        with self.con:
            self.con.execute("UPDATE papers SET venue = venue(comment, journal_ref)")

    def upsert(self, papers):
        """
        stores papers that are new, or newer versions of ones we have, in one transaction.
        returns how many were written. "newer" goes by version number, not time: times
        stored before calendar.timegm was used can be off by the fetching machine's utc offset
        """
        with self.con:
            cur = self.con.executemany(
                f"INSERT INTO papers ({_COLS}) VALUES ({', '.join('?' * len(_COLS.split(', ')))}) "
                "ON CONFLICT(id) DO UPDATE SET "
                + ', '.join(f'{c} = excluded.{c}' for c in _COLS.split(', ')[1:])
                + " WHERE excluded.version > papers.version",
                (_paper_to_row(p) for p in papers),
            )
        return cur.rowcount  # counts papers only, not the search index rows the triggers write

    def search(self, q, limit=2000):
        """
        best matching (pids, scores) for a free-text query: papers with every word first, then,
        if those are few, papers with any of them. words of 3+ letters also match as prefixes,
        so 'diffus' finds diffusion. stopwords are left out (else 'a survey of ...' matches every
        paper); ranking costs time per matching paper, ~0.5 s per 300k
        """
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # only here: a heavy import

        words = re.findall(r'\w+', q.lower())
        words = [w for w in words if w not in ENGLISH_STOP_WORDS] or words
        if not words:
            return [], []
        terms = [f'"{w}"*' if len(w) >= 3 else f'"{w}"' for w in words]
        bm25 = 'bm25(papers_fts, %s)' % ', '.join(str(w) for w in SEARCH_WEIGHTS)

        def run(match, n):
            # rank inside the index and join only the best: joining every match costs as much again
            return self.con.execute(
                f"SELECT p.id, s FROM (SELECT rowid AS r, -{bm25} AS s FROM papers_fts "
                f"WHERE papers_fts MATCH ? ORDER BY {bm25} LIMIT ?) JOIN papers p ON p.n = r ORDER BY s DESC",
                (match, n),
            ).fetchall()

        rows = run(' AND '.join(terms), limit)
        if len(terms) > 1 and len(rows) < SEARCH_MIN_ALL:
            seen = {r[0] for r in rows}
            low = min((r[1] for r in rows), default=0.0)
            rest = [r for r in run(' OR '.join(terms), limit) if r[0] not in seen]
            # below every paper that has all the words
            rows += [(p, min(sc, low) - 1e-6 * (i + 1)) for i, (p, sc) in enumerate(rest)][: limit - len(rows)]
        return [r[0] for r in rows], [float(r[1]) for r in rows]

    def _vocab(self):
        self.con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.papers_vocab USING fts5vocab(main, papers_fts, row)")

    def terms(self, min_docs):
        """{word: number of papers} of the search index, for the words in at least min_docs (seconds)"""
        self._vocab()
        return dict(self.con.execute('SELECT term, doc FROM temp.papers_vocab WHERE doc >= ?', (min_docs,)))

    def term_docs(self, word):
        """in how many papers this word is (a few ms)"""
        self._vocab()
        row = self.con.execute('SELECT doc FROM temp.papers_vocab WHERE term = ?', (word,)).fetchone()
        return row[0] if row else 0


def save_terms(terms):
    tmp = SEARCH_TERMS_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(terms, f, separators=(',', ':'))
    os.replace(tmp, SEARCH_TERMS_FILE)


def load_terms():
    """(the words saved by compute.py, the file's mtime), or (None, None) before its first run"""
    try:
        mtime = os.path.getmtime(SEARCH_TERMS_FILE)
        with open(SEARCH_TERMS_FILE) as f:
            return json.load(f), mtime
    except FileNotFoundError:
        return None, None


def get_papers_db(path=None):
    return PapersDB(path)


# -----------------------------------------------------------------------------
# users: accounts, tags, downvotes

USERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    name TEXT PRIMARY KEY,
    pw_hash TEXT,                  -- NULL for accounts from before passwords existed
    email TEXT NOT NULL DEFAULT '',
    last_active INTEGER NOT NULL DEFAULT 0,
    public INTEGER NOT NULL DEFAULT 0,  -- 1: anyone may view this user's tags read-only, without the password
    pending_email TEXT NOT NULL DEFAULT '',  -- entered, but the confirmation link not clicked yet
    email_confirmed REAL  -- when the owner of `email` confirmed it (double opt-in): the consent record
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
-- papers per tag in the digest, where the user changed it (0 = leave the tag out)
CREATE TABLE IF NOT EXISTS digest_sizes (
    user TEXT NOT NULL,
    tag TEXT NOT NULL,
    n INTEGER NOT NULL,
    PRIMARY KEY (user, tag)
) WITHOUT ROWID;
"""


_users_columns_done = set()


def users_db():
    con = connect(USERS_DB_FILE, USERS_SCHEMA)
    key = os.path.abspath(USERS_DB_FILE)
    if key not in _users_columns_done:
        # columns added after the table first shipped; CREATE TABLE IF NOT EXISTS won't add them
        cols = {r[1] for r in con.execute('PRAGMA table_info(users)')}
        added = {
            'public': 'INTEGER NOT NULL DEFAULT 0',
            'pending_email': "TEXT NOT NULL DEFAULT ''",
            'email_confirmed': 'REAL',
        }
        with con:
            for col, decl in added.items():
                if col not in cols:
                    con.execute(f'ALTER TABLE users ADD COLUMN {col} {decl}')
        _users_columns_done.add(key)
    return con


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


USER_KEYS = ('name', 'pw_hash', 'email', 'last_active', 'public', 'pending_email', 'email_confirmed')


def get_user(name):
    """{'name', 'pw_hash', 'email', ...} (see USER_KEYS) or None"""
    keys = USER_KEYS
    r = _query(f'SELECT {", ".join(keys)} FROM users WHERE name = ?', (name,))
    return dict(zip(keys, r[0], strict=True)) if r else None


def set_public(name, public):
    """whether others may view this user's tags read-only, without the password"""
    _run('UPDATE users SET public = ? WHERE name = ?', (int(bool(public)), name))


def set_password(name, pw_hash):
    _run(
        'INSERT INTO users (name, pw_hash) VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET pw_hash = excluded.pw_hash',
        (name, pw_hash),
    )


def request_email(name, email):
    """an address the user entered; it receives nothing but the confirmation mail until confirmed"""
    _run(
        'INSERT INTO users (name, pending_email) VALUES (?, ?) '
        'ON CONFLICT(name) DO UPDATE SET pending_email = excluded.pending_email',
        (name, email),
    )


def confirm_email(name, email):
    """the owner clicked the confirmation link: the address gets digests from now on"""
    return (
        _run(
            "UPDATE users SET email = ?, email_confirmed = ?, pending_email = '' WHERE name = ? AND pending_email = ?",
            (email, time.time(), name, email),
        )
        > 0
    )


def set_email(name, email):
    """only for removing an address ('', e.g. unsubscribe); new ones go through request_email"""
    assert email == '', 'new addresses must be confirmed first, see request_email'
    _run("UPDATE users SET email = '', email_confirmed = NULL, pending_email = '' WHERE name = ?", (name,))


def get_emails():
    """{user: email} for users with a confirmed address"""
    return dict(_query("SELECT name, email FROM users WHERE email != '' AND email_confirmed IS NOT NULL"))


def delete_user(name):
    """erases everything stored about a user (the GDPR's right to erasure)"""
    con = users_db()
    try:
        with con:
            for table, col in (
                ('tags', 'user'),
                ('digests', 'user'),
                ('digest_sizes', 'user'),
                ('fetches', 'user'),
                ('users', 'name'),
            ):
                con.execute(f'DELETE FROM {table} WHERE {col} = ?', (name,))
    finally:
        con.close()
    delete_models(name)


def export_user(name):
    """everything stored about a user, as plain data (the GDPR's right to data portability)"""
    u = get_user(name) or {}
    kinds = {LIKED: 'liked', DOWNVOTED: 'not for', SEEN: 'neutral'}
    tags = {}
    for tag, pid, neg in _query('SELECT tag, pid, neg FROM tags WHERE user = ? ORDER BY tag, pid', (name,)):
        tags.setdefault(tag, {}).setdefault(kinds[neg], []).append(pid)
    return {
        'username': name,
        'email': u.get('email', ''),
        'email_confirmed_at': u.get('email_confirmed'),
        'pending_email': u.get('pending_email', ''),
        'last_active': u.get('last_active'),
        'shares_tags_read_only': bool(u.get('public')),
        'tags': tags,
        'latest_digest': get_digest(name),
        'digest_papers_per_tag': get_digest_sizes(name),
        'arxiv_fetches_last_two_days': [
            {'time': t, 'pid': p} for t, p in _query('SELECT time, pid FROM fetches WHERE user = ?', (name,))
        ],
        'note': 'the password is stored only as a salted hash, which is not included',
    }


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
    _run('DELETE FROM digest_sizes WHERE user = ? AND tag = ?', (user, tag))
    existed = _run('DELETE FROM tags WHERE user = ? AND tag = ?', (user, tag)) > 0
    delete_models(user, tag)
    return existed


def get_digest_sizes(user):
    """{tag: papers in the digest} for the tags where the user changed it from the default"""
    return dict(_query('SELECT tag, n FROM digest_sizes WHERE user = ?', (user,)))


def set_digest_size(user, tag, n):
    """n papers of this tag in the digest, None for the default"""
    if n is None:
        _run('DELETE FROM digest_sizes WHERE user = ? AND tag = ?', (user, tag))
    else:
        _run('INSERT OR REPLACE INTO digest_sizes (user, tag, n) VALUES (?, ?, ?)', (user, tag, n))


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
HEAVY_LOCK_FILE = os.path.join(DATA_DIR, 'heavy.lock')  # compute.py vs embed_daemon.py
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


def hold_heavy_lock():
    """
    for compute.py: waits until no embed_daemon.py session runs, and keeps it from starting one
    until this process exits, so the two (~2 GB of RAM each) never run at the same time
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    f = open(HEAVY_LOCK_FILE, 'a')
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("waiting for the embedding session to finish (at most a few minutes)...", flush=True)
        fcntl.flock(f, fcntl.LOCK_EX)
    return f  # the lock lasts as long as the file stays open, so until exit


@contextmanager
def heavy_slot():
    """for embed_daemon.py: yields False (do nothing) while compute.py runs, else True"""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(HEAVY_LOCK_FILE, 'a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


LAST_DIGEST_FILE = os.path.join(DATA_DIR, 'last_digest')


def get_last_digest():
    """unix time of the last scheduled digest send, or None before the first (send_emails.py)"""
    try:
        with open(LAST_DIGEST_FILE) as f:
            return float(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def set_last_digest(t):
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = LAST_DIGEST_FILE + '.tmp'
    with open(tmp, 'w') as f:
        f.write(str(t))
    os.replace(tmp, LAST_DIGEST_FILE)  # atomic: a crash can't leave an empty file behind


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

    def vectors(self, pids):
        """float32 rows of pids that are cached, in the order given (missing ones are skipped)"""
        out = []
        for i in range(0, len(pids), 500):
            chunk = pids[i : i + 500]
            q = f"SELECT pid, vec FROM embeddings WHERE model = ? AND pid IN ({','.join('?' * len(chunk))})"
            got = dict(self.con.execute(q, (self.model, *chunk)))
            out += [np.frombuffer(got[p], dtype=np.float16) for p in chunk if p in got]
        return np.array(out, dtype=np.float32)

    def matrix(self, pids, partial=False, transform=None, chunk=20000):
        """
        float32 matrix with the rows of pids, which must all be cached; with partial, also
        returns which are (the others stay zero), and (None, mask) if none is. transform maps
        a block of rows to their final form (e.g. PCA) while they stream in, so the full-size
        matrix never exists (at 657k papers x 768 dims that alone is 2 GB)
        """
        transform = transform or (lambda v: v)
        ix = {p: i for i, p in enumerate(pids)}
        x, have = None, np.zeros(len(pids), dtype=bool)
        rows, buf = [], []

        def flush():
            nonlocal x
            out = transform(np.array(buf, dtype=np.float32))
            if x is None:
                x = np.zeros((len(pids), out.shape[1]), dtype=np.float32)
            x[rows] = out
            rows.clear()
            buf.clear()

        for pid, vec in self.con.execute('SELECT pid, vec FROM embeddings WHERE model = ?', (self.model,)):
            i = ix.get(pid)
            if i is None:
                continue
            rows.append(i)
            buf.append(np.frombuffer(vec, dtype=np.float16))
            have[i] = True
            if len(buf) >= chunk:
                flush()
        if buf:
            flush()
        if partial:
            return x, have
        if not have.all():
            raise KeyError(f"{int((~have).sum())} papers have no cached {self.model} embedding")
        return x

    def merge(self, path):
        """
        adds the embeddings of another embeddings.sqlite (e.g. computed on a GPU machine) that
        this cache doesn't have yet, for every model; returns how many. existing ones are kept
        """
        self.con.execute('ATTACH DATABASE ? AS other', (path,))
        try:
            with self.con:
                return self.con.execute(
                    'INSERT OR IGNORE INTO embeddings (model, pid, vec) SELECT model, pid, vec FROM other.embeddings'
                ).rowcount
        finally:
            self.con.execute('DETACH DATABASE other')


# -----------------------------------------------------------------------------
# features: what compute.py writes for ranking
#
# every write goes to a new directory features/<version>/; features/current is a symlink
# to the newest one and is swapped atomically, so readers never see half a write. the
# arrays are memory-mapped: all processes share them through the page cache, and a
# process that is still using an old version keeps its mapping after it is deleted.

_ARRAYS = ('x_data', 'x_indices', 'x_indptr', 'x_emb', 'emb_mask')


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
    if features.get('emb_mask') is not None:
        arrays['emb_mask'] = np.ascontiguousarray(features['emb_mask'], dtype=bool)
    for k, v in arrays.items():
        np.save(os.path.join(d, k + '.npy'), v)
    # (keys with _ are what load_features adds)
    meta = {k: v for k, v in features.items() if k not in ('x', 'x_emb', 'emb_mask', 'ptoi') and not k.startswith('_')}
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
    features['emb_mask'] = a.get('emb_mask')  # None: every paper has an embedding
    return features


_features_cache = {}
_features_lock = threading.Lock()


def features_version():
    """changes whenever load_features() would return a different features dict"""
    entry = _features_cache.get('entry')
    return entry[0] if entry else None


def load_features(path=None):
    """
    the features dict: pids, x (sparse tfidf), vocab, idf, and x_emb / emb_model with
    embeddings, plus ptoi (pid -> row). cached per process until compute.py writes a new
    version. path can also be an old-style features .p pickle (e.g. for eval.py)
    """
    path = path or FEATURES_DIR
    if path.endswith('.p'):
        key = (path, os.path.getmtime(path))
    else:
        key = (path, os.path.realpath(os.path.join(path, 'current')))
    cached = _features_cache.get('entry')
    if cached and cached[0] == key:
        return cached[1]
    # a new version takes seconds to load: while one thread loads it, the others keep
    # answering from the old one instead of queueing behind the lock
    stale = cached[1] if cached and cached[0][0] == path else None
    if not _features_lock.acquire(blocking=stale is None):
        return stale
    try:
        cached = _features_cache.get('entry')
        if not (cached and cached[0] == key):
            if path.endswith('.p'):
                with open(path, 'rb') as f:
                    features = pickle.load(f)
            else:
                features = _load(key[1])
            features['ptoi'] = {p: i for i, p in enumerate(features['pids'])}
            features['_version'] = str(key[1])  # what a model trained on these is valid for
            _features_cache['entry'] = cached = (key, features)  # one swap, never half updated
        return cached[1]
    finally:
        _features_lock.release()


# -----------------------------------------------------------------------------
# trained ranking models (see aslite/rank.py fit), cached across requests and web workers

MODELS_SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    user TEXT NOT NULL,
    tags TEXT NOT NULL,       -- what it ranks by: a tag, several (a,b) or 'all'
    variant TEXT NOT NULL,    -- feats|model|C
    fp TEXT NOT NULL,         -- rank.fingerprint() of the exact papers it was trained on
    version TEXT NOT NULL,    -- the features version it was trained on
    used REAL NOT NULL,
    meta TEXT NOT NULL,       -- the model dict without its arrays, as json
    arrays BLOB NOT NULL,     -- its arrays, as an .npz
    PRIMARY KEY (user, tags, variant, fp, version)
);
"""
MODELS_PER_SLOT = 3  # older training sets of the same user, tags and variant get dropped


def _models_db():
    return connect(MODELS_DB_FILE, MODELS_SCHEMA)


def has_model(user, tags, variant, fp, version):
    """whether get_model would find it, without reading it"""
    con = _models_db()
    try:
        q = 'SELECT 1 FROM models WHERE user=? AND tags=? AND variant=? AND fp=? AND version=?'
        return con.execute(q, (user, tags, variant, fp, version)).fetchone() is not None
    finally:
        con.close()


def get_model(user, tags, variant, fp, version):
    """
    the model trained on exactly fp with features version, or None. the key names the
    training set, so a model can never stand in for another state of the user's tags
    """
    con = _models_db()
    try:
        key = (user, tags, variant, fp, version)
        row = con.execute(
            'SELECT meta, arrays, used FROM models WHERE user=? AND tags=? AND variant=? AND fp=? AND version=?', key
        ).fetchone()
        if row is None:
            return None
        now = time.time()
        if now - row[2] > 3600:  # for pruning; at most one write an hour per model
            with con:
                con.execute(
                    'UPDATE models SET used=? WHERE user=? AND tags=? AND variant=? AND fp=? AND version=?',
                    (now, *key),
                )
    finally:
        con.close()
    m = json.loads(row[0])
    with np.load(io.BytesIO(row[1]), allow_pickle=False) as a:
        for i, part in enumerate(m['parts']):
            part['coef'] = a[f'coef{i}']
        for k in ('top', 'quantiles'):
            if k in a:
                m[k] = a[k]
    return m


def put_model(user, tags, variant, fp, version, m):
    """
    stores a model. the same key always holds the same model (training is deterministic),
    so two workers storing it at once is harmless. keeps the newest MODELS_PER_SLOT per
    user, tags and variant, and drops models of other features versions an hour after their use
    """
    arrays = {f'coef{i}': p['coef'] for i, p in enumerate(m['parts'])}
    arrays.update({k: m[k] for k in ('top', 'quantiles') if k in m})
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    meta = {k: v for k, v in m.items() if k not in ('top', 'quantiles')}
    meta['parts'] = [{k: v for k, v in p.items() if k != 'coef'} for p in m['parts']]
    now = time.time()
    con = _models_db()
    try:
        with con:
            con.execute(
                'INSERT OR REPLACE INTO models VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (user, tags, variant, fp, version, now, json.dumps(meta), buf.getvalue()),
            )
            con.execute(
                'DELETE FROM models WHERE user=? AND tags=? AND variant=? AND rowid NOT IN '
                '(SELECT rowid FROM models WHERE user=? AND tags=? AND variant=? ORDER BY used DESC LIMIT ?)',
                (user, tags, variant, user, tags, variant, MODELS_PER_SLOT),
            )
            con.execute('DELETE FROM models WHERE version != ? AND used < ?', (version, now - 3600))
    finally:
        con.close()


def delete_models(user, tag=None):
    """erases the user's models (derived from their tags), or those that a tag went into"""
    if not os.path.exists(MODELS_DB_FILE):
        return
    con = _models_db()
    try:
        with con:
            if tag is None:
                con.execute('DELETE FROM models WHERE user=?', (user,))
            else:
                for (tags,) in con.execute('SELECT DISTINCT tags FROM models WHERE user=?', (user,)).fetchall():
                    if tags == 'all' or tag in tags.split(','):
                        con.execute('DELETE FROM models WHERE user=? AND tags=?', (user, tags))
    finally:
        con.close()


# -----------------------------------------------------------------------------
# metrics.sqlite: how long pages and background jobs take, for /stats. no user, ip or query,
# only the kind of page; a week is kept. safe to delete

METRICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS timings (
    t REAL NOT NULL,          -- the minute these requests were counted in (one row per worker)
    name TEXT NOT NULL,       -- the kind of page, see aslite/metrics.py
    n INTEGER NOT NULL,
    total REAL NOT NULL,      -- seconds, summed
    hist TEXT NOT NULL        -- json {bucket: count}, see metrics.BUCKET
);
CREATE INDEX IF NOT EXISTS timings_t ON timings(t);
CREATE TABLE IF NOT EXISTS jobs (
    t REAL NOT NULL,          -- when it finished
    name TEXT NOT NULL,       -- compute, embed, arxiv
    seconds REAL NOT NULL,
    info TEXT NOT NULL        -- json: papers, peak memory, ...
);
CREATE INDEX IF NOT EXISTS jobs_t ON jobs(t);
"""
METRICS_KEEP = 7 * 86400


def _metrics_db():
    return connect(METRICS_DB_FILE, METRICS_SCHEMA)


def put_timings(rows):
    """rows: (minute, name, n, total seconds, {bucket: count})"""
    con = _metrics_db()
    try:
        with con:
            con.executemany(
                'INSERT INTO timings (t, name, n, total, hist) VALUES (?, ?, ?, ?, ?)',
                ((t, name, n, total, json.dumps(hist)) for t, name, n, total, hist in rows),
            )
            con.execute('DELETE FROM timings WHERE t < ?', (time.time() - METRICS_KEEP,))
    finally:
        con.close()


def get_timings(since):
    con = _metrics_db()
    try:
        q = 'SELECT t, name, n, total, hist FROM timings WHERE t >= ?'
        return [(t, name, n, total, json.loads(h)) for t, name, n, total, h in con.execute(q, (since,))]
    finally:
        con.close()


def put_job(name, seconds, **info):
    """a background job finished (compute.py, an embedding session, the arXiv fetch). never raises"""
    try:
        con = _metrics_db()
        try:
            with con:
                con.execute(
                    'INSERT INTO jobs (t, name, seconds, info) VALUES (?, ?, ?, ?)',
                    (time.time(), name, seconds, json.dumps(info)),
                )
                con.execute('DELETE FROM jobs WHERE t < ?', (time.time() - METRICS_KEEP,))
        finally:
            con.close()
    except Exception as e:  # only statistics
        print("recording job %s failed: %s" % (name, e))


def get_jobs(since):
    """[(finished, name, seconds, info)], oldest first"""
    con = _metrics_db()
    try:
        q = 'SELECT t, name, seconds, info FROM jobs WHERE t >= ? ORDER BY t'
        return [(t, name, sec, json.loads(i)) for t, name, sec, i in con.execute(q, (since,))]
    finally:
        con.close()


# -----------------------------------------------------------------------------
# refs.sqlite: the references of papers (aslite/refs.py, enrich_daemon.py), and the titles of
# every arXiv paper to match them by (import_snapshot.py --titles). no user data in it

REFS_SCHEMA = """
CREATE TABLE IF NOT EXISTS refs (
    citing TEXT NOT NULL,     -- the paper whose bibliography this is
    pos INTEGER NOT NULL,     -- its place in that bibliography
    cited TEXT,               -- the arXiv id it points to, NULL if none was found
    title TEXT NOT NULL,      -- its title as the bibliography gives it (best guess)
    doi TEXT,
    raw TEXT NOT NULL,        -- the whole reference, to match it again later
    PRIMARY KEY (citing, pos)
);
CREATE INDEX IF NOT EXISTS refs_cited ON refs(cited);
CREATE TABLE IF NOT EXISTS fetched (
    pid TEXT PRIMARY KEY,     -- a paper enrich_daemon.py has processed
    status TEXT NOT NULL,     -- ok (done, whatever it found) | error (tried again later)
    t REAL NOT NULL,          -- when it was last tried
    tries INTEGER NOT NULL,
    n INTEGER NOT NULL,       -- references found
    refs_from TEXT NOT NULL DEFAULT '',  -- html | pdf | '' (none could be read)
    figure INTEGER NOT NULL DEFAULT 0,   -- thumbnails made (only for open licenses)
    page INTEGER NOT NULL DEFAULT 0,
    license TEXT NOT NULL DEFAULT ''     -- thumbs.license_code(): what the thumbnails may be shown under
);
CREATE TABLE IF NOT EXISTS licenses (
    id TEXT PRIMARY KEY,      -- an arXiv paper of our categories (import_snapshot.py --titles)
    license TEXT NOT NULL     -- thumbs.license_code()
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS titles (
    h INTEGER PRIMARY KEY,    -- refs.title_hash() of a normalized title
    id TEXT NOT NULL          -- the arXiv paper with that title (the oldest, if several)
);
"""


_refs_migrated = set()


class RefsDB:
    def __init__(self, path=None):
        path = path or REFS_DB_FILE
        self.con = connect(path, REFS_SCHEMA)
        key = os.path.abspath(path)
        if key not in _refs_migrated:
            self._migrate()
            _refs_migrated.add(key)

    def _migrate(self):
        """
        refs.sqlite from before the thumbnails (refs_daemon.py, statuses ok/missing, no
        refs_from/figure/page/license): its fetched table is replaced, so those papers are
        processed again and get thumbnails and the PDF fallback. their references and the
        titles table stay. under the write lock, so the updater and web workers can't both do it
        """
        if 'license' in {r[1] for r in self.con.execute('PRAGMA table_info(fetched)')}:
            return
        self.con.execute('BEGIN IMMEDIATE')
        try:
            if 'license' not in {r[1] for r in self.con.execute('PRAGMA table_info(fetched)')}:
                self.con.execute('DROP TABLE fetched')
                for stmt in REFS_SCHEMA.split(';'):  # creates it anew (the others exist)
                    if stmt.strip():
                        self.con.execute(stmt)
            self.con.execute('COMMIT')
        except BaseException:
            if self.con.in_transaction:
                self.con.execute('ROLLBACK')
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.con.close()

    def lookup_title(self, h):
        row = self.con.execute('SELECT id FROM titles WHERE h = ?', (h,)).fetchone()
        return row[0] if row else None

    def put_titles(self, pairs):
        """(title hash, arXiv id) pairs; the first id stored for a title stays"""
        with self.con:
            self.con.executemany('INSERT OR IGNORE INTO titles (h, id) VALUES (?, ?)', pairs)

    def n_titles(self):
        return self.con.execute('SELECT count(*) FROM titles').fetchone()[0]

    def put_licenses(self, pairs):
        """(arXiv id, license code) pairs"""
        with self.con:
            self.con.executemany('INSERT OR REPLACE INTO licenses (id, license) VALUES (?, ?)', pairs)

    def license_of(self, pid):
        """the license code of a paper, or None if it isn't known (not in the snapshot)"""
        row = self.con.execute('SELECT license FROM licenses WHERE id = ?', (pid,)).fetchone()
        return row[0] if row else None

    def thumb_license(self, pid):
        """the license code the paper's thumbnails were made under, '' if none"""
        row = self.con.execute('SELECT license FROM fetched WHERE pid = ?', (pid,)).fetchone()
        return row[0] if row else ''

    def put_refs(self, citing, status, rows=(), refs_from='', figure=False, page=False, license=''):
        """
        what processing citing found: rows are its references (pos, cited, title, doi, raw);
        status 'error' keeps whatever an earlier run stored and counts a try
        """
        with self.con:
            if status == 'ok':
                self.con.execute('DELETE FROM refs WHERE citing = ?', (citing,))
                self.con.executemany(
                    'INSERT INTO refs (citing, pos, cited, title, doi, raw) VALUES (?, ?, ?, ?, ?, ?)',
                    ((citing, *r) for r in rows),
                )
            self.con.execute(
                'INSERT INTO fetched (pid, status, t, tries, n, refs_from, figure, page, license) '
                'VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?) ON CONFLICT(pid) DO UPDATE SET status = excluded.status, '
                't = excluded.t, tries = fetched.tries + 1, n = excluded.n, refs_from = excluded.refs_from, '
                'figure = excluded.figure, page = excluded.page, license = excluded.license',
                (citing, status, time.time(), len(rows), refs_from, int(figure), int(page), license),
            )

    def fetched(self):
        """{pid: (status, last try, tries)}"""
        return {p: (s, t, n) for p, s, t, n in self.con.execute('SELECT pid, status, t, tries FROM fetched')}

    def without_big_thumbs(self):
        """the papers with thumbnails from before there were big ones (for the enlarged view)"""
        rows = self.con.execute("SELECT pid, figure, page FROM fetched WHERE status = 'ok' AND (figure OR page)")
        return {
            p
            for p, figure, page in rows
            if any(
                made and not os.path.isfile(_thumb_file(p, k, True)) for k, made in (('figure', figure), ('page', page))
            )
        }

    def unmatched(self):
        """[(citing, pos, title, raw)] of the references no paper was found for"""
        return self.con.execute('SELECT citing, pos, title, raw FROM refs WHERE cited IS NULL').fetchall()

    def set_cited(self, rows):
        """(cited, citing, pos) for references matched later"""
        with self.con:
            self.con.executemany('UPDATE refs SET cited = ? WHERE citing = ? AND pos = ?', rows)

    def cited_by(self, pids):
        """{cited arXiv id: set of the pids citing it}, and the papers among pids whose references are known"""
        pids = list(pids)
        cited, known = {}, set()
        for i in range(0, len(pids), 500):
            chunk = pids[i : i + 500]
            marks = ','.join('?' * len(chunk))
            q = f"SELECT pid FROM fetched WHERE status = 'ok' AND refs_from != '' AND pid IN ({marks})"
            known |= {r[0] for r in self.con.execute(q, chunk)}
            q = f'SELECT cited, citing FROM refs WHERE cited IS NOT NULL AND citing IN ({marks})'
            for c, p in self.con.execute(q, chunk):
                cited.setdefault(c, set()).add(p)
        return cited, known

    def titles_of_cited(self, cited):
        """{arXiv id: a title some bibliography gives it}, for cited papers that aren't in papers.sqlite"""
        cited = list(cited)
        out = {}
        for i in range(0, len(cited), 500):
            chunk = cited[i : i + 500]
            q = f"SELECT cited, title FROM refs WHERE title != '' AND cited IN ({','.join('?' * len(chunk))})"
            for c, t in self.con.execute(q, chunk):
                out.setdefault(c, t)
        return out

    def counts(self):
        """for /stats: papers done and failed, where their references came from, thumbnails made,
        and references stored and matched"""
        row = self.con.execute(
            "SELECT count(*) FILTER (WHERE status = 'ok'), count(*) FILTER (WHERE status = 'error'), "
            "count(*) FILTER (WHERE refs_from = 'html'), count(*) FILTER (WHERE refs_from = 'pdf'), "
            'coalesce(sum(figure), 0), coalesce(sum(page), 0) FROM fetched'
        ).fetchone()
        c = dict(zip(('done', 'error', 'html', 'pdf', 'figure', 'page'), row, strict=True))
        c['refs'], c['matched'] = self.con.execute('SELECT count(*), count(cited) FROM refs').fetchone()
        return c


def get_refs_db(path=None):
    return RefsDB(path)


def _thumb_file(pid, kind, big=False):
    big = '.big' if big else ''
    return os.path.join(THUMBS_DIR, f"{pid.replace('/', '_')}.{kind}{big}.jpg")  # old ids: cs/0701001


def save_thumb(pid, kind, data, big=False):
    """a thumbnail (JPEG bytes) of kind 'figure' or 'page', written atomically; big: the enlarged view's"""
    os.makedirs(THUMBS_DIR, exist_ok=True)
    path = _thumb_file(pid, kind, big)
    with open(path + '.tmp', 'wb') as f:
        f.write(data)
    os.replace(path + '.tmp', path)


def thumb_file(pid, kinds, big=False):
    """
    (path, kind) of the first of these kinds this paper has a thumbnail of, else (None, None).
    big: the enlarged one, falling back to the card's (thumbnails made before there were big ones)
    """
    for kind in kinds:
        path = _thumb_file(pid, kind, True) if big else None
        if path is None or not os.path.isfile(path):
            path = _thumb_file(pid, kind)
        if os.path.isfile(path):
            return path, kind
    return None, None


def all_digest_pids():
    """the papers in everyone's latest digest"""
    out = set()
    for (sections,) in _query('SELECT sections FROM digests'):
        for s in json.loads(sections):
            out.update(p[0] if isinstance(p, list) else p for p in s['papers'])
    return out
