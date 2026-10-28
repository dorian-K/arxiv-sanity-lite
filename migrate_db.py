"""
One-time migration from the old sqlitedict storage (papers.db, dict.db, embeddings.db,
features.p) to the current one (papers.sqlite, users.sqlite, embeddings.sqlite, features/),
see aslite/db.py. The old files are only read, never changed; delete them once you're happy.

    uv run python migrate_db.py                            # data/
    ARXIV_SANITY_DATA=data/5y uv run python migrate_db.py  # another corpus

Safe to rerun: papers and embeddings are upserted, tags are merged, and accounts that
already have a password keep it.
"""

import os
import re
import sys
import zlib
import pickle
import sqlite3
import argparse

from aslite import db
from aslite.embed import ST_MODEL_CONFIGS


def old_table(path, table, compressed):
    """(key, value) of a sqlitedict table, decoded the way sqlitedict stored it"""
    con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        if not con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
            return
        for k, v in con.execute(f'SELECT key, value FROM "{table}"'):
            v = bytes(v)
            yield k, pickle.loads(zlib.decompress(v) if compressed else v)
    finally:
        con.close()


def old_tables(path):
    con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        return [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    finally:
        con.close()


def migrate_papers(path):
    n, batch = 0, []
    with db.get_papers_db() as pdb:
        for _, p in old_table(path, 'papers', compressed=True):
            batch.append(p)
            if len(batch) == 20000:
                n += pdb.upsert(batch)
                batch = []
                print(f"  {n} papers", end='\r')
        n += pdb.upsert(batch)
        print(f"papers: {n} written, {len(pdb)} in {db.PAPERS_DB_FILE}")


def migrate_users(path):
    tags = dict(old_table(path, 'tags', compressed=True))
    neg_tags = dict(old_table(path, 'neg_tags', compressed=True))
    emails = dict(old_table(path, 'email', compressed=False))
    last_active = dict(old_table(path, 'last_active', compressed=False))
    pw = dict(old_table(path, 'users', compressed=False))
    names = set(tags) | set(neg_tags) | set(emails) | set(last_active) | set(pw)
    con = db.users_db()
    with con:
        for name in names:
            # pw_hash stays NULL for accounts from before passwords: the first login claims them
            con.execute(
                "INSERT INTO users (name, pw_hash, email, last_active) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET pw_hash = coalesce(users.pw_hash, excluded.pw_hash), "
                "email = CASE WHEN users.email = '' THEN excluded.email ELSE users.email END",
                (name, pw.get(name), emails.get(name, '') or '', int(last_active.get(name, 0))),
            )
        n = 0
        # downvotes first, so a paper that ended up in both (shouldn't happen) counts as liked
        for neg, d in ((1, neg_tags), (0, tags)):
            for user, user_tags in d.items():
                for tag, pids in user_tags.items():
                    for pid in pids:
                        con.execute(
                            "INSERT INTO tags (user, tag, pid, neg) VALUES (?, ?, ?, ?) "
                            "ON CONFLICT DO UPDATE SET neg = excluded.neg",
                            (user, tag, pid, neg),
                        )
                        n += 1
    con.close()
    print(f"users: {len(names)} accounts ({len(pw)} with a password), {n} likes and downvotes")


def migrate_embeddings(path, models):
    # old table names were the model string with every non-alphanumeric run replaced by _
    by_table = {'emb_' + re.sub(r'[^A-Za-z0-9]+', '_', m): m for m in models}
    for table in old_tables(path):
        model = by_table.get(table)
        if model is None:
            print(f"embeddings: skipping table {table}: unknown model, pass it with --model")
            continue
        pids, vecs = [], []
        with db.EmbeddingCache(model) as cache:
            for pid, v in old_table(path, table, compressed=False):
                pids.append(pid)
                vecs.append(v)
                if len(pids) == 20000:
                    cache.put(pids, vecs)
                    pids, vecs = [], []
            cache.put(pids, vecs)
            print(f"embeddings: {len(cache.pids())} for {model}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='migrate the old sqlitedict storage')
    parser.add_argument('--model', action='append', default=[], help='embedding model strings to look for')
    args = parser.parse_args()

    d = db.DATA_DIR
    old = {k: os.path.join(d, k) for k in ('papers.db', 'dict.db', 'embeddings.db', 'features.p')}
    if not any(os.path.isfile(p) for p in old.values()):
        sys.exit(f"nothing to migrate in {d}/")

    models = set(args.model) | {'st:' + m for m in ST_MODEL_CONFIGS} | {'st:BAAI/bge-small-en-v1.5'}
    features = None
    if os.path.isfile(old['features.p']):
        features = db.load_features(old['features.p'])
        if features.get('emb_model'):
            models.add(features['emb_model'])

    if os.path.isfile(old['papers.db']):
        migrate_papers(old['papers.db'])
    if os.path.isfile(old['dict.db']):
        migrate_users(old['dict.db'])
    if os.path.isfile(old['embeddings.db']):
        migrate_embeddings(old['embeddings.db'], models)
    if features is not None and not db.features_exist():
        db.save_features({k: v for k, v in features.items() if k != 'ptoi'})
        print(f"features: {len(features['pids'])} papers to {db.FEATURES_DIR}/")

    print(
        f"\ndone. once everything works, the old files can go: rm {' '.join(p for p in old.values() if os.path.isfile(p))}"
    )
