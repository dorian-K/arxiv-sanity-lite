"""
Imports years of papers from the arXiv metadata snapshot on Hugging Face
(librarian-bots/arxiv-metadata-snapshot: a weekly CC0 mirror of the Kaggle dataset
Cornell-University/arxiv, all ~3.2M papers in 11 parquet files, 2.7GB). Nothing is
requested from arXiv itself. Run arxiv_daemon.py afterwards for the last few days the
snapshot doesn't have yet.

    uv sync --extra bulk
    ARXIV_SANITY_DATA=data/5y uv run python import_snapshot.py --years 5
    ARXIV_SANITY_DATA=data/5y uv run python import_snapshot.py --years 5 --check data/papers.db

Verification, before anything is imported:
- every file is checked against the size and sha256 Hugging Face lists for the pinned
  revision, so a truncated or corrupted download is caught (and fetched again)
- the parquet row counts must add up to the row count in the dataset card
After the import:
- papers per month, so a gap in the snapshot shows up
- --check DB...: papers fetched directly from arXiv (e.g. by arxiv_daemon.py) that should
  be in the snapshot, and the share of them that is
"""

import os
import sys
import json
import time
import hashlib
import argparse
import urllib.request
from collections import Counter

import pyarrow.parquet as pq

from aslite.arxiv import make_paper, CATEGORIES
from aslite.db import get_papers_db, get_metas_db, CompressedSqliteDict, DATA_DIR

REPO = 'librarian-bots/arxiv-metadata-snapshot'
REVISION = 'c428226b689a0e17db2cd55265133b9eca15d97f'  # 2026-10-27; --revision main for the latest
API = 'https://huggingface.co/api/datasets/' + REPO
COLUMNS = ['id', 'title', 'abstract', 'categories', 'versions', 'authors_parsed', 'comments']


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def download(f, rev, cache):
    """downloads one file unless a verified copy exists; returns its path"""
    path = os.path.join(cache, os.path.basename(f['path']))
    want_size, want_sha = f['size'], f['lfs']['oid']
    if os.path.isfile(path) and os.path.getsize(path) == want_size and sha256(path) == want_sha:
        print(f"  ok (cached)  {f['path']}")
        return path
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{rev}/{f['path']}"
    tmp = path + '.part'
    h, done, t0 = hashlib.sha256(), 0, time.time()
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, 'wb') as out:
        for chunk in iter(lambda: r.read(1 << 20), b''):
            out.write(chunk)
            h.update(chunk)
            done += len(chunk)
    if done != want_size or h.hexdigest() != want_sha:
        os.remove(tmp)
        sys.exit(f"{f['path']}: got {done} bytes / sha256 {h.hexdigest()[:12]}, expected {want_size} / {want_sha[:12]}")
    os.rename(tmp, path)
    print(f"  ok           {f['path']} ({done / 1e6:.0f}MB, {done / 1e6 / (time.time() - t0):.0f}MB/s)")
    return path


def author_names(parsed):
    """authors_parsed rows are [last, first, suffix]"""
    return [' '.join(x for x in (first, last, suffix) if x) for last, first, suffix, *_ in parsed]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='import papers from the arXiv metadata snapshot on Hugging Face')
    parser.add_argument('--years', type=float, default=5, help='papers first submitted in the last this many years')
    parser.add_argument('--revision', default=REVISION, help='dataset revision (commit sha, or main)')
    parser.add_argument('--cache', default='data/arxiv-snapshot', help='where the parquet files are kept')
    parser.add_argument(
        '--check', nargs='*', default=[], help='papers.db files fetched from arXiv to check coverage against'
    )
    args = parser.parse_args()

    info = get_json(f'{API}/revision/{args.revision}')
    rev = info['sha']
    files = [f for f in get_json(f'{API}/tree/{rev}/data') if f['path'].endswith('.parquet')]
    expected_rows = info['cardData']['dataset_info']['splits'][0]['num_examples']
    cache = os.path.join(args.cache, rev[:12])
    os.makedirs(cache, exist_ok=True)
    print(
        f"{REPO}@{rev[:12]} ({info['lastModified'][:10]}): {len(files)} files, "
        f"{sum(f['size'] for f in files) / 1e9:.1f}GB, {expected_rows} rows"
    )

    # 1. files: size + sha256
    paths = [download(f, rev, cache) for f in sorted(files, key=lambda f: f['path'])]
    # 2. rows
    rows = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
    if rows != expected_rows:
        sys.exit(f"row count {rows} != {expected_rows} from the dataset card")
    print(f"verified: all sha256 match, {rows} rows as listed")

    # import
    since = time.time() - args.years * 365.25 * 86400
    pdb = get_papers_db(flag='c', autocommit=False)
    mdb = get_metas_db(flag='c', autocommit=False)
    months, kept_ids, newest = Counter(), set(), 0.0
    n_new = 0
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=20000, columns=COLUMNS):
            papers = []
            for r in batch.to_pylist():
                if not CATEGORIES & set(r['categories'].split()):
                    continue
                made = make_paper(
                    r['id'],
                    [v['created'] for v in r['versions']],
                    r['title'],
                    r['abstract'],
                    author_names(r['authors_parsed']),
                    r['categories'],
                    r['comments'],
                )
                if made is None or made[1] < since:
                    continue
                p, first = made
                kept_ids.add(p['_id'])
                months[time.strftime('%Y-%m', time.gmtime(first))] += 1
                newest = max(newest, first)
                if p['_id'] not in pdb or pdb[p['_id']]['_time'] < p['_time']:
                    papers.append(p)
            # both tables live in one sqlite file, so commit one before writing the other
            for p in papers:
                pdb[p['_id']] = p
            pdb.commit()
            for p in papers:
                mdb[p['_id']] = {'_time': p['_time']}
            mdb.commit()
            n_new += len(papers)
        print(f"  imported {os.path.basename(path)}: {len(kept_ids)} papers in window so far")
    print(
        f"{len(kept_ids)} papers first submitted since {time.strftime('%Y-%m-%d', time.gmtime(since))}, "
        f"{n_new} new or updated in {DATA_DIR}/papers.db (now {len(pdb)}); newest submission "
        f"{time.strftime('%Y-%m-%d', time.gmtime(newest))}"
    )
    pdb.close()
    mdb.close()

    # 3. papers per month: a hole in the snapshot would show up as a dip
    print("\npapers per month (first submission):")
    for m in sorted(months):
        print(f"  {m} {months[m]:>6} {'#' * (months[m] // 400)}")

    # 4. coverage against papers fetched straight from arXiv
    for ref in args.check:
        should, have = 0, 0
        with CompressedSqliteDict(ref, tablename='papers', flag='r') as rdb:
            for pid, p in rdb.items():
                first = time.mktime(time.strptime(p['published'], '%Y-%m-%dT%H:%M:%SZ'))
                # in the window, in our categories, and old enough to be in the snapshot
                if since <= first <= newest - 86400 and CATEGORIES & {t['term'] for t in p['tags']}:
                    should += 1
                    have += pid in kept_ids
        print(f"\ncoverage vs {ref}: {have}/{should} papers found ({have / max(should, 1):.2%})")
