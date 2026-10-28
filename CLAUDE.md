# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

There is no test suite, linter or build step. Every script runs from the repo root, because paths such as `data/`, `static/thumb` and `secret_key.txt` are relative to it.

```bash
pip install -r requirements.txt            # core
pip install -r requirements-embed.txt      # only for local embeddings (compute.py --embed_model st:...)

python3 arxiv_daemon.py --num 2000         # pull new papers from the arxiv API into data/papers.db
python3 compute.py                         # rebuild data/features.p (tfidf); add --embed_model to also embed
python3 fetch_papers.py <arxiv ids>        # add specific papers by id, then rerun compute.py
export FLASK_APP=serve.py; flask run       # web UI (make fun); `make up` = daemon + compute
python3 send_emails.py --dry-run 1 -u <user>   # check recommendation emails without sending
python3 thumb_daemon.py                    # paper thumbnails into static/thumb
python3 eval.py run --source tags          # compare ranking variants on held-out liked papers
```

`eval.py` is the closest thing to a test for ranking changes. It compares every feature set × model combination by checking where held-out positives land in the ranking.

## Architecture

The pipeline runs in batch through cron jobs, and the web server only reads its results:

1. **`arxiv_daemon.py` / `fetch_papers.py`** write paper records to `data/papers.db`, using the `papers` table (zlib-compressed pickles) and the lighter `metas` table. `aslite/arxiv.py` wraps the arxiv API.
2. **`compute.py`** reads all papers and writes a single pickle, `data/features.p`, with the keys `pids`, `x` (sparse tfidf), `vocab` and `idf`. Embeddings add `x_emb` (dense, PCA'd to 256 dims and L2-normalized) and `emb_model`. `aslite/embed.py` produces those embeddings and caches them per paper in `data/embeddings.db`, with one table per model string, so a rerun only embeds new papers.
3. **`serve.py`** (Flask) and **`send_emails.py`** both rank through **`aslite/rank.py`**. Its `rank(features, pos_pids, neg_pids, feats, model, C)` trains a small linear model on the fly for each request or tag and returns sorted pids, scores and top tfidf words. Nothing is precomputed per user.
   - `feats`: `tfidf | emb | concat | fusion | auto`. Every option except `tfidf` needs `x_emb`, and `auto` falls back to tfidf when it is missing.
   - `model`: `svm` (the original: every untagged paper counts as a negative, and downvotes get extra weight) or `logreg` (Scholar Inbox recipe: random negatives plus S/V class weights).
   - `DEFAULT_C` and `DEFAULTS` are untuned starting points, meant to be tuned with `eval.py`.

All file I/O and DB access goes through **`aslite/db.py`**. Scripts don't touch the filesystem directly, so add new storage there.
- Every store is a `sqlitedict` table. `data/dict.db` holds user state: `tags` and `neg_tags` (both `user -> {tag: set(pids)}`), `last_active` and `email`.
- `get_neg_tags_db` always opens with `flag='c'` on purpose, because older installs don't have that table yet.
- `load_features()` caches the pickle in memory and reloads it only when the file's mtime changes. `save_features` writes atomically, so `compute.py` can run while the server is up.
- `rank.py` has its own `_concat_cache` keyed on `id(features)`.

User model: login is just a username stored in the Flask session, with no passwords. The session is signed with `secret_key.txt`, falling back to `devkey`. For a given user and tag, a paper is either liked (`/add`) or downvoted (`/addneg`), never both: each endpoint removes the paper from the other set.

Frontend: Jinja templates in `templates/` plus plain-JS rendering in `static/*.js`. `serve.py` passes data to the page through `context['gvars']`, which carries things like the paper list and the `has_emb` flag.
