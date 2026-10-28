# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working rules (from the user, keep these)

- Commit on your own after every substantial change, without waiting to be asked. Use a brief title and explain the reasons in the commit body. Don't push; the user reviews and pushes.
- Whenever the user gives guidance or corrects a mistake, add it to this section so it doesn't happen again, and commit that too.
- Git author is `Dorian Koch <do.ko1605.dk@gmail.com>`, set in the repo's local git config.
- Flag evaluation caveats yourself, as soon as you see them, and fix them before drawing conclusions. Don't sit on a concern you noticed. (Example: the arxiv_daemon background papers are newer than every Scholar Inbox rating, so eval scores against them are inflated. Use `eval.py run --pool rated`.)
- Never hammer arXiv; the user does not want to be banned. Pause ≥3s between API calls (harvest_papers.py uses 5s), honor 503 Retry-After, back off on errors, and don't re-download what is already in papers.sqlite. For bulk data, use the Hugging Face snapshot (import_snapshot.py) instead of arXiv's APIs.
- arXiv calls a user can trigger (`/import`) must go through `db.arxiv_slot()` (one call per 3.5 s across processes) and count against `FETCH_LIMIT_USER`/`FETCH_LIMIT_TOTAL` via the `fetches` table. Never fetch from a GET handler.
- Anything outward-facing (emails, pushes to other services) is off by default and never fires on first start or right after being enabled; the user turns it on after testing. (The digest once defaulted to on and would have mailed everyone before the DNS records existed.)
- Work and commit directly on `master`; don't create feature branches, even if a pasted prompt asks for one.

## Commands

The shell is zsh, which doesn't word-split an unquoted `$VAR`. A command stored in a variable has to be expanded as `${=VAR}`, and a list of ids should be looped over as an array (`for p in ${=P}`). A failed split often shows up only as an empty log.

Never gate a commit on `uv run pytest | tail -1 && git commit`: the pipe returns `tail`'s exit code, so a failing run still commits. Use `uv run pytest -q && git commit ...`, or `setopt pipefail` first.


Dependencies are managed with uv (`pyproject.toml` + `uv.lock`, Python 3.12+, `package = false`). Every script runs from the repo root, because paths such as `data/`, `static/thumb` and `secret_key.txt` are relative to it.

```bash
uv sync                                    # core + dev (pytest); --extra embed|serve|bulk|thumbs for optional deps
uv sync --extra embed                      # sentence-transformers for compute.py --embed_model st:...
                                           # (uv sync drops extras you don't list, so pass every one you need)
uv run pytest                              # smoke tests, synthetic corpus in a temp dir, no network
uv run ruff check . && uv run ruff format . # lint (bug-finding rules) + format; run both before committing
uv run pytest tests/test_rank.py -k logreg # single test

uv run python arxiv_daemon.py --num 2000   # pull new papers from the arxiv API into data/papers.sqlite
uv run python compute.py                   # rebuild data/features/ (tfidf); add --embed_model to also embed
uv run python fetch_papers.py <arxiv ids>  # add specific papers by id, then rerun compute.py
ARXIV_SANITY_DATA=data/5y uv run python import_snapshot.py --years 5 --check data/papers.sqlite   # bulk import from the HF snapshot (extra bulk), verified; no arXiv load
ARXIV_SANITY_DATA=data/5y uv run python harvest_papers.py --years 5   # fallback: same via OAI-PMH, ~5h of polite requests
./gpu_embed.sh user@host st:google/embeddinggemma-2 data/5y   # embeddings on a GPU box over ssh; GPU=MIG-<uuid> pins a device
uv run flask --app serve run               # web UI (make fun); `make up` = daemon + compute
docker compose up -d --build               # public deployment: gunicorn web + daily updater, behind an https proxy (.env.example)
uv run flask --app serve set-password <user>   # create/reset an account (needed when ALLOW_SIGNUP=0)
uv run python send_emails.py --dry-run 1 -u <user> --save d.html   # compose (and store) a digest without sending
uv run python thumb_daemon.py              # paper thumbnails into static/thumb (needs ImageMagick `convert`)
uv run python eval.py run --source tags    # compare ranking variants on held-out liked papers
```

Tests (`tests/conftest.py`) chdir into a temp dir with 60 fake papers from two separable topics, then run the real `compute.py` there. Before every commit, run them. Because the tests tag papers before loading any page, they don't catch fresh-install crashes, so check those against a live `flask run` on an empty data dir.

`eval.py` measures ranking quality. It compares every feature set × model combination by checking where held-out positives land in the ranking.

## Architecture

The pipeline runs in batch through cron jobs, and the web server only reads its results:

1. **`arxiv_daemon.py` / `fetch_papers.py`** (and the bulk importers) write papers to `data/papers.sqlite` with `PapersDB.upsert`, which keeps only the newest version. `aslite/arxiv.py` wraps the arxiv API.
2. **`compute.py`** reads all papers and writes `data/features/`, which loads as a dict with the keys `pids`, `x` (sparse tfidf), `vocab`, `idf` and `ptoi` (pid -> row). Embeddings add `x_emb` (dense, PCA'd to 256 dims and L2-normalized) and `emb_model`. `aslite/embed.py` produces those embeddings and caches them per paper and model in `data/embeddings.sqlite`, so a rerun only embeds new papers.
3. **`serve.py`** (Flask) and **`send_emails.py`** both rank through **`aslite/rank.py`**. Its `rank(features, pos_pids, neg_pids, feats, model, C)` trains a small linear model on the fly for each request or tag and returns sorted pids, scores and top tfidf words. Nothing is precomputed per user.
   - `feats`: `tfidf | emb | concat | fusion | auto`. Every option except `tfidf` needs `x_emb`, and `auto` falls back to tfidf when it is missing.
   - `model`: `svm` (the original: every untagged paper counts as a negative, up to a random 20k sample on big corpora, and downvotes get extra weight) or `logreg` (Scholar Inbox recipe: random negatives plus S/V class weights).
   - `DEFAULT_C` and `DEFAULTS` are untuned starting points, meant to be tuned with `eval.py`.

All file I/O and DB access goes through **`aslite/db.py`**. Scripts don't touch the filesystem directly, so add new storage there.
- Plain SQLite through the stdlib, every database in WAL mode, so the web server keeps reading while the updater writes. The schema is created on first connect (`connect()`). Copy a live database with `snapshot()`, never with `cp`.
- `papers.sqlite`: one row per paper. `PapersDB` returns the same paper dicts (`_id`, `_time`, `authors: [{name}]`, `tags: [{term}]`, ...) the code always used, and also has `get_many`, `times` and `search`. Search is an FTS5 index over the table itself, kept in sync by triggers (bm25, weights title > authors > abstract).
- `users.sqlite`: `users` (password hash, email, last_active), `tags` (one row per user/tag/paper, `neg` marks a downvote, so like and downvote are exclusive by construction), `fetches`, `digests` (the latest digest per user).
- Digest: `aslite/digest.py` computes it (one rank per tag, last N days, minus rated papers). `send_emails.py` stores it and emails it over SMTP; `/digest` shows the stored one with one-click votes. Email links come from `PUBLIC_URL`, never a hardcoded domain. `docker/mail/` is the optional outgoing Postfix + OpenDKIM container.
- `features/<version>/` holds `.npy` arrays that are memory-mapped (all workers share one copy in the page cache), and `features/current` is a symlink swapped atomically. `load_features()` caches per process until `current` changes. Never build a full concat or copy of `x` per process; `rank.py` stacks only the training rows and scores the blocks separately.
- `migrate_db.py` converts the old sqlitedict files (papers.db, dict.db, embeddings.db, features.p). It reads them raw, so sqlitedict is no longer a dependency.
- `load_features()` caches the pickle in memory and reloads it only when the file's mtime changes. `save_features` writes atomically, so `compute.py` can run while the server is up.
- serve.py caches every paper's time in memory per worker (`get_times`), and reloads it when papers.sqlite (or its WAL) changes. Never iterate all papers per request.

User model: accounts have werkzeug password hashes in `users.sqlite`. The first login with a new username signs up, unless `ALLOW_SIGNUP=0`. The Flask session is signed with `$SECRET_KEY`, else `secret_key.txt`, else `data/secret_key.txt`, which is generated on first run. Every endpoint that changes state is POST-only, and cross-site POSTs are rejected through SameSite=Lax cookies plus a `Sec-Fetch-Site` check. Keep it that way for new endpoints. Expensive endpoints call `rate_limited(kind)` (in-memory token buckets per user or ip, see `RATE_LIMITS`); add a kind for any new expensive path. `TRUST_PROXY` and `SECURE_COOKIES` are for running behind the https proxy. For a given user and tag, a paper is either liked (`/add`) or downvoted (`/addneg`), never both.

Frontend: Jinja templates in `templates/` plus plain-JS rendering in `static/*.js`. `serve.py` passes data to the page through `context['gvars']`, which carries things like the paper list and the `has_emb` flag.
