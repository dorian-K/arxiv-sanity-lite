# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working rules (from the user, keep these)

- Commit on your own after every substantial change, without waiting to be asked. Use a brief title and explain the reasons in the commit body. Don't push; the user reviews and pushes.
- Whenever the user gives guidance or corrects a mistake, add it to this section so it doesn't happen again, and commit that too.
- Git author is `Dorian Koch <do.ko1605.dk@gmail.com>`, set in the repo's local git config.
- Flag evaluation caveats yourself, as soon as you see them, and fix them before drawing conclusions. Don't sit on a concern you noticed. (Example: the arxiv_daemon background papers are newer than every Scholar Inbox rating, so eval scores against them are inflated. Use `eval.py run --pool rated`.)
- Never hammer arXiv; the user does not want to be banned. Pause ≥3s between API calls (harvest_papers.py uses 5s), honor 503 Retry-After, back off on errors, and don't re-download what is already in papers.db. For bulk data, use the Hugging Face snapshot (import_snapshot.py) instead of arXiv's APIs.
- Work and commit directly on `master`; don't create feature branches, even if a pasted prompt asks for one.

## Commands

The shell is zsh, which doesn't word-split an unquoted `$VAR`. A command stored in a variable has to be expanded as `${=VAR}`, and a list of ids should be looped over as an array (`for p in ${=P}`). A failed split often shows up only as an empty log.

Never gate a commit on `uv run pytest | tail -1 && git commit`: the pipe returns `tail`'s exit code, so a failing run still commits. Use `uv run pytest -q && git commit ...`, or `setopt pipefail` first.


Dependencies are managed with uv (`pyproject.toml` + `uv.lock`, Python 3.12+, `package = false`). Every script runs from the repo root, because paths such as `data/`, `static/thumb` and `secret_key.txt` are relative to it.

```bash
uv sync                                    # core + dev (pytest); --extra embed|email|thumbs for optional deps
uv sync --extra embed                      # sentence-transformers for compute.py --embed_model st:...
                                           # (uv sync drops extras you don't list, so pass every one you need)
uv run pytest                              # smoke tests, synthetic corpus in a temp dir, no network
uv run ruff check . && uv run ruff format . # lint (bug-finding rules) + format; run both before committing
uv run pytest tests/test_rank.py -k logreg # single test

uv run python arxiv_daemon.py --num 2000   # pull new papers from the arxiv API into data/papers.db
uv run python compute.py                   # rebuild data/features.p (tfidf); add --embed_model to also embed
uv run python fetch_papers.py <arxiv ids>  # add specific papers by id, then rerun compute.py
ARXIV_SANITY_DATA=data/5y uv run python import_snapshot.py --years 5 --check data/papers.db   # bulk import from the HF snapshot (extra bulk), verified; no arXiv load
ARXIV_SANITY_DATA=data/5y uv run python harvest_papers.py --years 5   # fallback: same via OAI-PMH, ~5h of polite requests
./gpu_embed.sh user@host st:google/embeddinggemma-2 data/5y   # embeddings on a GPU box over ssh
uv run flask --app serve run               # web UI (make fun); `make up` = daemon + compute
uv run python send_emails.py --dry-run 1 -u <user>   # check recommendation emails without sending
uv run python thumb_daemon.py              # paper thumbnails into static/thumb (needs ImageMagick `convert`)
uv run python eval.py run --source tags    # compare ranking variants on held-out liked papers
```

Tests (`tests/conftest.py`) chdir into a temp dir with 60 fake papers from two separable topics, then run the real `compute.py` there. Before every commit, run them. Because the tests tag papers before loading any page, they don't catch fresh-install crashes, so check those against a live `flask run` on an empty `data/dict.db`.

`eval.py` measures ranking quality. It compares every feature set × model combination by checking where held-out positives land in the ranking.

## Architecture

The pipeline runs in batch through cron jobs, and the web server only reads its results:

1. **`arxiv_daemon.py` / `fetch_papers.py`** write paper records to `data/papers.db`, using the `papers` table (zlib-compressed pickles) and the lighter `metas` table. `aslite/arxiv.py` wraps the arxiv API.
2. **`compute.py`** reads all papers and writes a single pickle, `data/features.p`, with the keys `pids`, `x` (sparse tfidf), `vocab` and `idf`. Embeddings add `x_emb` (dense, PCA'd to 256 dims and L2-normalized) and `emb_model`. `aslite/embed.py` produces those embeddings and caches them per paper in `data/embeddings.db`, with one table per model string, so a rerun only embeds new papers.
3. **`serve.py`** (Flask) and **`send_emails.py`** both rank through **`aslite/rank.py`**. Its `rank(features, pos_pids, neg_pids, feats, model, C)` trains a small linear model on the fly for each request or tag and returns sorted pids, scores and top tfidf words. Nothing is precomputed per user.
   - `feats`: `tfidf | emb | concat | fusion | auto`. Every option except `tfidf` needs `x_emb`, and `auto` falls back to tfidf when it is missing.
   - `model`: `svm` (the original: every untagged paper counts as a negative, up to a random 20k sample on big corpora, and downvotes get extra weight) or `logreg` (Scholar Inbox recipe: random negatives plus S/V class weights).
   - `DEFAULT_C` and `DEFAULTS` are untuned starting points, meant to be tuned with `eval.py`.

All file I/O and DB access goes through **`aslite/db.py`**. Scripts don't touch the filesystem directly, so add new storage there.
- Every store is a `sqlitedict` table. `data/dict.db` holds user state: `tags` and `neg_tags` (both `user -> {tag: set(pids)}`), `last_active` and `email`.
- sqlitedict tables only exist after their first write, and a read-only open of a missing table raises. `_flag()` in `db.py` therefore upgrades `'r'` to `'c'` when the file or table is missing. Route every new opener through it.
- `load_features()` caches the pickle in memory and reloads it only when the file's mtime changes. `save_features` writes atomically, so `compute.py` can run while the server is up.
- `rank.py` has its own `_concat_cache` keyed on `id(features)`.

User model: login is just a username stored in the Flask session, with no passwords. The session is signed with `secret_key.txt`, falling back to `devkey`. For a given user and tag, a paper is either liked (`/add`) or downvoted (`/addneg`), never both: each endpoint removes the paper from the other set.

Frontend: Jinja templates in `templates/` plus plain-JS rendering in `static/*.js`. `serve.py` passes data to the page through `context['gvars']`, which carries things like the paper list and the `has_emb` flag.
