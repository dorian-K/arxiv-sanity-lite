
# arxiv-sanity-lite

A much lighter-weight arxiv-sanity from-scratch re-write. Periodically polls arxiv API for new papers. Then allows users to tag papers of interest, and recommends new papers for each tag based on SVMs over tfidf features of paper abstracts. Allows one to search, rank, sort, slice and dice these results in a pretty web UI. Lastly, arxiv-sanity-lite can send you daily emails with recommendations of new papers based on your tags. Curate your tags, track recent papers in your area, and don't miss out!

I am running a live version of this code on [arxiv-sanity-lite.com](https://arxiv-sanity-lite.com).

![Screenshot](screenshot.jpg)

#### To run

To run this locally I usually run the following script to update the database with any new papers. I typically schedule this via a periodic cron job:

```bash
#!/bin/bash

uv run python arxiv_daemon.py --num 2000

if [ $? -eq 0 ]; then
    echo "New papers detected! Running compute.py"
    uv run python compute.py
else
    echo "No new papers were added, skipping feature computation"
fi
```

You can see that updating the database is a matter of first downloading the new papers via the arxiv api using `arxiv_daemon.py`, and then running `compute.py` to compute the tfidf features of the papers. Finally to serve the flask server locally we'd run something like:

```bash
uv run flask --app serve run
```

All of the database will be stored inside the `data` directory. Finally, if you'd like to run your own instance on the interwebs I recommend simply running the above on a [Linode](https://www.linode.com), e.g. I am running this code currently on the smallest "Nanode 1 GB" instance indexing about 30K papers, which costs $5/month.

(Optional) Finally, if you'd like to send periodic emails to users about new papers, see the `send_emails.py` script. Install its dependency with `uv sync --extra email`. I run this script in a daily cron job.

#### Requirements

Dependencies are managed with [uv](https://docs.astral.sh/uv/). This creates `.venv` with the pinned versions from `uv.lock`:

```bash
uv sync                              # core + dev tools (pytest)
uv sync --extra embed                # + sentence-transformers for local embeddings
uv sync --extra email --extra thumbs # + send_emails.py / thumb_daemon.py deps
uv run pytest                        # smoke tests on a synthetic corpus, no network needed
```

`uv sync` with a different set of extras removes the extras you left out, so pass every one you need.

#### Embeddings, downvotes and other ranking models (optional)

By default nothing changes: papers are ranked with tfidf SVMs as above. On top of that you can:

**Add dense embeddings.** `compute.py --embed_model ...` also embeds every paper's title and abstract and stores the vectors next to the tfidf features. Embeddings are cached in `data/embeddings.db`, so the periodic run only embeds the new papers. Two ways to get them:

```bash
# locally, with sentence-transformers (uv sync --extra embed)
uv run python compute.py --embed_model st:BAAI/bge-small-en-v1.5

# or from any OpenAI-compatible embeddings endpoint, e.g. a local Ollama server,
# which keeps torch off a small server entirely
export EMBED_API_BASE=http://localhost:11434/v1
uv run python compute.py --embed_model api:embeddinggemma
```

You can also set `EMBED_MODEL` in the environment instead of passing `--embed_model`. Embeddings are reduced to 256 dimensions with PCA by default (`--emb_dims`, 0 to keep all).

**Pick features and model per request.** The front page gets two new options, also usable as URL parameters and as `send_emails.py --feats/--model`:

- `feats`: `tfidf`, `emb` (embeddings only), `concat` (both in one model), `fusion` (one model each, scores averaged) or `auto` (fusion if embeddings exist, else tfidf).
- `model`: `svm` (the original) or `logreg`, the recipe from [Scholar Inbox](https://arxiv.org/abs/2504.08385): logistic regression on your liked papers, your downvotes and 5000 random other papers, with tuned class weights.

The word list under "inspect" still shows the tfidf words, also for `concat` and `fusion`. With `feats=emb` there are no words to show.

**Downvote papers.** The new ✗ button on a paper asks for a tag and marks the paper as *not* what that tag is about. Downvotes are used as explicit negatives when ranking that tag, hidden by `skip_have`, and shown as crossed-out tags on the paper (click one to undo). Liking a paper for a tag removes its downvote for that tag and vice versa.

**Fetch specific papers.** `uv run python fetch_papers.py <arxiv ids>` (or `--file ids.txt`) adds papers by id, e.g. older papers you want to tag. Run `compute.py` afterwards.

**Compare the variants.** `eval.py` holds out some of the liked papers of each tag and checks where they land in the ranking, for every feature set and model. Run it on your own tags with `uv run python eval.py run --source tags`, or on the public [Scholar Inbox ratings](https://github.com/avg-dev/scholar_inbox_datasets) (thumbs up/down from thousands of researchers):

```bash
uv run python eval.py prepare-si --ratings path/to/rated_papers.csv --users 60
uv run python fetch_papers.py --file data/si_ids.txt
uv run python compute.py --embed_model st:BAAI/bge-small-en-v1.5
uv run python eval.py run --source si
```

The defaults in `aslite/rank.py` (`DEFAULT_C`, `DEFAULTS`) are starting points; tune them with `eval.py` once you have results.

#### Todos

- The metas table should not be a sqlitedict but a proper sqlite table, for efficiency
- Build a reverse index to support faster search, right now we iterate through the entire database

#### License

MIT
