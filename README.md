
# Arxiv Resanity

**Try it: [sanity.dorianko.ch](https://sanity.dorianko.ch)**

A fork of Andrej Karpathy's [arxiv-sanity-lite](https://github.com/karpathy/arxiv-sanity-lite). It adds:
- better recommendations from dense embeddings, tuned on real user ratings (results below);
- downvotes ("not for this tag");
- a weekly email digest with one-click ratings;
- years of papers from a bulk import, plus adding any paper by link;
- a "Cited" tab with the papers your saved ones cite, and figure thumbnails;
- password accounts with read-only sharing;
- a docker deployment that respects EU privacy law.

Arxiv Resanity keeps up with arXiv for you. Save papers you like into tags, one per topic you follow, and it ranks every new paper against each tag with a small model trained on the spot (tfidf plus dense embeddings). 👎 on a paper teaches a tag what it isn't about. Once a week it can email a digest of the best new papers per tag, rated with one click.

![Screenshot](screenshot.jpg)

## Run it locally

```bash
uv sync                                    # pass every --extra you need: embed, enrich, bulk
uv run python arxiv_daemon.py --num 2000   # fetch new papers from the arXiv API
uv run python compute.py                   # features (add --embed_model st:google/embeddinggemma-2)
uv run flask --app serve run
uv run pytest                              # tests on a synthetic corpus, no network
```

Everything lives in `data/`. Rerun the first two steps from cron to keep up.

## Serve it publicly

```bash
cp .env.example .env                       # every setting is explained there
docker network create --ipv6=false arxiv-resanity
docker compose up -d --build
```

- **arxiv_resanity_web** is gunicorn on `127.0.0.1:8000`, meant to sit behind an https proxy. The proxy must send `Host`, `X-Forwarded-For`, `X-Forwarded-Proto` and `X-Forwarded-Host`.
- **arxiv_resanity_updater** fetches new papers and recomputes features daily, embeds papers slowly in the background, and backs up `users.sqlite` to `data/backups/`. Accounts and tags are the only data that can't be rebuilt.
- **arxiv_resanity_mail** is an optional outgoing-only Postfix with DKIM (`COMPOSE_PROFILES=mail`). Alternatively, send through any SMTP server (`SMTP_*`). Your provider has to allow outgoing port 25, and the container prints the DNS records to set on its first start.

With `ALLOW_SIGNUP=0`, create accounts with `docker compose exec arxiv_resanity_web flask --app serve set-password <name>`.

Five years of papers (~460k) runs in 2–3 GB of RAM. The daily rebuild needs ~2.2 GB more for a few minutes, and the embedder ~2 GB while it works.

**Outward-facing features are off until you turn them on.**
- **Digest:** set `DIGEST_ENABLED=1` once a test mail passes SPF and DKIM. To try it without sending: `send_emails.py --dry-run 1 -u <user> --save d.html`.
- **References and thumbnails:** `ENRICH_ENABLED=1`. One request to arxiv.org every 20 s, for saved and digest papers only.

**EU law** (GDPR, ePrivacy, German imprint duty), handled in code:
- double opt-in for email addresses;
- data export and account deletion for users;
- no third-party resources (vendored and enforced by CSP);
- no IPs in logs;
- `/privacy` and `/imprint` filled from `OPERATOR_*`.

Left to you: a data processing agreement with your VPS provider, short nginx log retention, and having the privacy text read if the site grows beyond friends.

## How it works

- **Ranking:** each tag trains a logistic regression on its saved papers, its downvotes and 5000 random papers ([Scholar Inbox](https://arxiv.org/abs/2504.08385) recipe). Models are cached and refit in the background after a vote, so pages load at once. svm, lda and other feature sets are under "Advanced".
- **Embeddings:** `embed_daemon.py` fills them in at 10% duty in a worker that exits between sessions; papers without one are left out of recommendations. `gpu_embed.sh user@host` does a backlog on a GPU machine over ssh, in minutes instead of days.
- **Bulk import:** `import_snapshot.py --years 5` (extra `bulk`) imports from the [Hugging Face arXiv snapshot](https://huggingface.co/datasets/librarian-bots/arxiv-metadata-snapshot) and asks arXiv for nothing. `--titles` stores all 3.2M arXiv titles and licenses for reference matching.
- **Adding papers:** the add page (`/import`) takes arXiv, Hugging Face and alphaXiv links, DOIs and ids. Fetches are capped per user and day.
- **Search:** wants every word and fixes typos ("difusion modles").
- **Tag check:** ranks each saved paper with a model trained without it, and lists the ones that don't fit.
- **Cited tab:** built from the references in arXiv's HTML version of each saved paper, with the PDF as a fallback. They're matched to arXiv papers by id or title, and no other service is asked.
- **Thumbnails:** only for papers under a Creative Commons license, because German copyright has no fair use.
- **AI agents:** an MCP server at `/mcp` and the same tools as JSON at `/api` (`/llms.txt` describes them). `rank_like` ranks every paper by a few example papers, the best way to find related work. Searching and ranking need no account; a user's tags, reading list and notes need an access token they make on `/agents`, read-only unless they allow changes.
- **arXiv politeness:** API calls at most one per 3.5 s across all processes; arxiv.org pages one per 15 s or more, as its robots.txt asks.

## Does it rank better?

`eval.py` holds out liked papers and checks where they land. The table below is from 120 [Scholar Inbox](https://github.com/avg-dev/scholar_inbox_datasets) users the defaults were never tuned on, ranked within the 465k-paper production corpus with EmbeddingGemma 2:
- **auc_rand:** how often a held-out like ranks above a random rated paper;
- **auc_hard:** how often it ranks above a downvoted paper;
- **top1pct:** the share of held-out likes in the top 1%.

| feats:model | auc_rand | auc_hard | top1pct | 5 likes only: auc_hard |
|---|---|---|---|---|
| tfidf:svm (original) | .954 | .769 | .388 | .698 |
| concat:svm | .957 | **.808** | .414 | .728 |
| **concat:logreg (default)** | .956 | .803 | **.417** | .735 |
| emb:logreg | .951 | .797 | .402 | **.737** |
| concat:lda | .952 | .799 | .412 | .733 |

- **Embeddings help most where it matters:** separating likes from near misses (+.035 auc_hard), and new tags with few papers.
- **Model sizes:** EmbeddingGemma 2 beats bge-small by only ~.004, at 8× the CPU cost.
- **lda** trains in 5 ms instead of 0.3 s, but ranks slightly worse.

To reproduce: `uv run python eval.py --help`. Use `--pool rated`, or papers newer than every rating inflate the scores.

## License

MIT
