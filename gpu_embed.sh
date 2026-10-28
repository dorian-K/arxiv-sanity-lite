#!/usr/bin/env bash
# Computes the embeddings for a corpus on another machine with a GPU, then copies them back.
#
#   ./gpu_embed.sh user@gpuhost [model] [data dir]
#   ./gpu_embed.sh me@h100 st:google/embeddinggemma-2 data/5y
#
# Copies the code (git ls-files, so no data or venv) and <data dir>/papers.sqlite over ssh,
# runs compute.py there with the CUDA build of torch (uv extra embed-gpu), and copies
# back embeddings.sqlite (the per-paper cache, so later CPU runs only embed new papers) and
# features/. The remote needs ssh access, rsync and uv; nothing is installed system-wide.
# User data (users.sqlite) never leaves this machine.
set -euo pipefail

host=${1:?usage: $0 user@host [model] [data dir]}
model=${2:-st:google/embeddinggemma-2}
data=${3:-data}
remote=arxiv-sanity-lite   # directory in the remote home
batch=${EMBED_BATCH:-512}

cd "$(dirname "$0")"
test -f "$data/papers.sqlite" || { echo "no $data/papers.sqlite (run migrate_db.py on old data)"; exit 1; }

# consistent copies, even while the web server or updater has the databases open
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
snap() { uv run --no-sync python -c "import sys; from aslite.db import snapshot; snapshot(sys.argv[1], sys.argv[2])" "$1" "$2"; }
snap "$data/papers.sqlite" "$out/papers.sqlite"
# reuse whatever is already embedded, so only missing papers are computed
if [ -f "$data/embeddings.sqlite" ]; then snap "$data/embeddings.sqlite" "$out/embeddings.sqlite"; fi

echo "== copying code and $data to $host:$remote"
ssh "$host" "mkdir -p $remote/$data"
git ls-files -z | rsync -az --from0 --files-from=- ./ "$host:$remote/"
rsync -az --info=progress2 "$out/" "$host:$remote/$data/"

echo "== embedding on $host with $model"
ssh -t "$host" "cd $remote && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && \
    uv sync --extra embed-gpu --no-dev && \
    ARXIV_SANITY_DATA=$data uv run --no-sync python compute.py --embed_model '$model' --embed_batch $batch"

echo "== copying results back"
# compute.py has exited, so the remote databases are checkpointed and safe to copy
rsync -az --info=progress2 "$host:$remote/$data/embeddings.sqlite" "$out/"
if [ -f "$data/embeddings.sqlite" ]; then snap "$data/embeddings.sqlite" "$data/embeddings.sqlite.bak"; fi
rm -f "$data/embeddings.sqlite-wal" "$data/embeddings.sqlite-shm"
mv "$out/embeddings.sqlite" "$data/embeddings.sqlite"
# features/<version>/ plus the current symlink; the server picks up the new version by itself
rsync -az --info=progress2 --delete-after "$host:$remote/$data/features/" "$data/features/"  # old version goes last
echo "done: $data/features and $data/embeddings.sqlite updated (previous cache kept as .bak)"
