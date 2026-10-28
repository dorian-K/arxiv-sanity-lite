#!/usr/bin/env bash
# Computes the embeddings for a corpus on another machine with a GPU, then copies them back.
#
#   ./gpu_embed.sh user@gpuhost [model] [data dir]
#   ./gpu_embed.sh me@h100 st:google/embeddinggemma-2 data/5y
#
# Copies the code (git ls-files, so no data or venv) and <data dir>/papers.db over ssh,
# runs compute.py there with the CUDA build of torch (uv extra embed-gpu), and copies
# back embeddings.db (the per-paper cache, so later CPU runs only embed new papers) and
# features.p. The remote needs ssh access, rsync and uv; nothing is installed system-wide.
set -euo pipefail

host=${1:?usage: $0 user@host [model] [data dir]}
model=${2:-st:google/embeddinggemma-2}
data=${3:-data}
remote=arxiv-sanity-lite   # directory in the remote home
batch=${EMBED_BATCH:-512}

cd "$(dirname "$0")"
test -f "$data/papers.db" || { echo "no $data/papers.db"; exit 1; }

echo "== copying code and $data/papers.db to $host:$remote"
ssh "$host" "mkdir -p $remote/$data"
git ls-files -z | rsync -az --from0 --files-from=- ./ "$host:$remote/"
rsync -az --info=progress2 "$data/papers.db" "$host:$remote/$data/"
# reuse whatever is already embedded, so only missing papers are computed
if [ -f "$data/embeddings.db" ]; then
    rsync -az --info=progress2 "$data/embeddings.db" "$host:$remote/$data/"
fi

echo "== embedding on $host with $model"
ssh -t "$host" "cd $remote && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && \
    uv sync --extra embed-gpu --no-dev && \
    ARXIV_SANITY_DATA=$data uv run --no-sync python compute.py --embed_model '$model' --embed_batch $batch"

echo "== copying results back"
for f in embeddings.db features.p; do
    test -f "$data/$f" && cp "$data/$f" "$data/$f.bak"
    rsync -az --info=progress2 "$host:$remote/$data/$f" "$data/"
done
echo "done: $data/features.p and $data/embeddings.db updated (previous versions kept as .bak)"
