"""
Dense embeddings of paper titles + abstracts.

Two backends, picked by the model string:

- "st:<huggingface model>"  runs locally with sentence-transformers, e.g.
      st:BAAI/bge-small-en-v1.5      (33M params, fast on CPU)
      st:google/embeddinggemma-300m
      st:google/embeddinggemma-2     (270M text-only, Oct 2026; ~1 paper/s on 6 CPU cores)
      st:Qwen/Qwen3-Embedding-0.6B   (better, but slow on a small CPU box)
- "api:<model name>"        calls any OpenAI-compatible /v1/embeddings endpoint,
      configured with the env vars EMBED_API_BASE and EMBED_API_KEY. This also
      covers a local Ollama server (EMBED_API_BASE=http://localhost:11434/v1),
      which keeps torch off a small server entirely.

Embeddings are cached per paper in data/embeddings.db, keyed by model name, so
the periodic compute.py run only embeds papers that are new since last time.
"""

import os
import json
import time
import urllib.request

import numpy as np

from aslite.db import get_embeddings_db


# per-model settings for sentence-transformers models that need more than the defaults:
#   text_format     how title + abstract are put together (some models are trained with prefixes)
#   config_kwargs   passed to the HF config, e.g. to skip loading unused encoders
#   max_seq_length  cap on input tokens; abstracts fit in 512 and long contexts cost CPU time
ST_MODEL_CONFIGS = {
    # multimodal, but text-only loading drops the vision/audio encoders (740M -> 270M params).
    # documents with a title are trained with this prefix, see the model card
    'google/embeddinggemma-2': dict(
        text_format='title: {title} | text: {summary}',
        config_kwargs={'vision_config': None, 'audio_config': None},
        max_seq_length=512,
    ),
}


def paper_text(p, text_format='{title}\n\n{summary}'):
    """ the text we embed for a paper: title and abstract, whitespace-normalized """
    title = ' '.join(p['title'].split())
    summary = ' '.join(p['summary'].split())
    return text_format.format(title=title, summary=summary)


class SentenceTransformerBackend:

    def __init__(self, name):
        import torch # lazy: torch is heavy
        from sentence_transformers import SentenceTransformer
        cfg = ST_MODEL_CONFIGS.get(name, {})
        self.text_format = cfg.get('text_format', '{title}\n\n{summary}')
        # some models default to bfloat16, which is many times slower on most CPUs
        dtype = torch.float32 if not torch.cuda.is_available() else 'auto'
        self.model = SentenceTransformer(name, trust_remote_code=False, model_kwargs={'dtype': dtype},
                                         config_kwargs=cfg.get('config_kwargs'))
        if 'max_seq_length' in cfg:
            self.model.max_seq_length = cfg['max_seq_length']

    def encode(self, texts):
        x = self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                              show_progress_bar=False, convert_to_numpy=True)
        return x.astype(np.float32)


class APIBackend:

    def __init__(self, name):
        self.name = name
        self.base = os.environ.get('EMBED_API_BASE', 'https://api.openai.com/v1').rstrip('/')
        self.key = os.environ.get('EMBED_API_KEY', '')
        self.text_format = '{title}\n\n{summary}'

    def encode(self, texts):
        body = json.dumps({'model': self.name, 'input': texts}).encode('utf-8')
        headers = {'Content-Type': 'application/json'}
        if self.key:
            headers['Authorization'] = 'Bearer ' + self.key
        req = urllib.request.Request(self.base + '/embeddings', data=body, headers=headers)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read())
                break
            except Exception as e:
                if attempt == 4:
                    raise
                print(f"embedding API error ({e}), retrying...")
                time.sleep(2 ** attempt)
        rows = sorted(data['data'], key=lambda d: d['index'])
        x = np.array([r['embedding'] for r in rows], dtype=np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x


def get_backend(model):
    kind, _, name = model.partition(':')
    if kind == 'st':
        return SentenceTransformerBackend(name)
    if kind == 'api':
        return APIBackend(name)
    raise ValueError(f"model must look like 'st:<hf model>' or 'api:<model>', got {model!r}")


def embed_papers(pids, pdb, model, batch_size=64, log_every=1000):
    """
    Returns an (len(pids), d) float32 matrix of unit-norm embeddings, aligned with pids.
    Only papers missing from the cache for this model are actually embedded.
    """
    cache = get_embeddings_db(model, flag='c', autocommit=False)
    try:
        missing = [pid for pid in pids if pid not in cache]
        print(f"embeddings for {model}: {len(pids) - len(missing)} cached, {len(missing)} to compute")
        if missing:
            backend = get_backend(model)
            texts = {pid: paper_text(pdb[pid], backend.text_format) for pid in missing}
            # batches get padded to their longest text, so group similar lengths together;
            # on CPU this saves about a third of the compute
            missing.sort(key=lambda pid: len(texts[pid]))
            t0 = time.time()
            for i in range(0, len(missing), batch_size):
                batch = missing[i:i + batch_size]
                x = backend.encode([texts[pid] for pid in batch])
                for pid, v in zip(batch, x):
                    cache[pid] = v.astype(np.float16) # half precision is plenty and halves the db
                cache.commit()
                done = i + len(batch)
                if done % log_every < batch_size or done == len(missing):
                    rate = done / max(time.time() - t0, 1e-6)
                    print(f"  embedded {done}/{len(missing)} ({rate:.1f} papers/s)")
        x = np.stack([cache[pid] for pid in pids]).astype(np.float32)
    finally:
        cache.close()
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x
