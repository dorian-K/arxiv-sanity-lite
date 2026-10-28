# One image for both compose services; EXTRAS picks the optional dependencies:
#   web:     --extra serve  (gunicorn)
#   updater: --extra embed  (sentence-transformers with CPU torch, for --embed_model)
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/app/.venv/bin:$PATH \
    HF_HOME=/cache/huggingface \
    HOME=/tmp

WORKDIR /app
ARG EXTRAS="--extra serve"
# dependencies first, so code changes don't reinstall them
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev $EXTRAS

COPY . .
# the containers run as the host user (see docker-compose.yml), so data/ and the
# model cache must be writable by anyone
RUN mkdir -p /app/data /cache && chmod 777 /app/data /cache

EXPOSE 8000
# access log without client ips, user agents or query strings (search terms, the tokens of
# confirm/unsubscribe links): data minimization, see templates/privacy.html
CMD ["sh", "-c", "exec gunicorn serve:app --bind 0.0.0.0:8000 --workers ${WEB_WORKERS:-2} --threads ${WEB_THREADS:-4} --timeout 120 --access-logfile - --access-logformat '%(t)s \"%(m)s %(U)s\" %(s)s %(b)s %(L)ss'"]
