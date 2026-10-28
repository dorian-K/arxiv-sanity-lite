# Outgoing-only mail server for the digest emails: Postfix delivers straight to the
# recipients' mail servers, OpenDKIM signs every message. Not reachable from outside:
# docker-compose.yml publishes no port, only the other services can hand it mail.
FROM debian:bookworm-slim
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        postfix opendkim opendkim-tools ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY entrypoint.sh /entrypoint.sh
EXPOSE 25
CMD ["sh", "/entrypoint.sh"]
