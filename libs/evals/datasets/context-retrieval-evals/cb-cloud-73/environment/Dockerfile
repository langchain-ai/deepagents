FROM python:3.12-slim

# Pre-install curl at build time (the build phase has network) so the
# in-sandbox agent's runtime bootstrap skips apt; runtime egress is then
# all-HTTPS via the task's network allowlist.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY files/ /app/files/
