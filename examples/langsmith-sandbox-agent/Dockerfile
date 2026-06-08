FROM python:3.12-slim

# Bake the agent runtime into the box so it is ready the moment the sandbox boots.
# `truststore` lets the box's TLS clients (httpx, used by the Anthropic SDK) verify
# against the OS trust store, which the sandbox populates with its egress-proxy CA
# at boot. `pytest` is used to verify the agent's work.
RUN pip install --no-cache-dir "deepagents>=0.6" langchain-anthropic truststore pytest

WORKDIR /app
