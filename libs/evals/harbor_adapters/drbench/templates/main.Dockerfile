# The `main` compose service: where Harbor installs and runs the agent. It holds NO
# task data — the documents live in the `drbench` service and are reached over HTTP,
# which is the point of app mode. Keeping the corpus out of here also keeps
# `/drbench/task/env.json` (which labels each document insight-vs-distractor) out of
# the agent's reach.
FROM python:3.12-slim

# Installed at build time (the build phase has network) so the in-sandbox agent's
# runtime bootstrap skips apt. `curl` is the agent's entire transport to the app
# stack; `poppler-utils` provides pdftotext as a fallback alongside pypdf.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# The corpus is PDF/DOCX/XLSX/PPTX/JSONL, so every document the agent downloads
# arrives as binary. Both the agent and the verifier need to turn those into text:
# the agent to research, the verifier to check factuality against the sources.
# Same libraries upstream's own agent uses.
RUN /usr/local/bin/python3 -m pip install --no-cache-dir \
        openpyxl==3.1.5 \
        pypdf==6.1.1 \
        python-docx==1.2.0 \
        python-pptx==1.0.2

# The launcher names the interpreter by absolute path. Harbor builds the agent its
# own uv venv inside this container, so a bare `python3` would resolve to that venv,
# which has none of the libraries above.
COPY extract_text.py /usr/local/lib/extract_text.py
RUN printf '#!/bin/sh\nexec /usr/local/bin/python3 /usr/local/lib/extract_text.py "$@"\n' \
        > /usr/local/bin/extract-text \
    && chmod 0755 /usr/local/bin/extract-text

# Fail the build, not the run, if the launcher's interpreter cannot import an
# extractor: at runtime that surfaces as an unreadable corpus and a zero score.
RUN printf 'probe' > /tmp/probe.txt \
    && extract-text /tmp/probe.txt > /dev/null \
    && /usr/local/bin/python3 -c "import openpyxl, pypdf, docx, pptx" \
    && rm /tmp/probe.txt

WORKDIR /app
