FROM python:3.12-slim

# Installed at build time (the build phase has network) so the in-sandbox agent's
# runtime bootstrap skips apt. `poppler-utils` provides pdftotext for the corpus'
# PDFs; curl lets the agent fetch web pages it finds.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# DRBench's corpus is PDF/DOCX/XLSX/PPTX/JSONL. The benchmark scores research and
# synthesis, not container-format parsing, so the task ships the same extraction
# libraries upstream's own agent uses, behind one `extract-text` entry point.
#
# Both the install and the launcher name the interpreter by absolute path. Harbor's
# LangGraph agent builds its own uv venv inside this container, so a bare `python3`
# would resolve to whichever interpreter is first on the agent's PATH -- likely that
# venv, which does not have these libraries.
RUN /usr/local/bin/python3 -m pip install --no-cache-dir \
        openpyxl==3.1.5 \
        pypdf==6.1.1 \
        python-docx==1.2.0 \
        python-pptx==1.0.2

COPY extract_text.py /usr/local/lib/extract_text.py
RUN printf '#!/bin/sh\nexec /usr/local/bin/python3 /usr/local/lib/extract_text.py "$@"\n' \
        > /usr/local/bin/extract-text \
    && chmod 0755 /usr/local/bin/extract-text

# Fail the build, not the run, if the launcher's interpreter cannot import an
# extractor: at runtime this surfaces as an unreadable corpus and a zero score.
RUN printf 'x' > /tmp/probe.txt \
    && extract-text /tmp/probe.txt > /dev/null \
    && /usr/local/bin/python3 -c "import openpyxl, pypdf, docx, pptx" \
    && rm /tmp/probe.txt

COPY files/ /app/files/
WORKDIR /app
