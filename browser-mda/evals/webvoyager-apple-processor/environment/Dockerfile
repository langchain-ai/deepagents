# Harbor runs the project's sandbox/setup.sh AND the MDA python runner inside
# this image, so the trial environment must satisfy both:
#   * Node + npm  -> setup.sh installs the `browse` CLI
#   * python      -> the MDA Harbor adapter's runner (it invokes `python`, not
#                    `python3`, so the symlink is required, not cosmetic)
# Chrome itself is installed by setup.sh, not baked here.
FROM node:24-bookworm-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl grep procps python3 python3-venv python3-pip \
 && ln -sf /usr/bin/python3 /usr/local/bin/python \
 && rm -rf /var/lib/apt/lists/*

# Debian marks its system Python as externally managed (PEP 668), so the MDA
# adapter's `pip install` of the agent runtime is refused. This image exists only
# to host one trial, so allowing installs into system Python is safe here and
# avoids forcing the adapter through a venv it does not know about.
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN rm -f /usr/lib/python3*/EXTERNALLY-MANAGED

WORKDIR /workspace
