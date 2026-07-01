FROM ghcr.io/astral-sh/uv:python3.13-bookworm

ARG GCX_VERSION=0.4.0
ARG TEA_VERSION=0.14.1

ENV DEBIAN_FRONTEND=noninteractive \
    NODE_PATH=/opt/whatsapp-bridge/node_modules \
    PATH="/opt/deepagents-talon-venv/bin:${PATH}" \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/deepagents-talon-venv

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        chromium \
        curl \
        ffmpeg \
        fonts-liberation \
        gh \
        git \
        nodejs \
        npm \
        ripgrep \
        xz-utils && \
    arch="$(dpkg --print-architecture)" && \
    case "$arch" in \
        amd64) tool_arch=amd64 ;; \
        arm64) tool_arch=arm64 ;; \
        *) echo "Unsupported architecture for gcx/tea: $arch" >&2; exit 1 ;; \
    esac && \
    gcx_asset="gcx_${GCX_VERSION}_linux_${tool_arch}.tar.gz" && \
    curl -fsSLO "https://github.com/grafana/gcx/releases/download/v${GCX_VERSION}/${gcx_asset}" && \
    curl -fsSLO "https://github.com/grafana/gcx/releases/download/v${GCX_VERSION}/gcx_${GCX_VERSION}_checksums.txt" && \
    grep "  ${gcx_asset}$" "gcx_${GCX_VERSION}_checksums.txt" | sha256sum -c - && \
    tar -xzf "${gcx_asset}" -C /usr/local/bin gcx && \
    rm -f "${gcx_asset}" "gcx_${GCX_VERSION}_checksums.txt" && \
    tea_asset="tea-${TEA_VERSION}-linux-${tool_arch}.xz" && \
    curl -fsSLO "https://gitea.com/gitea/tea/releases/download/v${TEA_VERSION}/${tea_asset}" && \
    curl -fsSLO "https://gitea.com/gitea/tea/releases/download/v${TEA_VERSION}/${tea_asset}.sha256" && \
    sha256sum -c "${tea_asset}.sha256" && \
    xz -d "${tea_asset}" && \
    install -m 0755 "tea-${TEA_VERSION}-linux-${tool_arch}" /usr/local/bin/tea && \
    rm -f "tea-${TEA_VERSION}-linux-${tool_arch}" "${tea_asset}.sha256" && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/deepagents-src

COPY libs/deepagents ./libs/deepagents
COPY libs/code ./libs/code
COPY libs/partners/quickjs ./libs/partners/quickjs
COPY libs/talon ./libs/talon
COPY examples/talon/AGENTS.md /opt/deepagents-talon/AGENTS.md

RUN mkdir -p /opt/whatsapp-bridge && \
    cp libs/talon/deepagents_talon/channels/whatsapp_bridge/package.json \
        /opt/whatsapp-bridge/package.json && \
    cd /opt/whatsapp-bridge && \
    npm install --omit=dev

RUN cd libs/talon && uv sync --frozen --extra speech --extra video

WORKDIR /workspace

CMD ["/opt/deepagents-talon-venv/bin/deepagents-talon"]
