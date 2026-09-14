FROM ghcr.io/steel-dev/steel-browser-api@sha256:f9a4648883dc06c402f5ffbec1c906bf9a803b5b737a1347de4e0aa0ca8d944a
USER 1000:1000
WORKDIR /opt/browser
COPY --chown=1000:1000 . /opt/browser/
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENTRYPOINT ["python3"]
