FROM ghcr.io/steel-dev/steel-browser-api@sha256:f9a4648883dc06c402f5ffbec1c906bf9a803b5b737a1347de4e0aa0ca8d944a
LABEL org.opencontainers.image.base.revision="2b41124d8e2953b0afe355c534e3c9aa71edae26"
USER root
RUN mkdir -p /var/lib/steel/profile && chown 1000:1000 /files && chmod 0700 /files
COPY profile.py bootstrap.mjs /app/api/talon/
COPY managed-policy.json /etc/chromium/policies/managed/talon.json
ENV CHROME_USER_DATA_DIR=/var/lib/steel/profile PROXY_URL=http://172.30.14.3:8080 NODE_ENV=production DISABLE_CHROME_SANDBOX=true
WORKDIR /app/api
STOPSIGNAL SIGTERM
ENTRYPOINT ["python3", "/app/api/talon/profile.py"]
