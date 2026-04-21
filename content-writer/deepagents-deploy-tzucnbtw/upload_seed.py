#!/usr/bin/env python3
"""Upload _seed.json to GCS under a random object key and update .env."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

from google.cloud import storage
from google.oauth2 import service_account

ROOT = Path(__file__).parent
SEED_FILE = ROOT / "_seed.json"
ENV_FILE = ROOT / ".env"
ENV_VAR = "GCS_SEED_PATH"
KEY_ENV_VAR = "GCS_SERVICE_ACCOUNT_KEY"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _set_env_var(path: Path, key: str, value: str) -> None:
    new_line = f"{key}={value}"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        if stripped.split("=", 1)[0].strip() == key:
            lines[i] = new_line
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return
    lines.append(new_line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bucket", help="Target GCS bucket name")
    parser.add_argument(
        "--prefix",
        default="deepagents-seeds",
        help="Object key prefix (default: deepagents-seeds)",
    )
    args = parser.parse_args()

    if not SEED_FILE.exists():
        print(f"error: {SEED_FILE} not found", file=sys.stderr)
        return 2

    _load_env_file(ENV_FILE)
    key_json = os.environ.get(KEY_ENV_VAR)
    if not key_json:
        print(f"error: {KEY_ENV_VAR} env var not set", file=sys.stderr)
        return 2

    key_info = json.loads(key_json)
    credentials = service_account.Credentials.from_service_account_info(key_info)
    client = storage.Client(
        project=key_info.get("project_id"),
        credentials=credentials,
    )

    object_name = f"{args.prefix}/{secrets.token_urlsafe(24)}/_seed.json"
    blob = client.bucket(args.bucket).blob(object_name)
    blob.upload_from_filename(str(SEED_FILE), content_type="application/json")

    gs_uri = f"gs://{args.bucket}/{object_name}"
    _set_env_var(ENV_FILE, ENV_VAR, gs_uri)

    print(f"uploaded {SEED_FILE.name} -> {gs_uri}")
    print(f"updated {ENV_FILE} with {ENV_VAR}={gs_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
