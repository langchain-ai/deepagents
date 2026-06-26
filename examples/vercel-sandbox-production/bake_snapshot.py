"""Bake a Vercel Sandbox snapshot with the data/plotting stack pre-installed.

Cold-booting a sandbox and running `pip install pandas matplotlib ...` on every
run costs 30-60s. Baking those packages into a snapshot once drops cold start to
a few seconds — the snapshot ships the libraries already installed.

Run once, then export the printed id so `agent.py` boots from it:

    python bake_snapshot.py
    export VERCEL_SANDBOX_SNAPSHOT_ID=snap_xxx

Re-bake whenever you change the package set. Requires VERCEL_TOKEN,
VERCEL_TEAM_ID, VERCEL_PROJECT_ID in the environment.
"""

from __future__ import annotations

from dotenv import load_dotenv
from vercel.sandbox import Sandbox

load_dotenv()

# Keep this in sync with what your agents actually import. Add reportlab/pypdf/
# python-docx/openpyxl/Pillow here if you generate documents.
PACKAGES = [
    "pandas",
    "numpy",
    "matplotlib",
]


def main() -> None:
    print("Creating bootstrap sandbox...")
    sandbox = Sandbox.create(runtime="python3.13", timeout=15 * 60 * 1000)

    try:
        print(f"Installing: {' '.join(PACKAGES)}")
        result = sandbox.run_command("pip", ["install", "-q", *PACKAGES])
        if result.exit_code != 0:
            print(result.stderr())
            msg = f"pip install failed with exit code {result.exit_code}"
            raise SystemExit(msg)

        print("Capturing snapshot (this can take a minute)...")
        snapshot = sandbox.snapshot(expiration=0)  # 0 = never expires
        snapshot_id = snapshot.snapshot_id
    finally:
        sandbox.stop()

    print()
    print(f"VERCEL_SANDBOX_SNAPSHOT_ID={snapshot_id}")
    print()
    print("Export it before running agent.py:")
    print(f"    export VERCEL_SANDBOX_SNAPSHOT_ID={snapshot_id}")


if __name__ == "__main__":
    main()
