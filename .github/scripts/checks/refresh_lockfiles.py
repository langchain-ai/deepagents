"""Refresh affected lockfiles and print their repository-relative paths."""

from __future__ import annotations

import subprocess
import sys

from check_lockfiles_pre_commit import REPO_ROOT, python_version, stale_lock_dirs


def main(manifests: list[str]) -> None:
    """Re-lock changed packages and their transitive local consumers."""
    for directory in stale_lock_dirs(manifests):
        subprocess.run(
            [
                "uv",
                "lock",
                "--directory",
                directory,
                "--python",
                python_version(REPO_ROOT / directory),
            ],
            check=True,
            cwd=REPO_ROOT,
            stdout=sys.stderr,
        )
        print(f"{directory}/uv.lock")


if __name__ == "__main__":
    main(sys.argv[1:])
