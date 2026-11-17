"""A stand-in for the MXC executor, used by unit tests.

Invoked as `python fake_executor.py --config <file> [--dry-run]`. It mimics
the observed contract (see EXECUTOR_CONTRACT.md):

- `--dry-run` prints `Dry run completed. Result: validation passed` and exits 0.
- A `process.commandLine` containing `FAKE_EXEC_TIMEOUT` sleeps past the
  request's `process.timeout` and then exits 255 with the JSON envelope the
  real executor emits.
- A `process.commandLine` containing `FAKE_HANG` starts a child process,
  writes the child's pid to the file named after `FAKE_HANG=`, and then both
  sleep forever ignoring the request timeout. This exercises the host-side
  kill of the whole tree.
- Anything else echoes `ok` and exits 0.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time


def main(argv: list[str]) -> int:
    config = None
    dry_run = False
    for index, arg in enumerate(argv):
        if arg == "--config":
            config = argv[index + 1]
        elif arg == "--dry-run":
            dry_run = True
    if config is None:
        sys.stderr.write("Error: No config provided. Use a positional path, --config, or --config-base64\n")
        return 1
    with open(config, encoding="utf-8") as handle:
        request = json.load(handle)
    if request.get("version") != "1.0.0":
        sys.stderr.write(
            "Request error\nConfiguration parse error: Unsupported contract version. Registered versions are: 0.9.0-alpha, 1.0.0, 1.1.0-alpha\n"
        )
        return 1
    if dry_run:
        sys.stdout.write("Script code length: 1\nRunner completed in 0ms\nDry run completed. Result: validation passed\n")
        return 0

    command = request["process"]["commandLine"]
    timeout_ms = int(request["process"].get("timeout") or 0)

    if "FAKE_EXEC_TIMEOUT" in command:
        sys.stdout.write("started\n")
        sys.stdout.flush()
        time.sleep(timeout_ms / 1000 if timeout_ms else 1)
        sys.stderr.write(json.dumps({"error": {"code": "backend_error", "message": f"script timed out after {timeout_ms}ms"}}) + "\n")
        return 255

    match = re.search(r"FAKE_HANG=(\S+)", command)
    if match:
        pid_file = match.group(1).strip("'\"")
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(3600)"],
            start_new_session=sys.platform != "win32",
        )
        with open(pid_file, "w", encoding="utf-8") as handle:
            handle.write(str(child.pid))
        sys.stdout.write("hanging\n")
        sys.stdout.flush()
        time.sleep(3600)
        return 0

    sys.stdout.write("ok\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
