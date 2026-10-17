"""Check imports script.

Quickly verify that a list of Python files can be loaded by the Python interpreter
without raising any errors. Ran before running more expensive tests. Useful in
Makefiles.

If loading a file fails, the script prints the problematic filename and the detailed
error traceback.
"""

import os
import random
import string
import sys
import tempfile
import traceback
from importlib.machinery import SourceFileLoader

if __name__ == "__main__":
    files = sys.argv[1:]
    has_failure = False
    with tempfile.TemporaryDirectory() as home:
        # Point the home directory at a throwaway dir so importing a module can't
        # read or depend on the developer's real `~` state (e.g. `~/.deepagents`
        # config, MCP auth tokens). `Path.home()` resolves from `HOME` on POSIX
        # and `USERPROFILE` / `HOMEDRIVE`+`HOMEPATH` on Windows, so override all
        # of them to keep the isolation cross-platform.
        os.environ["HOME"] = home
        os.environ["USERPROFILE"] = home
        os.environ.pop("HOMEDRIVE", None)
        os.environ.pop("HOMEPATH", None)
        for file in files:
            try:
                module_name = "".join(
                    random.choice(string.ascii_letters) for _ in range(20)
                )
                SourceFileLoader(module_name, file).load_module()
            except Exception:
                has_failure = True
                print(file)
                traceback.print_exc()
                print()

    sys.exit(1 if has_failure else 0)
