"""Script to check for import errors in specified Python files."""

import importlib
import sys
import traceback


def file_to_module(filepath: str) -> str:
    """Convert a file path to a module name."""
    # Remove .py extension and convert path separators to dots
    return filepath.removesuffix(".py").replace("/", ".").replace("\\", ".")


if __name__ == "__main__":
    files = sys.argv[1:]
    has_failure = False
    for file in files:
        module_name = file_to_module(file)
        try:
            importlib.import_module(module_name)
        except Exception:
            has_failure = True
            print(file)  # noqa: T201
            traceback.print_exc()
            print()  # noqa: T201

    sys.exit(1 if has_failure else 0)
