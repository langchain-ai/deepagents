"""Optional provider packages must not load as a side effect of `import deepagents`."""

import subprocess
import sys


def test_import_does_not_load_langchain_openai() -> None:
    """`langchain_openai` pulls in the `openai` SDK, a large import non-OpenAI users never need."""
    code = "import sys, deepagents; print('langchain_openai' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)  # noqa: S603  # fixed argv: this interpreter and a literal snippet
    assert result.stdout.strip() == "False"
