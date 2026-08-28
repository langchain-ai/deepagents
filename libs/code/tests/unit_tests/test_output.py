"""Tests for the JSON output utility."""

import json
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from deepagents_code.output import write_json


class TestWriteJson:
    """Tests for write_json envelope format."""
