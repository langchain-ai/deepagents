"""Harbor verifier: grade /app/answer.txt with the official OOLONG scorer."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from official_scorer import synth_process_response  # noqa: E402

_HERE = Path(os.path.dirname(os.path.abspath(__file__)))
_ANSWER = Path("/app/answer.txt")
_DATAPOINT = _HERE / "datapoint.json"
_REWARD = Path("/logs/verifier/reward.json")


def main() -> None:
    datapoint = json.loads(_DATAPOINT.read_text())
    output = _ANSWER.read_text() if _ANSWER.exists() else ""
    graded = synth_process_response(datapoint, output, datapoint.get("model", "unknown"))
    score = float(graded["score"])
    _REWARD.parent.mkdir(parents=True, exist_ok=True)
    _REWARD.write_text(json.dumps({"score": score}))
    print(
        f"OOLONG score={score} parse={graded['attempted_parse']!r} "
        f"gold={graded['answer']!r} confidence={graded['parse_confidence']}"
    )


if __name__ == "__main__":
    main()
