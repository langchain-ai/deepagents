from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
EXTERNAL_ROOT = WORKSPACE_ROOT / "external"
OUTPUT_PATH = REPO_ROOT / "libs" / "deepagents" / "tests" / "evals" / "data" / "curated_external_evals.json"
SUMMARY_PATH = REPO_ROOT / "libs" / "deepagents" / "tests" / "evals" / "CURATED_EXTERNAL_EVALS.md"


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sanitize_answer_substrings(values: list[str]) -> list[str]:
    return [value for value in values if value]


def build_toolbench_cases() -> list[dict[str, Any]]:
    selected = {
        ("G1", 1): {
            "expected_calls": [
                {
                    "tool_key": ("SQUAKE", "Checkhealth"),
                    "allowed_args": {},
                    "output": "SQUAKE API status: healthy.",
                },
                {
                    "tool_key": ("SQUAKE", "Projects"),
                    "allowed_args": {},
                    "output": "Projects available: EcoRoute, SAF Horizon, and GreenMiles.",
                },
            ],
            "answer_substrings": ["healthy", "EcoRoute"],
        },
        ("G1", 5): {
            "expected_calls": [
                {
                    "tool_key": (
                        "Transportistas de Argentina",
                        "/tracking/correo_argentino/create_task/:service/:tracking_code",
                    ),
                    "allowed_args": {"service": ["ecommerce"], "tracking_code": ["ABC123"]},
                    "output": "Created tracking task successfully. task_id=task-abc123.",
                },
            ],
            "answer_substrings": ["task-abc123"],
        },
        ("G2", 1): {
            "expected_calls": [
                {
                    "tool_key": ("Create Container Tracking", "Get Tracking Data"),
                    "allowed_args": {"id": ["6045e2f44e1b233199a5e77a"]},
                    "output": "Tracking data for 6045e2f44e1b233199a5e77a: In transit, expected delivery Friday.",
                },
                {
                    "tool_key": ("SQUAKE", "Checkhealth"),
                    "allowed_args": {},
                    "output": "SQUAKE authentication system status: healthy.",
                },
            ],
            "answer_substrings": ["In transit", "healthy"],
        },
        ("G2", 2): {
            "expected_calls": [
                {
                    "tool_key": ("Turkey Postal Codes", "il"),
                    "allowed_args": {"il": [34, "34"]},
                    "output": "Istanbul postal codes include 34010 Fatih, 34367 Sisli, and 34710 Kadikoy.",
                },
                {
                    "tool_key": (
                        "Transportistas de Argentina",
                        "/tracking/correo_argentino/result_task/:task_id",
                    ),
                    "allowed_args": {"task_id": ["987654321"]},
                    "output": "Tracking task 987654321: Delivered to recipient.",
                },
            ],
            "answer_substrings": ["34010", "Delivered"],
        },
        ("G3", 1): {
            "expected_calls": [
                {
                    "tool_key": ("The Cocktail DB", "List of Cocktails"),
                    "allowed_args": {},
                    "output": "Popular cocktails: Sunset Spritz (id=45) and Citrus Fizz (id=52).",
                },
                {
                    "tool_key": ("The Cocktail DB", "Detailed Cocktail Recipe by ID"),
                    "allowed_args": {"id": ["45", 45]},
                    "output": "Sunset Spritz recipe: combine orange juice, soda water, and bitters; shake and pour over ice.",
                },
                {
                    "tool_key": ("Web Search", "newsSearch"),
                    "allowed_args": {},
                    "output": "Birthday celebration news: 'Community birthday festival ideas' and 'How hosts plan memorable parties'.",
                },
            ],
            "answer_substrings": ["Sunset Spritz", "birthday"],
        },
        ("G3", 2): {
            "expected_calls": [
                {
                    "tool_key": ("The Cocktail DB", "List of Cocktails"),
                    "allowed_args": {},
                    "output": "Popular cocktails: Sunset Spritz (id=45) and Herb Garden Collins (id=61).",
                },
                {
                    "tool_key": ("The Cocktail DB", "Detailed Cocktail Recipe by ID"),
                    "allowed_args": {"id": ["61", 61]},
                    "output": "Herb Garden Collins recipe: muddle basil, add gin, lemon, and tonic, then stir over ice.",
                },
                {
                    "tool_key": ("Web Search", "newsSearch"),
                    "allowed_args": {},
                    "output": "Culinary news: 'Chefs embrace regional pairings' and 'Culinary trends for home hosts'.",
                },
                {
                    "tool_key": (
                        "Investors Exchange (IEX) Trading",
                        "IEX Regulation SHO Threshold Securities List",
                    ),
                    "allowed_args": {"symbol": ["NVDA"]},
                    "output": "Threshold securities entry: NVDA remains on the list for the current report.",
                },
            ],
            "answer_substrings": ["Herb Garden Collins", "NVDA"],
        },
    }
    files = {
        "G1": EXTERNAL_ROOT / "ToolBench" / "data_example" / "instruction" / "G1_query.json",
        "G2": EXTERNAL_ROOT / "ToolBench" / "data_example" / "instruction" / "G2_query.json",
        "G3": EXTERNAL_ROOT / "ToolBench" / "data_example" / "instruction" / "G3_query.json",
    }

    cases: list[dict[str, Any]] = []
    for group, path in files.items():
        for item in load_json(path):
            key = (group, item["query_id"])
            if key not in selected:
                continue

            tool_name_map: dict[tuple[str, str], str] = {}
            tools: list[dict[str, Any]] = []
            for api in item["api_list"]:
                tool_key = (api["tool_name"], api["api_name"])
                tool_name = f"toolbench_{slugify(api['tool_name'])}_{slugify(api['api_name'])}"
                tool_name_map[tool_key] = tool_name
                properties = {}
                required = []
                for parameter in api.get("required_parameters", []):
                    properties[parameter["name"]] = {
                        "type": parameter["type"].lower(),
                        "description": parameter.get("description", ""),
                    }
                    required.append(parameter["name"])
                for parameter in api.get("optional_parameters", []):
                    properties[parameter["name"]] = {
                        "type": parameter["type"].lower(),
                        "description": parameter.get("description", ""),
                    }
                description = (
                    f"ToolBench proxy for {api['tool_name']} -> {api['api_name']}. "
                    f"{api.get('api_description', '').strip()} Method: {api.get('method', 'GET')}."
                )
                tools.append(
                    {
                        "name": tool_name,
                        "description": description.strip(),
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    }
                )

            expected_calls = []
            for call in selected[key]["expected_calls"]:
                expected_calls.append(
                    {
                        "name": tool_name_map[call["tool_key"]],
                        "allowed_args": call["allowed_args"],
                        "output": call["output"],
                    }
                )

            cases.append(
                {
                    "id": f"toolbench_{group.lower()}_{item['query_id']}",
                    "source": "toolbench",
                    "source_case_id": f"{group}:{item['query_id']}",
                    "prompt": item["query"],
                    "tools": tools,
                    "expected_calls": expected_calls,
                    "answer_substrings": sanitize_answer_substrings(selected[key]["answer_substrings"]),
                    "provenance": {
                        "path": str(path.relative_to(WORKSPACE_ROOT)),
                        "query_id": item["query_id"],
                    },
                }
            )
    return cases


BFCL_SELECTED_IDS = [
    "simple_python_0",
    "simple_python_1",
    "simple_python_2",
    "simple_python_3",
    "simple_python_5",
    "simple_python_7",
    "multiple_0",
    "multiple_1",
    "multiple_2",
    "multiple_3",
    "multiple_4",
    "multiple_5",
    "multiple_6",
    "parallel_0",
    "parallel_1",
    "parallel_2",
    "parallel_4",
    "parallel_5",
    "parallel_6",
    "parallel_7",
]

BFCL_OUTPUTS = {
    "simple_python_0": (["Area: 25 square units."], ["25"]),
    "simple_python_1": (["Factorial result: 120."], ["120"]),
    "simple_python_2": (["Hypotenuse: 6.403."], ["6.403"]),
    "simple_python_3": (["Roots: 1 and 2."], ["1", "2"]),
    "simple_python_5": (["Roots: 4 and -0.3333."], ["4"]),
    "simple_python_7": (["Circumference: 25.13 inches."], ["25.13"]),
    "multiple_0": (["Triangle properties: area 6, perimeter 12, angles 37, 53, and 90 degrees."], ["area 6", "perimeter 12"]),
    "multiple_1": (["Triangle area: 6 square units."], ["6"]),
    "multiple_2": (["Capital of Brazil: Brasilia."], ["Brasilia"]),
    "multiple_3": (["Euclidean distance: 2.83."], ["2.83"]),
    "multiple_4": (["Displacement: 225 meters."], ["225"]),
    "multiple_5": (["Weather on 2019-12-13: 8 C with 16 km/h winds."], ["8 C", "16"]),
    "multiple_6": (["Capacitance: 8.85e-9 farads."], ["8.85e-9"]),
    "parallel_0": (
        ["Queued Taylor Swift for 20 minutes.", "Queued Maroon 5 for 15 minutes."],
        ["Taylor Swift", "Maroon 5"],
    ),
    "parallel_1": (
        ["Induced electromagnetic force: 2.5 volts.", "Induced electromagnetic force: 1.0 volts."],
        ["2.5", "1.0"],
    ),
    "parallel_2": (
        ["Copper wire resistance: 0.0084 ohms.", "Aluminum wire resistance: 0.0141 ohms."],
        ["0.0084", "0.0141"],
    ),
    "parallel_4": (
        ["BMI: 23.9.", "BMI: 21.4."],
        ["23.9", "21.4"],
    ),
    "parallel_5": (
        ["Netflix ratings: Friends 8.9.", "Hulu ratings: The Office 8.8 and Stranger Things 8.7."],
        ["Friends", "The Office"],
    ),
    "parallel_6": (
        ["Chicago sales tax: 3.13.", "Sacramento sales tax: 4.19.", "Portland sales tax: 0.00."],
        ["Chicago", "Sacramento", "Portland"],
    ),
    "parallel_7": (
        ["Factorial result: 120.", "Factorial result: 3628800.", "Factorial result: 1307674368000."],
        ["120", "3628800", "1307674368000"],
    ),
}


def build_bfcl_cases() -> list[dict[str, Any]]:
    data_files = {
        "simple_python_": EXTERNAL_ROOT / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data" / "BFCL_v4_simple_python.json",
        "multiple_": EXTERNAL_ROOT / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data" / "BFCL_v4_multiple.json",
        "parallel_": EXTERNAL_ROOT / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data" / "BFCL_v4_parallel.json",
    }
    answer_files = {
        "simple_python_": EXTERNAL_ROOT / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data" / "possible_answer" / "BFCL_v4_simple_python.json",
        "multiple_": EXTERNAL_ROOT / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data" / "possible_answer" / "BFCL_v4_multiple.json",
        "parallel_": EXTERNAL_ROOT / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data" / "possible_answer" / "BFCL_v4_parallel.json",
    }

    selected_ids = set(BFCL_SELECTED_IDS)
    answers: dict[str, list[dict[str, Any]]] = {}
    for prefix, path in answer_files.items():
        del prefix
        for item in load_jsonl(path):
            if item["id"] in selected_ids:
                answers[item["id"]] = item["ground_truth"]

    cases: list[dict[str, Any]] = []
    for prefix, path in data_files.items():
        answer_path = answer_files[prefix]
        for item in load_jsonl(path):
            if item["id"] not in selected_ids:
                continue

            tools = []
            for function in item["function"]:
                tools.append(
                    {
                        "name": function["name"],
                        "description": function["description"],
                        "parameters": {
                            "type": "object",
                            "properties": function["parameters"].get("properties", {}),
                            "required": function["parameters"].get("required", []),
                        },
                    }
                )

            outputs, answer_substrings = BFCL_OUTPUTS[item["id"]]
            expected_calls = []
            for index, ground_truth in enumerate(answers[item["id"]]):
                name, allowed_args = next(iter(ground_truth.items()))
                expected_calls.append(
                    {
                        "name": name,
                        "allowed_args": allowed_args,
                        "output": outputs[index],
                    }
                )

            cases.append(
                {
                    "id": f"bfcl_{item['id']}",
                    "source": "bfcl",
                    "source_case_id": item["id"],
                    "prompt": item["question"][0][0]["content"],
                    "tools": tools,
                    "expected_calls": expected_calls,
                    "answer_substrings": sanitize_answer_substrings(answer_substrings),
                    "provenance": {
                        "path": str(path.relative_to(WORKSPACE_ROOT)),
                        "possible_answer_path": str(answer_path.relative_to(WORKSPACE_ROOT)),
                    },
                }
            )
    return cases


APIBENCH_SELECTION = {
    "huggingface": [0, 2, 4, 7],
    "torchhub": [0, 1, 3, 5],
    "tensorflow": [0, 4, 5, 6],
}


def build_apibench_cases() -> list[dict[str, Any]]:
    files = {
        "huggingface": EXTERNAL_ROOT / "gorilla" / "data" / "apibench" / "huggingface_eval.json",
        "torchhub": EXTERNAL_ROOT / "gorilla" / "data" / "apibench" / "torchhub_eval.json",
        "tensorflow": EXTERNAL_ROOT / "gorilla" / "data" / "apibench" / "tensorflow_eval.json",
    }

    all_rows = {label: load_jsonl(path) for label, path in files.items()}
    cases: list[dict[str, Any]] = []

    for label, indices in APIBENCH_SELECTION.items():
        selected_rows = [(index, all_rows[label][index]) for index in indices]
        tools = []
        for _, row in selected_rows:
            api_name = row["api_data"].get("api_name") or row["api_call"]
            tool_name = f"{label}_{slugify(str(api_name))}"
            description = (
                f"APIBench proxy for {api_name}. "
                f"Functionality: {row['api_data'].get('functionality', '')}. "
                f"Description: {row['api_data'].get('description', '')}"
            ).strip()
            tools.append(
                {
                    "name": tool_name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                    "output": f"Recommended API: {api_name}. Provider: {row['provider']}.",
                    "api_name": str(api_name),
                }
            )

        tool_map = {tool["api_name"]: tool["name"] for tool in tools}

        for row_index, row in selected_rows:
            instruction = row["code"].split("###Instruction: ", 1)[1].split("\n###Output:", 1)[0].strip()
            api_name = str(row["api_data"].get("api_name") or row["api_call"])
            cases.append(
                {
                    "id": f"apibench_{label}_{row_index}_{slugify(api_name)}",
                    "source": "gorillabench",
                    "source_case_id": api_name,
                    "prompt": instruction,
                    "tools": [
                        {
                            "name": tool["name"],
                            "description": tool["description"],
                            "parameters": tool["parameters"],
                        }
                        for tool in tools
                    ],
                    "expected_calls": [
                        {
                            "name": tool_map[api_name],
                            "allowed_args": {},
                            "output": f"Recommended API: {api_name}. Provider: {row['provider']}.",
                        }
                    ],
                    "answer_substrings": [api_name],
                    "provenance": {
                        "path": str(files[label].relative_to(WORKSPACE_ROOT)),
                        "provider": row["provider"],
                    },
                }
            )
    return cases


HOTPOT_SELECTION = [
    ("5a8b57f25542995d1e6f1371", ["yes"]),
    ("5a8c7595554299585d9e36b6", ["Chief of Protocol"]),
    ("5a85ea095542994775f606a8", ["Animorphs"]),
    ("5adbf0a255429947ff17385a", ["no"]),
    ("5a8e3ea95542995a26add48d", ["Greenwich Village"]),
    ("5abd94525542992ac4f382d2", ["YG Entertainment"]),
    ("5a85b2d95542997b5ce40028", ["Eenasul Fateh"]),
    ("5a87ab905542996e4f3088c1", ["3,677"]),
    ("5a7bbb64554299042af8f7cc", ["Terry Richardson"]),
    ("5a8db19d5542994ba4e3dd00", ["yes"]),
    ("5a7166395542994082a3e814", ["Kansas Song"]),
    ("5a877e5d5542993e715abf7d", ["David Weissman"]),
]


def build_hotpot_cases() -> list[dict[str, Any]]:
    selection = {case_id: answer_substrings for case_id, answer_substrings in HOTPOT_SELECTION}
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    cases: list[dict[str, Any]] = []

    for row in dataset:
        if row["id"] not in selection:
            continue
        context = row["context"]
        supporting_titles: list[str] = []
        for title in row["supporting_facts"]["title"]:
            if title not in supporting_titles:
                supporting_titles.append(title)

        documents = []
        chosen_titles = list(supporting_titles)
        for title in context["title"]:
            if title not in chosen_titles and len(chosen_titles) < len(supporting_titles) + 2:
                chosen_titles.append(title)
        for title, sentences in zip(context["title"], context["sentences"]):
            if title in chosen_titles:
                documents.append({"title": title, "text": " ".join(sentence.strip() for sentence in sentences if sentence.strip())})

        cases.append(
            {
                "id": f"hotpot_{row['id']}",
                "source": "hotpotqa",
                "source_case_id": row["id"],
                "prompt": row["question"],
                "documents": documents,
                "answer_substrings": selection[row["id"]],
                "provenance": {
                    "dataset": "hotpotqa/hotpot_qa",
                    "config": "distractor",
                    "split": "validation",
                    "supporting_titles": supporting_titles,
                },
            }
        )
    return cases


def write_summary(cases: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["source"]] = counts.get(case["source"], 0) + 1

    lines = [
        "# Curated External Eval Pack",
        "",
        "This pack freezes 50 offline evals for the Deep Agents pytest/LangSmith harness.",
        "",
        "## Source Breakdown",
        "",
    ]
    for source in sorted(counts):
        lines.append(f"- `{source}`: {counts[source]} cases")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- ToolBench uses the official repository example queries because the repo does not vendor the full dataset in-tree.",
            "- BFCL and APIBench come from the official Gorilla repository.",
            "- HotpotQA comes from the official `hotpotqa/hotpot_qa` `distractor` validation split, and only the selected documents needed for each case are frozen into the JSON fixture.",
            "- The generated JSON is what tests consume in CI, so the GitHub Actions workflow stays offline and deterministic.",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    toolbench_cases = build_toolbench_cases()
    bfcl_cases = build_bfcl_cases()
    apibench_cases = build_apibench_cases()
    hotpot_cases = build_hotpot_cases()

    cases = toolbench_cases + bfcl_cases + apibench_cases + hotpot_cases
    payload = {
        "case_count": len(cases),
        "sources": {
            "toolbench": "OpenBMB/ToolBench official repository examples",
            "bfcl": "ShishirPatil/gorilla BFCL v4 data",
            "gorillabench": "ShishirPatil/gorilla APIBench eval files",
            "hotpotqa": "hotpotqa/hotpot_qa distractor validation split",
        },
        "cases": cases,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_summary(cases)
    print(f"Wrote {len(cases)} curated cases to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
