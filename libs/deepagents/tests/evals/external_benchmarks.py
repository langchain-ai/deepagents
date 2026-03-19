from __future__ import annotations

import contextlib
import copy
import inspect
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langsmith import testing as t

from deepagents import create_deep_agent
from tests.evals.data.bfcl_apis.message_api import MessageAPI
from tests.evals.data.bfcl_apis.ticket_api import TicketAPI
from tests.evals.data.bfcl_apis.trading_bot import TradingBot
from tests.evals.data.bfcl_apis.travel_booking import TravelAPI
from tests.evals.data.bfcl_apis.vehicle_control import VehicleControlAPI
from tests.evals.utils import (
    AgentTrajectory,
    SuccessAssertion,
    TrajectoryScorer,
    run_agent,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph


@dataclass(frozen=True)
class _NormalizedSubstringsPresent(SuccessAssertion):
    """Fail unless all expected snippets appear after normalizing whitespace, case, and quote characters."""

    snippets: tuple[str, ...]

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", "", text).lower().replace("'", "").replace('"', "").replace("`", "")

    def check(self, trajectory: AgentTrajectory) -> bool:
        answer = self._normalize(trajectory.answer)
        return all(self._normalize(snippet) in answer for snippet in self.snippets)

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        return f"Expected final text to contain all normalized snippets {list(self.snippets)}, got {trajectory.answer!r}"


_DATA_DIR = Path(__file__).parent / "data" / "benchmark_samples"

_FRAMES_IDS = {
    "frames_10",
    "frames_11",
    "frames_12",
    "frames_16",
    "frames_18",
}
_NEXUS_IDS = {
    "nexus_nvd_nested_13",
    "nexus_nvd_nested_14",
    "nexus_placesapi_15",
    "nexus_multiversemath_17",
    "nexus_multiversemath_18",
}
_BFCL_V3_IDS = {
    "multi_turn_composite_97",
    "multi_turn_composite_116",
    "multi_turn_composite_199",
    "multi_turn_miss_func_55",
    "multi_turn_miss_param_55",
}

_FILE_BACKED_SYSTEM_PROMPT = (
    "Use the files already present in the workspace to solve the task. "
    "Return only the final answer requested by the prompt. "
    "Do not use the task tool or any subagent delegation."
)


def _load_final(name: str) -> list[dict[str, Any]]:
    path = _DATA_DIR / f"{name}_final.json"
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return data.get("tasks", data.get("cases", []))
    return data


FRAMES_CASES: list[dict[str, Any]] = [case for case in _load_final("frames") if case["id"] in _FRAMES_IDS]
assert len(FRAMES_CASES) == len(_FRAMES_IDS), (
    f"Expected {len(_FRAMES_IDS)} FRAMES cases, found {len(FRAMES_CASES)}. Missing: {_FRAMES_IDS - {c['id'] for c in FRAMES_CASES}}"
)
NEXUS_CASES: list[dict[str, Any]] = [case for case in _load_final("nexus") if case["id"] in _NEXUS_IDS]
assert len(NEXUS_CASES) == len(_NEXUS_IDS), (
    f"Expected {len(_NEXUS_IDS)} Nexus cases, found {len(NEXUS_CASES)}. Missing: {_NEXUS_IDS - {c['id'] for c in NEXUS_CASES}}"
)
BFCL_V3_CASES: list[dict[str, Any]] = [case for case in _load_final("bfcl_v3") if case["id"] in _BFCL_V3_IDS]
assert len(BFCL_V3_CASES) == len(_BFCL_V3_IDS), (
    f"Expected {len(_BFCL_V3_IDS)} BFCL cases, found {len(BFCL_V3_CASES)}. Missing: {_BFCL_V3_IDS - {c['id'] for c in BFCL_V3_CASES}}"
)


def _external_eval_metadata(
    *,
    case_id: str,
    vertical: str,
    origin_benchmark: str,
    difficulty: str,
    tags: list[str] | None = None,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "vertical": vertical,
        "origin_benchmark": origin_benchmark,
        "difficulty": difficulty,
        "subset": "curated_external_hard",
        "tags": tags or [],
    }


def _create_file_backed_agent(model: BaseChatModel) -> CompiledStateGraph:
    return create_deep_agent(model=model, system_prompt=_FILE_BACKED_SYSTEM_PROMPT)


def _create_text_scorer(case: dict[str, Any]) -> TrajectoryScorer:
    return TrajectoryScorer().success(_NormalizedSubstringsPresent(snippets=tuple(case["answer_snippets"])))


def run_frames_case(case: dict[str, Any], model: BaseChatModel) -> None:
    """Run a FRAMES case with file-backed retrieval and text scoring.

    Args:
        case: A curated FRAMES case dict with keys `id`, `prompt`, `files`,
            and `answer_snippets`.
        model: The chat model to use for the agent.
    """
    agent = _create_file_backed_agent(model)
    run_agent(
        agent,
        model=model,
        query=case["prompt"],
        initial_files=case["files"],
        scorer=_create_text_scorer(case),
        eval_metadata=_external_eval_metadata(
            case_id=case["id"],
            vertical="retrieval",
            origin_benchmark="frames",
            difficulty=str(case.get("difficulty", "hard")),
            tags=["curated", "hard", "retrieval", *case.get("axes", [])],
        ),
    )


def run_nexus_case(case: dict[str, Any], model: BaseChatModel) -> None:
    """Run a Nexus case with file-backed reasoning and text scoring.

    Args:
        case: A curated Nexus case dict with keys `id`, `prompt`, `files`,
            and `answer_snippets`.
        model: The chat model to use for the agent.
    """
    agent = _create_file_backed_agent(model)
    run_agent(
        agent,
        model=model,
        query=case["prompt"],
        initial_files=case["files"],
        scorer=_create_text_scorer(case),
        eval_metadata=_external_eval_metadata(
            case_id=case["id"],
            vertical="reasoning",
            origin_benchmark="nexus",
            difficulty=str(case.get("difficulty", "hard")),
            tags=["curated", "hard", "reasoning", *case.get("axes", [])],
        ),
    )


# ---------------------------------------------------------------------------
# BFCL v3: live stateful tools
# ---------------------------------------------------------------------------

_BFCL_CLASS_REGISTRY: dict[str, type] = {
    "VehicleControlAPI": VehicleControlAPI,
    "MessageAPI": MessageAPI,
    "TradingBot": TradingBot,
    "TravelAPI": TravelAPI,
    "TicketAPI": TicketAPI,
}

_BFCL_SYSTEM_PROMPT = (
    "You are an assistant with access to domain-specific API tools. "
    "Use these tools to accomplish the user's requests. "
    "Do not use the task tool or any subagent delegation. "
    "Do not use file tools (ls, read_file, write_file, etc.)."
)


def _instantiate_bfcl_apis(case: dict[str, Any]) -> dict[str, Any]:
    """Create and load BFCL API instances from case config."""
    instances: dict[str, Any] = {}
    for class_name in case["involved_classes"]:
        cls = _BFCL_CLASS_REGISTRY[class_name]
        instance = cls()
        config = copy.deepcopy(case["initial_config"].get(class_name, {}))
        instance._load_scenario(config, long_context=False)
        instances[class_name] = instance
    return instances


def _wrap_bfcl_methods_as_tools(instances: dict[str, Any]) -> list[StructuredTool]:
    """Wrap all public methods of BFCL API instances as StructuredTools."""
    tools: list[StructuredTool] = []
    for instance in instances.values():
        for method_name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
            if method_name.startswith("_"):
                continue
            tools.append(
                StructuredTool.from_function(
                    func=method,
                    name=method_name,
                    description=(method.__doc__ or "").strip(),
                )
            )
    return tools


def _fix_bfcl_gt_call(call_str: str) -> str:
    """Fix known issues in BFCL ground truth call strings."""
    # Strip sender_id kwarg (not in MessageAPI.send_message signature).
    # Handle both middle position (trailing comma) and last position (leading comma).
    # Current data always has sender_id first, but future cases could have it last.
    call_str = re.sub(r",\s*sender_id=['\"][^'\"]*['\"](?=\s*\))", "", call_str)  # last arg
    return re.sub(r"sender_id=['\"][^'\"]*['\"],\s*", "", call_str)  # first/middle arg


def _replay_bfcl_ground_truth(case: dict[str, Any]) -> dict[str, Any]:
    """Replay ground truth calls on fresh API instances."""
    gt_instances = _instantiate_bfcl_apis(case)
    methods: dict[str, Any] = {
        name: method
        for instance in gt_instances.values()
        for name, method in inspect.getmembers(instance, predicate=inspect.ismethod)
        if not name.startswith("_")
    }
    for turn_gt in case["ground_truth"]:
        for call_str in turn_gt:
            # Suppress broadly: GT data has multiple known issues including
            # missing methods (shell commands like cat/ls), wrong kwargs
            # (e.g. travel_cost in book_flight), and extra params (sender_id).
            with contextlib.suppress(Exception):
                eval(_fix_bfcl_gt_call(call_str), {"__builtins__": {}}, methods)  # noqa: S307  # safe: builtins disabled, locals restricted to API methods, input is vendored benchmark data
    return gt_instances


def _bfcl_state_diff(
    model_instances: dict[str, Any],
    gt_instances: dict[str, Any],
    case: dict[str, Any],
) -> str:
    """Return human-readable diff of model vs ground-truth API state."""
    diffs: list[str] = []
    for class_name in case["involved_classes"]:
        model_inst = model_instances[class_name]
        gt_inst = gt_instances[class_name]
        for attr_name in vars(gt_inst):
            if attr_name.startswith("_"):
                continue
            model_val = getattr(model_inst, attr_name)
            gt_val = getattr(gt_inst, attr_name)
            if model_val != gt_val:
                diffs.append(f"  {class_name}.{attr_name}: got={model_val!r}, expected={gt_val!r}")
    return "\n".join(diffs)


def run_bfcl_case(case: dict[str, Any], model: BaseChatModel) -> None:
    """Run a BFCL v3 case with live stateful API tools."""
    model_instances = _instantiate_bfcl_apis(case)
    tools = _wrap_bfcl_methods_as_tools(model_instances)

    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=_BFCL_SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
    )

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    t.log_inputs(
        {
            "case_id": case["id"],
            "category": case["category"],
            "num_turns": case["num_turns"],
            "involved_classes": case["involved_classes"],
            "model": str(getattr(model, "model", None) or getattr(model, "model_name", "")),
            "eval_metadata": _external_eval_metadata(
                case_id=case["id"],
                vertical="tool_calling",
                origin_benchmark="bfcl",
                difficulty=str(case.get("difficulty", "hard")),
                tags=["curated", "hard", "tool_calling", *case.get("axes", [])],
            ),
        }
    )

    # Multi-turn conversation
    result: dict[str, Any] | None = None
    try:
        for turn_messages in case["conversation"]:
            if turn_messages:
                invoke_inputs: dict[str, Any] = {"messages": turn_messages}
            else:
                invoke_inputs = {"messages": [{"role": "user", "content": "Please continue and complete any remaining tasks."}]}
            result = agent.invoke(invoke_inputs, config)
    except Exception:
        # Log explicit correctness=0 so invoke failures appear in LangSmith
        # dashboards rather than as missing data points.
        t.log_feedback(key="correctness", value=0)
        raise

    if result is None:
        msg = f"No conversation turns in BFCL case {case['id']}"
        raise ValueError(msg)

    t.log_outputs(result)

    # Score via state comparison
    gt_instances = _replay_bfcl_ground_truth(case)
    diff = _bfcl_state_diff(model_instances, gt_instances, case)

    if diff:
        t.log_feedback(key="correctness", value=0)
        pytest.fail(f"BFCL state mismatch for {case['id']}:\n{diff}", pytrace=False)
    else:
        t.log_feedback(key="correctness", value=1)
