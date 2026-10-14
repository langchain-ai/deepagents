"""DeepAgents hooks for persisting model output and closing a completed Run.

Message conversion and OV writes belong to the injected OpenViking Writer.
The hooks only apply confirmed results to the checkpointed OVContext and
evaluate the existing commit policy at safe lifecycle points.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from openviking.integrations.langchain.ov_context.session_writer import (
    OVMessageWriter,
    WriteResult,
)
from openviking.integrations.langchain.ov_context.state import OVContext, OVPluginState

from commit_policy import CommitPolicy


class OVMiddleware(AgentMiddleware):
    """Apply confirmed Writer output and settle a normally completed Run."""

    state_schema = OVPluginState

    def __init__(
        self,
        *,
        writer: OVMessageWriter,
        session_client: Any,
        commit_policy: CommitPolicy,
    ) -> None:
        super().__init__()
        self.writer = writer
        self.session_client = session_client
        self.commit_policy = commit_policy

    async def aafter_model(
        self, state: Mapping[str, Any], runtime: Any
    ) -> dict[str, Any] | None:
        raw_context = state.get("ov_context")
        if raw_context is None:
            return None
        if not isinstance(raw_context, Mapping):
            raise TypeError(
                "state['ov_context'] must be an OVContext checkpoint mapping"
            )
        context = OVContext.from_state(raw_context)

        messages = list(state.get("messages") or ())
        end_index = len(messages)
        if end_index < context.write_index:
            raise ValueError(
                "state['messages'] was truncated after the OVContext write cursor"
            )
        if end_index == context.write_index:
            return None
        if context.run_id is None:
            raise RuntimeError("Cannot write messages without an active OVContext run")

        result = await self.writer.write(
            session_id=context.session_id,
            raw_messages=messages[context.write_index : end_index],
            write_index=context.write_index,
            run_id=context.run_id,
        )
        if not isinstance(result, WriteResult):
            raise TypeError("OVMessageWriter.write must return WriteResult")
        if result.write_index != end_index:
            raise ValueError("Writer must acknowledge the full message batch")

        context.append(result.model_messages)
        context.write_index = result.write_index

        # The current model response's tool calls run after this Hook returns.
        if result.model_messages and not getattr(
            result.model_messages[-1], "tool_calls", None
        ):
            session = await self.session_client.get_session(context.session_id)
            commit_result = await self.commit_policy.commit(
                session_id=context.session_id,
                pending_tokens=session["pending_tokens"],
                runs=context.completed_runs,
            )
            if commit_result.archived is True:
                context.completed_runs = 0
        return {"ov_context": context.to_state()}

    async def aafter_agent(
        self, state: Mapping[str, Any], runtime: Any
    ) -> dict[str, Any] | None:
        raw_context = state.get("ov_context")
        if raw_context is None:
            return None
        if not isinstance(raw_context, Mapping):
            raise TypeError(
                "state['ov_context'] must be an OVContext checkpoint mapping"
            )
        context = OVContext.from_state(raw_context)

        messages = list(state.get("messages") or ())
        end_index = len(messages)
        if end_index < context.write_index:
            raise ValueError(
                "state['messages'] was truncated after the OVContext write cursor"
            )
        changed = end_index > context.write_index
        if changed:
            if context.run_id is None:
                raise RuntimeError(
                    "Cannot write messages without an active OVContext run"
                )
            result = await self.writer.write(
                session_id=context.session_id,
                raw_messages=messages[context.write_index : end_index],
                write_index=context.write_index,
                run_id=context.run_id,
            )
            if not isinstance(result, WriteResult):
                raise TypeError("OVMessageWriter.write must return WriteResult")
            if result.write_index != end_index:
                raise ValueError("Writer must acknowledge the full message batch")
            context.append(result.model_messages)
            context.write_index = result.write_index

        run_id = context.run_id
        if not run_id:
            return {"ov_context": context.to_state()} if changed else None

        if run_id == context.last_completed_run:
            context.end_run()
            return {"ov_context": context.to_state()}

        context.completed_runs += 1
        context.last_completed_run = run_id

        session = await self.session_client.get_session(context.session_id)
        commit_result = await self.commit_policy.commit(
            session_id=context.session_id,
            pending_tokens=session["pending_tokens"],
            runs=context.completed_runs,
        )
        if commit_result.archived is True:
            context.completed_runs = 0
        context.end_run()
        return {"ov_context": context.to_state()}
