---
name: deepagents-thread-inspector
description: Inspect and explain conversations stored in the local Deep Agents Code SQLite session store. Use when asked to identify a dcode thread, summarize what happened in a thread or its latest turn, recover conversation/tool activity, inspect checkpoint metadata, list recent local threads, or parse ~/.deepagents/.state/sessions.db and a thread UUID or prefix.
license: MIT
compatibility: designed for deepagents-code
---

# Deep Agents Thread Inspector

Use `scripts/inspect_sessions.py` instead of manually decoding database blobs. It opens the database read-only and deserializes the root message channel with LangGraph's strict MsgPack loader — reading the materialized messages from the latest checkpoint, or replaying writes in checkpoint order when that fast path is unavailable — and emits JSON.

## Inspect a thread

Resolve `SKILL_DIR` to the directory containing this `SKILL.md`; do not assume a user, project, or installation-specific location. Start with the smallest useful view:

```bash
python3 "$SKILL_DIR/scripts/inspect_sessions.py" THREAD_ID --mode latest-turn
```

A unique thread-ID prefix is accepted. Select another view when needed:

```bash
python3 "$SKILL_DIR/scripts/inspect_sessions.py" THREAD_ID --mode summary
python3 "$SKILL_DIR/scripts/inspect_sessions.py" THREAD_ID --mode transcript
```

Use `--include-metadata` only when run, repository, model, checkpoint, or LangGraph metadata matters. Use `--max-content N` to raise or lower the default 4,000-character limit per message, tool result, or tool-call argument.

If the user does not know the ID, list recent threads first:

```bash
python3 "$SKILL_DIR/scripts/inspect_sessions.py" --list 20
```

Pass `--db PATH` only for a non-default session store. The default is `~/.deepagents/.state/sessions.db`; `DEEPAGENTS_SESSIONS_DB` can override it.

## Explain the result

Synthesize the JSON rather than pasting it verbatim.

- State the user's request, the assistant's conclusion, and significant tool actions or failures.
- Distinguish stored facts from your interpretation.
- For the latest turn, describe only the final user message and subsequent activity unless earlier context is required to make it understandable.
- Mention truncation when a relevant record has `content_truncated` or `args_truncated` set.
- Surface reconstruction problems when the result includes a top-level `warnings` array (for example, a corrupt checkpoint, a skipped write, or malformed metadata) so conclusions are appropriately hedged.
- Do not expose unrelated credentials, tokens, personal data, or hidden reasoning that may appear in local records.

## Safety

Keep inspection read-only. Do not deserialize an untrusted database: checkpoint deserialization is intended for trusted local Deep Agents state. Do not mutate or delete session rows unless the user separately and explicitly requests it.
