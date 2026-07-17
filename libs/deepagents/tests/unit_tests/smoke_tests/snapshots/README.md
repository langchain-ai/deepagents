# System prompt smoke snapshots

These files are golden snapshots for the prompts and tool schemas produced by `tests/unit_tests/smoke_tests/test_system_prompt.py`.

Each snapshot is self-contained: the tests compare the complete rendered prompt or complete tool list for one agent configuration. A `_with_*` name means that scenario includes the named feature; it does not mean the file stores a diff against a shared base snapshot.

Use focused snapshots for individual prompt/tool features, and add combination snapshots only when features interact in prompt or tool rendering. A full powerset of every feature combination would mostly duplicate shared prompt text and make expected wording changes noisy.

## Files

| Snapshot | Shell `execute` tool? | What it covers |
| --- | --- | --- |
| `system_prompt_with_execute.md` | Yes | Full system prompt for a local shell backend. |
| `system_prompt_with_execute_tools.json` | Yes | Tool schemas for the local shell backend, including `execute`. |
| `system_prompt_with_media_extra_tools.json` | Yes | Tool schemas when the optional `[video]` extra is enabled; this mainly covers the video-aware `read_file` description and schema. |
| `system_prompt_without_execute.md` | No | Full system prompt for a filesystem-only backend. |
| `system_prompt_without_execute_tools.json` | No | Tool schemas for the filesystem-only backend, without `execute`. |
| `custom_system_message.md` | No | Full prompt when the caller passes a custom `system_prompt`. |
| `custom_system_message_tools.json` | No | Tool schemas for the custom-prompt setup; the custom prompt should not change the tool surface. |
| `system_prompt_with_routed_backend.md` | Yes | Full prompt for a `CompositeBackend` with host-mapped, non-virtual, and unmapped virtual routes. |
| `system_prompt_with_routed_backend_tools.json` | Yes | Tool schemas for the routed-backend setup, including `execute` from the local shell default backend. |
| `system_prompt_with_sandbox_default.md` | Sandbox-capable backend | Full prompt when the default backend is a remote/sandbox shell. Local filesystem routes should not be described as local shell-accessible host paths. |
| `system_prompt_with_sync_and_async_subagents.md` | No | Full prompt when both local/sync subagents and remote/async subagents are configured. |
| `system_prompt_with_sync_and_async_subagents_tools.json` | No | Tool schemas for the sync+async subagent setup, including async task-management tools. |
| `system_prompt_with_memory_and_skills.md` | No | Full prompt when memory files and skills directories are present. |
| `system_prompt_with_memory_and_skills_tools.json` | No | Tool schemas for the memory+skills setup; memory and skills should affect prompt context, not the core filesystem tool surface. |
