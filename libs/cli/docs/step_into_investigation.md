# Step-Into Feature: Deep Investigation

Investigation date: 2026-03-07
Branch: feature/step-into-subagents-v2

## Confirmed Bugs

### BUG-1: Parent thread HITL interrupt never resolved (CRITICAL)

**What happens:**
When the user selects "Step into", `execute_task_textual` returns at line 1041
WITHOUT sending `Command(resume=...)` to LangGraph. The parent thread's `task`
tool call interrupt is left dangling forever.

When `/return` fires, it injects the summary via `_handle_user_message()` which
sends a new `HumanMessage` to the parent thread. But LangGraph expects
`Command(resume=...)` for the pending interrupt, not a new message. The parent
agent cannot continue properly.

**Evidence:**
- `textual_adapter.py:1041` - returns without resume
- `app.py:1625` - injects summary as regular HumanMessage
- `config.py:769` - `parent_tool_call_id` field EXISTS but is always None
- `textual_adapter.py:268` - `_create_branch_context` hardcodes `parent_tool_call_id=None`

**The field was designed for this but never wired up.**

**Fix approach:**
1. Capture interrupt_id when user steps into (store in ConversationContext)
2. On `/return`, send `Command(resume={interrupt_id: ApproveDecision + summary})`
   instead of a new HumanMessage

---

### BUG-2: Context not popped on subagent error (HIGH)

**What happens:**
In `_run_agent_task` (app.py:2098-2156):
1. Context is pushed at line 2122 (before subagent call)
2. Subagent call at line 2139 throws an exception
3. Exception is caught at line 2148 but context is NEVER popped
4. User is stuck in a dead subagent context

**Evidence:**
- `app.py:2120-2146` - push happens before the call
- `app.py:2148-2156` - except/finally blocks don't pop context
- `config.py:841-853` - pop_context() is the only way to remove a context

**Recovery:**
- `/return` DOES work (pops the stuck context) - it's a manual recovery path
- `/clear` resets everything but loses session state

**Fix approach:**
Add context-pop in the exception handler when step_into_context was pushed.

---

### BUG-3: Nested step-into result discarded (CRITICAL)

**What happens:**
In `_run_agent_task` (app.py:2139-2146), the second `execute_task_textual` call
discards its return value:

```python
await execute_task_textual(...)  # result NOT captured
```

If the subagent launches another Task tool call and the user steps into it,
`execute_task_textual` returns `ExecuteTaskResult(step_into_context=...)` but
this result is LOST. The nested context is never pushed.

**Evidence:**
- `app.py:2108` - first call captures result: `result = await ...`
- `app.py:2139` - second call discards result: `await ...`

**Fix approach:**
Loop on the result - if step_into_context is returned, push and recurse.

---

### BUG-4: Auto-approve leaks between contexts (HIGH)

**What happens:**
`session_state.auto_approve` is a single shared boolean across all contexts.
Toggling auto-approve (Ctrl+T) inside a subagent changes it globally.
When the user returns to root, auto-approve is still on.

**Evidence:**
- `config.py:797` - single `self.auto_approve` flag
- `textual_adapter.py:944` - reads `session_state.auto_approve`
- `textual_adapter.py:974` - writes `session_state.auto_approve = True`
- `ConversationContext` has no `auto_approve` field

**Fix approach:**
Either:
- Add `auto_approve` to ConversationContext and save/restore on push/pop
- Or scope auto-approve per context via a dict keyed by depth

---

### BUG-5: Summary files never cleaned up (LOW)

**What happens:**
Branch directories are created at `~/.deepagents/<agent>/branches/<id>/summary.md`
but never deleted. They accumulate indefinitely.

**Evidence:**
- `textual_adapter.py:247-261` - creates directory and file
- `app.py:1606-1625` - reads summary on /return but doesn't delete
- No `unlink()` or `rmtree()` anywhere in the codebase for branches

**Fix approach:**
Delete branch directory on `/return` after reading the summary.

---

## False Alarms

### /return while agent running: NOT A BUG
All input (including slash commands) is queued when `_agent_running == True`
(app.py:1128-1134). Commands execute in order after the agent completes.

### Thread ID management: WORKING CORRECTLY
`session_state.thread_id` is a property that returns `current_context.thread_id`,
so it always reflects the topmost context. Each context gets a unique thread_id.

### Multiple /return at root: HANDLED
Guard clause at app.py:1600 checks `depth == 0` and shows a friendly message.

---

## Priority Order for Fixes

1. BUG-1 (parent interrupt) - blocks the entire step-into/return lifecycle
2. BUG-3 (nested step-into) - prevents multi-level workflows
3. BUG-2 (context not popped on error) - leaves user in broken state
4. BUG-4 (auto-approve leak) - security concern
5. BUG-5 (summary cleanup) - disk space, low priority
