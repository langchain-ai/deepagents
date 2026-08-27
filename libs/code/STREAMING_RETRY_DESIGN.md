# Retrying interrupted model streams

## Status

Proposed design for retrying transient model failures after a response has already
streamed visible output. This document covers the main agent model node in both
interactive and non-interactive clients. It does not change auxiliary model calls,
which do not expose their output directly.

## Motivation

The current middleware retries only if the failed call emitted no message-stream
chunk. That avoids duplicate terminal output, but converts a common provider
failure mode into a terminal turn error: a connection can drop after useful text
has arrived but before the provider sends its completion event.

The existing rule treats presentation limitations as a model reliability policy.
Those concerns should be separated:

1. The server decides whether a model failure is transient and safe to rerun.
2. The client represents output from the failed attempt honestly on its surface.

A plain terminal cannot retract arbitrary bytes, but that should not prevent the
server from recovering the model call.

## Goals

- Retry transient read, connection, rate-limit, and provider failures even after
  streaming starts.
- Keep retry scope at the model node so completed tools are never replayed.
- Keep failed partial output out of graph state and future model context.
- Make attempt boundaries explicit to local, remote, interactive, and headless
  consumers.
- Preserve live token streaming in the interactive client and default headless
  mode.
- Keep tool hooks balanced when a failed attempt streamed a tool call that never
  executed.
- Preserve the existing retry budget, backoff, `Retry-After`, cancellation, and
  error-classification behavior.
- Remain compatible with clients and servers that do not yet understand attempt
  lifecycle events.

## Non-goals

- Token-exact provider stream resumption. Generic LangChain chat integrations do
  not expose a portable cursor or response sequence ID.
- Deduplicating semantically repeated text across attempts.
- Continuing an incomplete tool call.
- Retrying authentication, invalid-request, context-window, or other permanent
  failures.
- Hiding the fact that a failed request may still be billed by the provider.
- Changing provider SDK retry ownership or retry-count configuration.

## Current architecture

`CodeModelRetryMiddleware` wraps the model handler, not the whole agent turn.
`_retry_call` and `_aretry_call` classify errors, enforce the retry budget, emit a
status event, wait, and invoke the same handler again.

For streaming calls, `_MessageStreamTracker` replaces LangGraph's
`StreamMessagesHandler` callbacks with forwarding wrappers. It sets
`has_streamed` before forwarding a chunk because a writer can fail after putting
that chunk beyond server control. At the end of an attempt, `merge_seen()` copies
LangGraph's de-duplication IDs back to the original handlers so the graph does not
re-emit a completed message.

`_allow_retry_after_stream` currently refuses a retry when both of these are
true:

- the tracker observed a message chunk; and
- `stream_output_is_visible` is true for that middleware instance.

The main rubric grader explicitly uses `stream_output_is_visible=False` because
both clients filter its nested output, so that instance already retries
interrupted streams. Other hidden criteria and fallback agents currently use the
default `True`; implementation must audit those call sites and set `False` only
where both local and remote clients provably filter the stream.

Before a retry, the server emits this ephemeral custom-stream payload:

```json
{
  "type": "model_retry",
  "attempt": 1,
  "max_retries": 5,
  "message": "Retrying model request 1/5"
}
```

Both clients consume `messages`, `updates`, and `custom` stream modes. The TUI
turns `model_retry` into spinner text. The headless client prints a status line.
The event is not graph state and is not checkpointed.

### State boundary

A failed model handler does not return a `ModelResponse`, so the model node does
not commit its partial `AIMessage` to graph state. The retried handler receives
the same request messages and tools. Only the successful response is committed.

Message chunks may already exist in client presentation state, hook transcript
staging, and usage accounting. Those are the reconciliation targets; graph state
does not need rollback.

## Design choices

The client cannot reliably infer which chunks belong to which model attempt. A
single graph run can include the root model, hidden model calls, and nested agents;
provider message IDs may be reused; and a failed attempt never produces a graph
state update. The server therefore needs to mark attempt boundaries before
post-emission retries can be enabled.

There are two viable presentation policies once those boundaries exist.

### Preserve, annotate, and replay

The client finalizes whatever escaped from the failed attempt, labels it
incomplete, and renders the replay as a separate response.

Advantages:

- no arbitrary deletion API or DOM rollback is required;
- behavior matches irreversible stdout and degrades safely for older clients;
- hook settlement and transcript cleanup are monotonic operations; and
- event loss leaves an untidy but inspectable transcript rather than corrupting
  unrelated presentation state.

Disadvantages:

- users can see an orphaned fragment followed by repeated text; and
- append-only consumers may concatenate two independent generations unless they
  understand the boundary.

Codex and OpenCode choose variants of this availability-first tradeoff.

### Roll back and replay

The client removes every artifact owned by the failed attempt before rendering the
replay.

Advantages:

- the final TUI transcript contains only the successful generation; and
- repeated prefixes are less distracting when rollback succeeds.

Disadvantages:

- every assistant widget, timestamp, tool row, pending text buffer, tool hook, and
  transcript chunk needs attempt ownership;
- `MessageStore` needs atomic arbitrary deletion that cooperates with
  virtualization, active-message protection, and grouping;
- terminal stdout and already consumed remote events remain irreversible anyway;
  and
- partial event loss can delete the wrong UI state unless correlation and ordering
  are exact.

Crush and Kimi choose variants of this policy where their active render state is
replaceable.

### Recommendation

Use **preserve, annotate, and replay** for the first release. It is the smallest
safe policy shared by the TUI, streaming headless output, and remote clients. In
buffered headless mode, discard failed attempt output because it has not escaped;
that is commit-or-discard staging, not user-visible rollback.

A transient failure remains retryable after output may have started. The retry
reruns only the failed model call from the last committed graph state and never
reruns a completed tool. The model does not receive the partial response on
retry. Prompt-level continuation and provider-native resume remain future
optimizations.

## Attempt lifecycle protocol

Add one new custom event and extend the existing retry event. These events are
presentation protocol, not model context or durable graph state. The retry-loop
closure owns one bounded, opaque `call_id`; per-attempt stream trackers do not.

### Attempt start

Emitted immediately before each model handler invocation:

```json
{
  "type": "model_attempt",
  "phase": "start",
  "call_id": "server-generated opaque ID",
  "attempt": 0
}
```

`attempt` is zero for the initial call and increments for each retry. `call_id`
is stable across all attempts of one middleware invocation. The enclosing
LangGraph stream namespace scopes the event to the main agent or a subgraph.

Consumers use this event to open an attempt-local presentation and transcript
staging scope. It produces no visible UI by itself.

### Attempt complete

Emitted after the handler returns successfully and before the middleware returns
the response:

```json
{
  "type": "model_attempt",
  "phase": "complete",
  "call_id": "same opaque ID",
  "attempt": 0
}
```

Consumers commit attempt-local transcript staging and clear supersession metadata.
The graph still owns the authoritative state commit.

### Retry

The existing event keeps all current fields and adds correlation metadata:

```json
{
  "type": "model_retry",
  "attempt": 1,
  "max_retries": 5,
  "message": "Retrying model request 1/5",
  "call_id": "same opaque ID",
  "failed_attempt": 0,
  "output_may_have_started": true
}
```

The existing `attempt` field retains its meaning: the one-indexed retry number
that is about to run. `failed_attempt` is the zero-indexed attempt being
superseded. `output_may_have_started` is true when the tracker tried to forward a
message chunk from the failed attempt and the stream was configured as visible.
The tracker flags before forwarding because a writer can fail after partially
putting a chunk beyond server control. The server therefore cannot promise that a
consumer observed the chunk; conservative reconciliation is intentional.

The event is emitted after the failure is classified and before backoff. A
failed status writer never blocks the retry; consumers fall back to append-only
behavior if lifecycle information is missing.

### Ordering

For one model call, the observable order is:

```text
model_attempt(start, attempt=0)
message chunks for attempt 0
model_retry(failed_attempt=0, attempt=1)
model_attempt(start, attempt=1)
message chunks for attempt 1
model_attempt(complete, attempt=1)
```

This is an observational ordering contract, not a rollback gate: message chunks
may have updated a widget, fired a hook, or reached stdout before `model_retry`
arrives. The client must finish reconciliation before processing the following
`model_attempt(start)` or its message chunks.

If cancellation wins during backoff after `model_retry`, ordinary stream teardown
clears transient retry status and keeps any persistent incomplete marker. No next
attempt starts. Permanent errors and exhausted retry budgets do not emit
`model_retry`; existing stream-finalization paths settle the failed attempt.

### Compatibility

- An old client ignores `model_attempt` and handles the unchanged fields in
  `model_retry`. It may append regenerated output after the partial response,
  but it does not crash.
- A new client connected to an old server sees no lifecycle events. The old
  server still refuses post-output retries, so there is no reconciliation to
  perform.
- Event parsers treat unknown phases and fields as no-ops. They never trust
  provider error text or arbitrary values from the stream.

## Retry semantics

On a retryable exception:

1. Preserve the original exception for logs and final re-raise.
2. Compute the delay once, including jitter or a valid `Retry-After` value.
3. Emit `model_retry` with the failed tracker's `has_streamed` value as
   `output_may_have_started`, gated by `stream_output_is_visible`.
4. Let clients reconcile the failed attempt.
5. Wait using the existing bounded backoff.
6. Emit the next start event, create a fresh tracker, and invoke the same model
   handler again.
7. Continue merging `seen` IDs from every attempt into the original LangGraph
   stream handlers.

The retry does not add the partial assistant output or a continuation instruction
to `request.messages`. It does not change tools, model selection, or response
schema.

### Why replay is safe at this boundary

The middleware wraps only the model node. A tool emitted by an incomplete model
response cannot have run in the following tool node because the model handler has
not returned. Completed tools from earlier graph steps are already represented in
`request.messages`; replaying the current model call uses their results without
executing them again.

A streamed tool call can still have client-side presentation and hook effects.
Those must be settled as described below.

## Surface behavior

### Interactive TUI

When `model_retry.output_may_have_started` is true for the root namespace, the
Textual adapter:

1. Stops the active `MarkdownStream` and syncs its current content to
   `MessageStore`.
2. Clears the active assistant widget and pending-text references for that
   namespace so the retry creates a new assistant response.
3. Settles any streamed tool-call rows from the failed attempt as
   `Model response interrupted before tool execution`.
4. Clears incomplete tool argument buffers from the failed attempt.
5. Mounts a persistent, subdued status row:
   `Connection dropped; the partial response above is incomplete. Retrying 1/5.`
6. Keeps the existing retry spinner active during backoff.

The partial response remains visible. It is not represented as a successful
assistant turn when the thread is resumed because it never entered graph state.
The persistent marker explains that difference within the live session.

The first implementation deliberately does not remove rows from `MessageStore`.
Deletion must coordinate the DOM widget, timestamp footer, virtualization index,
active-message protection, tool grouping, and hook audit trail. Preserving and
marking the fragment is more reliable and matches surfaces that cannot retract
output.

A later TUI-only preference may collapse or hide superseded attempts, but the
protocol semantics remain supersession rather than deletion.

### Non-interactive, streaming

Text can reach stdout before the retry event that identifies its attempt as
failed, and once written it cannot be recalled. On a post-output retry:

- terminate the current line if necessary;
- print a retry boundary before regenerated output;
- reset attempt-local tool-call buffers; and
- close any emitted `tool.use` hook as an error before accepting the retry's tool
  calls.

In normal verbose mode, the boundary is visible in the same terminal stream. In
`--quiet` mode, diagnostics remain on stderr so stdout stays text-only. This means
a consumer that ignores stderr can see both the partial and regenerated text.
Callers requiring exactly one clean response must use `--no-stream` or a future
structured streaming format.

This limitation is explicit rather than a reason to disable recovery globally.

### Non-interactive, buffered

With `--no-stream`, assistant text remains client-side until the run ends. The
attempt lifecycle gives the client an offset for each attempt:

- `model_attempt(start)` records the current buffer length;
- `model_retry` truncates to that offset and clears attempt-local tool data; and
- `model_attempt(complete)` commits the staged segment.

Tool-call status is currently printed as soon as a name is parsed, even under
`--no-stream`. Buffered mode must stage that status with the attempt as well;
otherwise it cannot promise clean output. After that change, only successful model
output and tool status reach stdout. Earlier successful model steps in the same
agent run remain in the buffer.

### Structured consumers

There is no streaming JSON renderer today. A future renderer should expose the
lifecycle events directly rather than inventing a second retry protocol. A
consumer can then mark prior deltas as superseded, replace them in a UI, or retain
them for audit.

### Remote and nested execution

Remote clients receive the same custom events through `RemoteGraph.astream`.
They must correlate by stream namespace, `call_id`, and `attempt`, not by arrival
time alone. Duplicate lifecycle events are idempotent. A retry event without a
known start event degrades to a visible retry marker and append-only output.

The current clients intentionally do not render nested-agent message chunks. They
must still consume nested lifecycle events for transcript staging, usage identity,
and hook cleanup, but must not mutate the root assistant widget or print a root
retry boundary. A nested retry is visible only through the existing Task/subagent
status surface. The supersession guarantee applies to output a surface chose to
render; filtered nested text needs no chat marker.

## Tool calls and hooks

A tool call whose arguments fully parse can mount a client widget and dispatch
`tool.use` before the model response completes. If that model attempt fails, the
tool did not execute, but the hook must not remain open.

For every such call, the client dispatches:

```text
tool.error: Model response interrupted before tool execution
```

The TUI uses the same terminal settlement machinery as cancellation and rejected
approval. The headless client uses its orphaned-tool drain. Both surfaces retain
the existing invariant that every emitted `tool.use` has exactly one terminal
`tool.result` or `tool.error`.

Incomplete argument buffers that never emitted `tool.use` are dropped without a
terminal hook. Tool-call IDs remain fire-once within the run. A retry is expected
to produce new provider IDs; if a provider reuses an ID, the monotonic ID sets
prevent duplicate hook dispatch.

Prompt-level continuation is forbidden when any tool-call fragment was observed.
There is no portable way to continue partially generated JSON or preserve the
provider's required function-call ordering.

## Transcript behavior

`TranscriptRecorder` currently combines `BaseMessageChunk` values and appends a
record as soon as it sees the last chunk. Most interrupted streams therefore
leave only an in-memory partial chunk, but a provider can emit a final-marked chunk
and then fail while closing the transport. That message would be materialized even
though the model node never committed it.

The stream consumer, which sees both `messages` and `custom` modes, must coordinate
attempt lifecycle with the recorder. The recorder gains attempt-scoped methods and
stages even final-marked messages until completion:

- `start_attempt(namespace, call_id, attempt)` opens a staging scope;
- `record(...)` combines chunks inside that scope without appending them;
- `complete_attempt(...)` appends completed staged messages;
- `discard_attempt(...)` drops the failed scope on `model_retry`; and
- terminal stream cleanup drops every uncommitted scope.

The non-interactive stream loop calls these methods directly. The TUI either calls
the same API where it already processes custom events or moves transcript routing
to a small shared stream coordinator; `TranscriptRecorder` does not consume custom
events by itself.

Checkpoint-derived messages remain authoritative when a thread is resumed.
Lifecycle events themselves are not written to the hook transcript.

## Usage and cost accounting

A failed request may be billable. Superseding its text must not roll back usage.
Usage attached to streamed chunks is already recorded before render filtering; the
new attempt identity must prevent a retry that reuses a provider message ID from
revising the failed request in place.

- Key provisional request accounting by `(namespace, call_id, attempt,
  provider_message_id)` rather than the provider message ID alone.
- At `model_retry`, finalize known usage for the failed attempt and start a new
  accounting scope. Do not retract it when presentation is discarded.
- Record every usage report the provider actually supplies, including reports
  attached to chunks from a failed attempt.
- If a dropped stream supplied no usage metadata, exact provider spend is unknown.
  Count it only in an explicit estimated-request metric if one is added; do not
  invent tokens or cost, and surface that session totals may be incomplete.
- Session totals include known usage from successful and failed attempts because
  they represent spend, not accepted transcript content.

This requires changing `recorded_usage_requests` from a message-ID-only map to an
attempt-scoped ledger. It is not solved merely by clearing the current map: doing
that would preserve prior totals but lose the ability to apply a late correction
to the right request.

The middleware cannot recover usage that a provider never sent. Obtaining exact
failed-request cost would require a provider billing API or server-side usage
event and is outside this protocol.

## Cancellation and failure

`GraphBubbleUp`, user cancellation, and task interruption bypass retry exactly as
they do today. Attempt lifecycle cleanup may finalize presentation state, but it
must not emit another model request.

When the retry budget is exhausted after partial output:

- preserve and mark the final partial response;
- settle orphaned tool hooks;
- surface the original provider exception; and
- leave graph state at the last committed boundary.

A failure while rendering retry metadata is logged and ignored. Presentation
failure cannot prevent model recovery.

## Security and resource limits

- Lifecycle payloads contain generated identifiers, bounded counters, and static
  status text only. Raw provider messages, response bodies, headers, prompts, and
  exception strings never cross this channel.
- Clients render only static local labels and validated counters. They strip or
  escape terminal control characters and markup from every untrusted field.
- Consumers validate counters, phases, booleans, and identifier lengths before
  use. Unknown or malformed events degrade to the existing generic retry status.
- Existing retry limits, maximum `Retry-After`, and backoff caps remain the
  resource bounds. Attempt staging is bounded by the same model-output limits as
  the response itself and is released on retry, completion, cancellation, or
  stream teardown.
- Logs use structured fields or parameterized messages. Detailed exceptions stay
  in debug logs and are not interpolated into terminal markup.

## Alternatives considered

### Keep refusing retries after the first chunk

This is the current behavior. It guarantees no duplicate visible text but makes
mid-stream provider drops unrecoverable. It is too conservative for a coding
agent expected to survive transient network faults.

### Buffer every model attempt until success

This produces perfectly clean output on all surfaces and was previously prototyped
in the middleware. It also removes live token streaming, delays visible progress,
and makes a healthy request look frozen until completion. `--no-stream` already
provides this tradeoff for callers that want it.

### Delete partial output in the TUI

Crush and Kimi take this approach for active UI state. Dcode can add it later, but
making deletion the correctness mechanism is fragile: terminal scrollback may
already contain the text, remote consumers cannot be recalled, and Textual
removal must update virtualization, timestamps, grouping, and hooks atomically.
A persistent supersession marker is a safer first contract.

### Append the retry to the same assistant widget

This is simple but visually claims that two independent generations are one
response. It can splice duplicate prefixes together and hides the retry boundary.
The retry must start a separate assistant response.

### Ask the model to continue from the partial response

Qwen Code, Vibe, and Hermes use continuation for some text-only failures. It can
save tokens and avoid a repeated prefix, but it changes the prompt, can repeat or
skip text, and becomes invalid around tool calls. It may be explored later as a
bounded text-only optimization after replay semantics are reliable.

### Resume the provider stream

Letta can reconnect to a server-persisted run by sequence ID. This is preferable
when a provider exposes a durable response ID and cursor, but it is not available
across dcode's generic model integrations. Provider-specific resume can sit below
this design and suppress a full replay when it succeeds.

## Prior art

- **Codex:** retries streams that end before `response.completed`, preserves
  already rendered partial output, rebuilds from completed history, and can fall
  back from WebSocket to HTTPS after exhausting a transport budget.
- **OpenCode:** wraps the complete stream drain in a retry policy. Partial deltas
  remain persisted as message parts and the retried generation creates another
  part on the same assistant message.
- **Crush and Kimi:** reset the active streamed attempt before replay. Kimi
  buffers print-mode output, so a failed attempt does not reach headless stdout.
- **Qwen Code, Vibe, and Hermes:** preserve partial text and use a continuation
  prompt in selected cases, with restrictions around tool calls.
- **Letta:** resumes the same server-side run from a sequence cursor.
- **Whip:** explicitly refuses retries after the first visible delta.

The common pattern is that transport resilience and presentation reconciliation
are separate decisions. Most peers accept retry after emission; there is no
single cross-surface way to retract output.

## Implementation plan

### Phase 1: protocol and server retry

1. Add `model_attempt` start/complete builders and parsers beside the existing
   retry helpers.
2. Generate one bounded opaque `call_id` in the retry-loop closure per middleware
   invocation.
3. Emit start and complete events around each handler attempt.
4. Add retry correlation fields and `output_may_have_started` to `model_retry`.
5. Replace `_allow_retry_after_stream` with presentation metadata; retain
   `stream_output_is_visible` for intentionally filtered nested streams and audit
   every hidden criteria/grader middleware call site for the correct value.
6. Preserve `merge_seen()` and all existing error, delay, cancellation, and
   retry-budget behavior.

### Phase 2: client reconciliation

1. Add attempt-local state to the Textual adapter and `StreamState`.
2. Finalize the interrupted assistant widget and start the retry in a new widget.
3. Add the persistent TUI and headless retry boundaries.
4. Truncate only the failed segment and discard staged tool status in buffered
   headless mode.
5. Reset tool-call buffers and settle emitted tool hooks on retry.
6. Make lifecycle handling idempotent and namespace-aware.

### Phase 3: transcript and accounting

1. Add attempt-scoped staging APIs to `TranscriptRecorder` and route custom events
   to them from the stream consumers.
2. Include attempt identity in provisional usage keys and finalize known usage at
   retry boundaries without subtracting spend.
3. Verify local and remote execution produce equivalent lifecycle ordering.

These phases may land in one pull request, but each phase should be a reviewable
commit with focused tests.

## Test plan

### Middleware

- A transient error before output preserves the existing retry sequence.
- A transient error after text output retries and emits correlated lifecycle
  events in the documented order.
- Sync and async paths behave identically.
- Permanent errors, cancellation, `GraphBubbleUp`, and exhausted budgets do not
  start another attempt.
- Every audited hidden criteria/grader call retries without visible supersession
  metadata; visible or uncertain calls use the ordinary boundary.
- LangGraph v1 and v2 stream-handler de-duplication IDs survive every attempt.
- A failed custom-stream writer does not stop retry.

### Interactive client

- Post-output retry finalizes the partial assistant widget and creates a new one
  for regenerated output.
- The persistent interruption row remains after the spinner clears.
- Text before a streamed tool call is not joined to retry text.
- Parsed tool calls settle with one `tool.error`; incomplete calls emit no hook.
- Duplicate and out-of-order lifecycle events are safe no-ops.
- Subagent retry events do not mutate the root assistant widget.
- Interrupt cleanup and retry cleanup cannot leave protected or running widgets.

### Non-interactive client

- Streaming output contains a visible boundary between partial and replayed text.
- `--quiet` keeps diagnostics on stderr and text on stdout.
- `--no-stream` removes failed text and staged tool status while preserving
  earlier successful model steps.
- Tool hooks remain balanced on retry and terminal failure.

### Transcript and usage

- A failed partial attempt never reaches the materialized transcript.
- A final-marked chunk followed by a transport error is discarded unless the
  middleware emits attempt completion.
- Successful retry output appears once in the transcript.
- Known usage from failed and successful attempts is counted once each.
- A failed attempt with no provider usage is reported as unknown, not exact spend.
- Reused provider message IDs do not merge accounting across attempts.

### End-to-end

Use a controllable streaming model that emits text, optionally emits tool-call
fragments, raises a retryable read error, and then succeeds. Exercise local TUI,
local headless streaming, local buffered headless, and the remote graph path.

## Success criteria

- A mid-stream transient provider drop no longer ends the turn while retry budget
  remains.
- No completed tool executes more than once because of model retry.
- Only the successful model response enters graph state and resumed history.
- Every surface that rendered failed partial output identifies it as superseded.
- Buffered output contains no failed-attempt text or tool status.
- Hook invariants hold and known usage stays separated across retries.
- Pre-output retry behavior and latency remain unchanged.
