---
type: persistence architecture
title: State, Checkpoints, and Persistent Records
description: Explains Talon's distinct ownership boundaries for graph checkpoints, chat-scoped conversation archives and vector indexes, assistant-home records, and cron jobs, including recovery and partial-failure behavior.
tags: [talon, persistence, checkpoints, history, archives, scheduling, recovery]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-01T08:06:30.386Z
sources:
  - id: openwiki-source-6a038e6e1a11f450bcafce54
    resource: repo://libs/talon/deepagents_talon/__main__.py
  - id: openwiki-source-995d5d95882808a64071f617
    resource: repo://libs/talon/deepagents_talon/archive_saver.py
  - id: openwiki-source-81698d033a5726401d48b135
    resource: repo://libs/talon/deepagents_talon/config.py
  - id: openwiki-source-f55101eb12af3c6ae9b9d823
    resource: repo://libs/talon/deepagents_talon/cron/jobs.py
  - id: openwiki-source-470e982344d3fb19aa4cd0a7
    resource: repo://libs/talon/deepagents_talon/history_backends.py
  - id: openwiki-source-2318fb8a25701a5cdae717fe
    resource: repo://libs/talon/deepagents_talon/history_vector_backends.py
  - id: openwiki-source-6801a88de6305bc8cbdd259f
    resource: repo://libs/talon/deepagents_talon/host.py
  - id: openwiki-source-665a21e2fbd09a89d3f13ac0
    resource: repo://libs/talon/deepagents_talon/runtime.py
  - id: openwiki-source-811fef57cecdbee2ba06a7b5
    resource: repo://libs/talon/deepagents_talon/store_archive.py
  - id: openwiki-source-fcdff263e59dd54dfd953e9b
    resource: repo://libs/talon/deepagents_talon/store_records.py
  - id: openwiki-source-fdd0c2c3830b8e9a88502a57
    resource: repo://libs/talon/README.md
  - id: openwiki-source-c996df77875d3c6b30ca07cf
    resource: repo://libs/talon/tests/unit_tests/test_archive_saver.py
  - id: openwiki-source-c804dd581207efbddccb706e
    resource: repo://libs/talon/tests/unit_tests/test_conversation_deletion.py
  - id: openwiki-source-f2859f71853cf2cbdb40aaa3
    resource: repo://libs/talon/tests/unit_tests/test_scheduled_history.py
  - id: openwiki-source-06d3e41642ffa8b7153931b6
    resource: repo://libs/talon/tests/unit_tests/test_store_archive.py
generated: { by: "openwiki/0.4.2", at: "2026-10-01T08:06:30.386Z" }
---

# State, Checkpoints, and Persistent Records

Talon persistence is deliberately split by purpose. A LangGraph checkpoint makes an agent thread resumable; a conversation archive makes selected, user-visible chat history retrievable; vector data accelerates retrieval but is derived; and the assistant home holds operational configuration and scheduler records. These stores do **not** form one transaction or share one retention and erasure lifecycle.

> **Experimental and local-only boundary:** Talon is experimental, alpha-status software. It is not intended for production or enterprise use, and it is not a production or multi-tenant security boundary. In particular, channel access should be treated as access to the operator's agent, credentials, tools, and host resources; sandboxing is opt-in and does not cover every integration.

```mermaid
sequenceDiagram
    participant Runtime as Talon runtime
    participant Saver as ConversationSaver
    participant Checkpoint as LangGraph checkpoint store
    participant Archive as Chat-scoped archive
    participant Vectors as Optional vector index

    Runtime->>Saver: root checkpoint with trusted scope
    Saver->>Archive: register session ownership
    Saver->>Checkpoint: persist checkpoint
    Saver->>Archive: append changed message revisions
    Saver->>Archive: acknowledge checkpoint
    Archive->>Vectors: schedule derived indexing
```
*An eligible interactive checkpoint is persisted before its transcript revisions; archive acknowledgement records completion across two independent stores.*

## Ownership map

| Record | Owner and identity | Durability role |
| --- | --- | --- |
| Graph checkpoint and pending writes | LangGraph checkpointer, keyed by graph thread ID | Resume executable agent state. The default model host uses `checkpoints.sqlite`. |
| Conversation archive | `StoreConversationArchive`, scoped by trusted channel and chat, with a session registration | Read, list, search, and erase chat history independently of graph-state mechanics. |
| Semantic vectors | A separate vector store/index keyed to archive material and an embedding generation | Derived retrieval data; it can be rebuilt from retained transcript metadata. |
| Assistant operational records | Per-assistant home | Holds checkpoint path, model and conversation selection state, channel state, policies, media, and cron state. |
| Cron records | `<assistant home>/cron/jobs.json`, keyed by cron job ID | Durable job definitions, schedule progress, origin, delivery choice, and last-run outcome. |

Do not treat an archive as a backup of arbitrary graph state, or a vector index as the transcript authority. Likewise, deleting chat history does not imply deletion of cron jobs, assistant-home files, traces, or external backups.

## Assistant home and default checkpoint path

`TalonConfig` validates the assistant ID before deriving a home, normally `~/.deepagents/<assistant_id>`. IDs are safe path components (1–128 letters, digits, underscores, hyphens, or dots, excluding `.` and `..`). Home creation and materialized state directories use mode `0700`; state-path access verifies that the resolved file is directly inside the expected assistant home. Keep this home outside an agent workspace and manage filesystem access separately.

The home is operational state, not just a database directory. Relevant records include:

- `checkpoints.sqlite` for persistent LangGraph checkpoints;
- `conversations.json`, `models.json`, and `smart-model.json` for active conversation generations and model choices;
- `channels/`, `cron/`, tool policy, and inbound media directories; and
- fingerprint-specific `history-vectors-<generation>.sqlite` files when local vector indexing is used.

For a configured model host, the CLI opens an `AsyncSqliteSaver` at `checkpoint_path`, initializes it, opens history, and gives the graph a `ConversationSaver` that wraps the SQLite saver. The wrapper owns neither underlying connection: callers that embed Talon can provide the checkpointer and archive lifetimes, but must use `ConversationSaver` if they want the history tools and coordinated archive behavior.

## Archive writes: a repairable two-store boundary

`ConversationSaver` is the bridge between a LangGraph `BaseCheckpointSaver` and `StoreConversationArchive`. Synchronous write and administrative copy/prune APIs are deliberately unsupported so writes cannot silently bypass archival. A shared async lock serializes checkpoint/archive changes, reset, and selected deletion.

Only a root graph checkpoint carrying trusted `talon_history_channel` and `talon_history_chat` metadata is archive-eligible. Nested graph namespaces and unscoped writes remain normal checkpoints but do not enter the archive. Before checkpoint mutation, the wrapper registers the thread's archive ownership; a session is bound to its first scope and cannot subsequently move to another chat. The narrowly defined compatibility exception preserves legacy Slack public-thread history when it is accessed through its parent channel scope.

For an eligible write, the wrapper determines changed message revisions, persists the graph checkpoint, appends those revisions to the archive, then stores an archive acknowledgement for that checkpoint. There is no transaction spanning checkpoint and archive storage:

- If checkpoint persistence fails, uncommitted messages are not archived.
- If archive append fails after a successful checkpoint, the error propagates and the checkpoint remains a repair condition. Retrying the same checkpoint repairs the archive without duplicate revisions because archive chunks are revision-idempotent.
- Cancellation is shielded until the in-progress checkpoint/archive operation finishes, then cancellation is re-raised. Reset or deletion therefore cannot race an unfinished append.

Archive deletion follows the analogous retryable ordering: delete a backend thread first, then remove its archive registration. A failed step leaves the registration available for retry. Selected deletion rejects empty IDs, the current session, and running sessions; it only acts on sessions owned by the supplied chat scope. A batch can be partially complete, so retry the same requested IDs rather than assuming all-or-nothing erasure.

## Archive store, recovery, and delivery visibility

`StoreConversationArchive` stores transcript metadata through `StoreRecords`; metadata and vectors must be distinct `BaseStore` instances. It requires one active writer per namespace, read-after-write consistency, and no automatic TTL. Its bounded redo journal serializes access in-process, recovers a prior journal before records are exposed, writes the journal before an idempotent batch, and removes it only after successful replay. This makes partial batches restartable; it is neither a distributed lock nor a transaction with the checkpoint database.

Archive setup recovers records before retrieval or indexing begins, and close waits for active vector work. Session scope is authoritative: archive entries and deletion cannot cross channels or chats. Retrieval itself is bounded—pages are at most 20 items and scans stop at 500 records rather than silently claiming a complete result beyond that budget.

The archive distinguishes generated model output from a reply confirmed as delivered by the host. `record_delivery()` promotes the matching final reply (or records a delivery-specific entry) only after the host reports successful send, making it eligible for semantic indexing. If this post-delivery archive recording fails, the host logs it rather than changing the delivery result; delivery and history indexing are separate outcomes.

## History backends and derived vector indexes

`DEEPAGENTS_TALON_HISTORY_URI` selects archive metadata storage. If it is unset, history uses a separate SQLite connection to `checkpoints.sqlite`; SQLite or `file`, MongoDB, and PostgreSQL schemes are built in. Another scheme requires exactly one installed `deepagents_talon.history_backends` entry-point plugin. Archive records are namespaced by `("talon", assistant_id)`, allowing assistants to share a backend without sharing history.

Talon validates URI syntax at configuration time. Metadata startup, archive startup, and plugin failures are converted to generalized configuration errors so connection strings and credentials are not reflected in the user-facing error. Backend plugins are operator-installed, trusted code, not an isolation boundary.

Vector search is optional and independently configured. Metadata remains in the archive store while a vector backend—built in for the selected scheme or supplied by exactly one `deepagents_talon.history_vector_backends` plugin—holds embeddings. An embedding fingerprint guards reuse. If a retained index is incompatible, startup refuses it unless `DEEPAGENTS_TALON_HISTORY_REINDEX=1` is explicitly set; reindexing erases and rebuilds vectors while retaining transcripts, records durable progress, and resumes after interruption. Back up and erase transcript metadata independently from vectors, since vectors are reconstructible derived data.

## Scheduled work and cron records

`CronJobStore` persists an assistant-scoped, versioned `jobs.json`. A job records its assistant, prompt, schedule and repeat state, enablement, next/last/claim times, last status and error, delivery target, and origin. Origin contains the source conversation and channel plus optional sender and `history_chat`, allowing a scheduled Discord or Slack job to use the appropriate parent chat scope.

Publication is an atomic-file-replacement protocol: Talon serializes the complete JSON envelope to a temporary file in the cron directory, flushes and fsyncs it, sets mode `0600`, replaces `jobs.json`, and fsyncs the directory. Reads cache the parsed contents by file identity. This is durable publication for the explicitly single-writer read-all/write-all design, not cross-process coordination.

A scheduled invocation uses its own `:talon-cron` graph thread. When its origin has a trusted history scope, it may read that chat's archive; its graph writes are marked archive-read-only, so the unattended trace does not become transcript entries. It cannot delete history. The cron session registration is nevertheless retained so a chat reset can erase its owned checkpoint. If—and only if—the host successfully delivers a non-silent final result to the destination, it separately records that delivered reply into the origin chat's archive.

## Operating and testing this boundary

- Back up and set retention independently for checkpoints, assistant-home records, archive metadata, cron JSON, and vectors. A successful archive search does not prove a checkpoint can be resumed, and a checkpoint does not prove archive acknowledgement succeeded.
- Keep one active archive writer per assistant namespace and one cron writer. Do not add cross-process writers without adding coordination at the storage layer.
- Treat a checkpoint that lacks archive acknowledgement as repairable incomplete history. Preserve the checkpoint-first ordering and let an archive failure surface for retry.
- Expect reset and selected deletion to be retryable but potentially partial. Stop active chat work before a full reset; do not conflate chat-history erasure with cron-job deletion.
- Focus regression tests on boundary failures: checkpoint versus archive failure ordering and cancellation, scope reassignment and scoped deletion, redo-journal recovery, scan limits, legacy Slack scope compatibility, scheduled read-only history and successful-delivery recording, cron origin persistence, and atomic cron-file publication.

See [Runtime behavior](/openwiki/architecture/runtime-behavior.md), [Backends](/openwiki/concepts/backends.md), [Talon scheduling](/openwiki/concepts/talon-scheduling.md), [Cost and sessions](/openwiki/operations/cost-and-sessions.md), and [Security](/openwiki/operations/security.md).
