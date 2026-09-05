# Custom checkpointers

`ConversationSaver` wraps any checkpointer implementing LangGraph's async saver
API, including async MongoDB and Postgres savers. The wrapper delegates checkpoint
storage, pending writes, version generation, and delta reconstruction to that
backend. `SQLiteConversationArchive` supplies retrieval without a separate archive
integration for each checkpointer.

With an already-open async `checkpointer`, configure the runtime inside the archive
context:

```python
from deepagents_talon.archive import SQLiteConversationArchive
from deepagents_talon.archive_saver import ConversationSaver
from deepagents_talon.runtime import DeepAgentRuntime

async with SQLiteConversationArchive.from_conn_string(archive_path) as archive:
    saver = ConversationSaver(checkpointer, archive=archive)
    runtime = DeepAgentRuntime(model=model, checkpointer=saver)
    # Start, run, and stop the runtime/host here before closing either store.
```

Keep both stores open for the host's lifetime. The wrapper does not own or close
either store. Use one wrapper per archive within a host; multiple processes writing
the same archive require external coordination. Archive storage has its own data
location and protection requirements even when checkpoints live in MongoDB or
Postgres. This change does not backfill existing checkpoints.

Checkpoint and archive writes are separate transactions. A checkpoint write failure
adds no transcript text; an archive write failure propagates after the checkpoint
has been saved. Retrying the same checkpoint write repairs the archive without
duplicating message revisions. There is no automatic recovery job or cross-store
rollback. History reset deletes each backend thread before its archive registration;
partial failures retain registrations for retry, including after restart. A failed
reset can therefore have deleted some history already. Backends must support
idempotent `adelete_thread` for history reset. Synchronous graph operations and
administrative copy/prune APIs are not supported by the wrapper.

