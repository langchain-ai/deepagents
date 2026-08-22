# Agents

All agents are defined in `src/deep_agent/graph.py`.

## Conventions

- Prefer async-native code wherever possible for performance. Tools, tests, and any new I/O should use async.
- New tools should be low-dependency and safe to run on a remote server.
- This deploys in a web server. Avoid calls to actual file system.
