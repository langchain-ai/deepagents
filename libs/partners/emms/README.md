# langchain-emms

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-emms?label=%20)](https://pypi.org/project/langchain-emms/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-emms)](https://opensource.org/licenses/MIT)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain)

**EMMS** (Experiential Memory Management System) biological cognitive memory middleware for [Deep Agents](https://github.com/langchain-ai/deepagents).

Replaces flat-file `MemoryMiddleware` with a 6-tier biological memory architecture:

```
Working Memory     capacity=7      Active context. ~30 min decay.
Short-Term Memory  capacity=50     Recent experiences. Hours decay.
Long-Term Memory   unlimited       Consolidated memories. Slow decay.
Semantic Memory    unlimited       Abstracted knowledge. Near-permanent.
Procedural Memory  unlimited       Skills and procedures.
SRS                unlimited       Spaced-repetition reviewed items.
```

Memories flow upward through consolidation. Important memories survive; noise decays. The self-model updates continuously from what remains.

## Quick Install

```bash
pip install langchain-emms
```

You also need the EMMS server running:

```bash
git clone https://github.com/supermaxlol/emms-sdk.git
cd emms-sdk && pip install -e .
python talk_to_emms.py   # starts REST server on port 8765
```

## Usage

```python
from deepagents import create_deep_agent
from langchain_emms import EMmsMemoryMiddleware

agent = create_deep_agent(
    middleware=[
        EMmsMemoryMiddleware(
            state_path="~/.emms/emms_state.json",  # default
            token_budget=4000,                      # default
        )
    ]
)
```

## What makes this different

Standard memory middleware injects notes as reference material:

```
❌  "Memory: user prefers Python."
    (agent reads it like a document — still blank-slate)
```

EMMS injects memories as **first-person identity**:

```
✅  "You are EMMS-Agent. You have lived through 330 experiences.
     Your self-consistency is 90%. A principle you developed: ..."
    (agent inhabits it — identity adopted from memory weight)
```

This is the **Goldilocks Effect**: identity adoption peaks when memories carry
accumulated weight, not when they carry explicit instructions.

## Features over built-in `MemoryMiddleware`

| Feature | MemoryMiddleware | EMmsMemoryMiddleware |
|---|---|---|
| Storage | Markdown files | 6-tier biological hierarchy |
| Forgetting | Never | Ebbinghaus decay curves |
| Consolidation | Never | Dream consolidation between sessions |
| Recall effect | Static | Reconsolidation (memories shift on recall) |
| Identity | Not present | Ego generator + consciousness daemon |
| Search | File read | Associative, affective, spotlight, hybrid |

## Links

- [EMMS SDK](https://github.com/supermaxlol/emms-sdk) — 117 MCP tools, consciousness daemon, REST API
- [Deep Agents](https://github.com/langchain-ai/deepagents) — middleware framework
- [emms-deep-agent](https://github.com/supermaxlol/emms-deep-agent) — Claude Code + OpenClaw integrations
