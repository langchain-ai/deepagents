# langchain-sprites

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-sprites?label=%20)](https://pypi.org/project/langchain-sprites/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-sprites)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-sprites)](https://pypistats.org/packages/langchain-sprites)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain_oss)

[Fly.io Sprites](https://sprites.dev) sandbox integration for Deep Agents. Sprites are persistent, named Linux VMs. A Sprite starts in 1 to 2 seconds, stops automatically when it is idle (a stopped Sprite has no cost), and supports fast checkpoint and restore of the full machine state.

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Quick Install

```bash
uv add langchain-sprites
```

```python
import os

from sprites import SpritesClient

from langchain_sprites import SpritesSandbox

client = SpritesClient(os.environ["SPRITES_TOKEN"])
sprite = client.create_sprite("my-agent-sandbox")
backend = SpritesSandbox(sprite=sprite, timeout=300)
result = backend.execute("echo hello")
print(result.output)
```

## Usage with Deep Agents

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5",
    system_prompt="You are a coding assistant with sandbox access.",
    backend=backend,
)
```

## Persistent sandboxes

Sprites are named and persistent. Do not delete the Sprite, and you can pick it up again later — with all installed packages, files, and state intact:

```python
# Day 1
sprite = client.create_sprite("my-agent-env")
backend = SpritesSandbox(sprite=sprite)
backend.execute("pip install flask")

# Day 2 — the Sprite resumes in about 1 second
sprite = client.sprite("my-agent-env")
backend = SpritesSandbox(sprite=sprite)
backend.execute("python app.py")
```

## Checkpoint and restore

Make a checkpoint of the full machine state before an agent run. Roll back if necessary:

```python
backend.sprite.create_checkpoint("before agent run")
checkpoints = backend.sprite.list_checkpoints()

# Roll back
backend.sprite.restore_checkpoint(checkpoints[0].id)
```

## Cleanup

```python
sprite.destroy()  # permanently deletes the Sprite
```
