# 🧠🤖 Deep Agents

[![PyPI - Version](https://img.shields.io/pypi/v/deepagents?label=%20)](https://pypi.org/project/deepagents/#history)
[![PyPI - License](https://img.shields.io/pypi/l/deepagents)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/deepagents)](https://pypistats.org/packages/deepagents)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain)

Looking for the JS/TS version? Check out [Deep Agents.js](https://github.com/langchain-ai/deepagentsjs).

To help you ship LangChain apps to production faster, check out [LangSmith](https://smith.langchain.com).
LangSmith is a unified developer platform for building, testing, and monitoring LLM applications.

## Quick Install

```bash
pip install deepagents
# or
uv add deepagents
```

## 🤔 What is this?

Using an LLM to call tools in a loop is the simplest form of an agent. This architecture, however, can yield agents that are "shallow" and fail to plan and act over longer, more complex tasks.

Applications like "Deep Research", "Manus", and "Claude Code" have gotten around this limitation by implementing a combination of four things: a **planning tool**, **sub agents**, access to a **file system**, and a **detailed prompt**.

`deepagents` is a Python package that implements these in a general purpose way so that you can easily create a deep agent for your application. For a full overview and quickstart of Deep Agents, the best resource is our [docs](https://docs.langchain.com/oss/python/deepagents/overview).

**Acknowledgements: This project was primarily inspired by Claude Code, and initially was largely an attempt to see what made Claude Code general purpose, and make it even more so.**

## 📖 Resources

- **[Documentation](https://docs.langchain.com/oss/python/deepagents)** — Full documentation
- **[API Reference](https://reference.langchain.com/python/deepagents/)** — Full SDK reference documentation
- **[Chat LangChain](https://chat.langchain.com)** - Chat interactively with the docs

## Optional features

### Video input (optional)

When the agent calls `read_file` on a video file (`.mp4`, `.mov`, `.webm`, etc.), `deepagents` extracts JPEG frames using **FFmpeg** and returns them as image content blocks. The agent picks the window by reusing the standard `read_file` paging arguments:

- `offset` — seconds into the source to start sampling (default `0`)
- `limit` — seconds of source to sample (default `100`)

The frames-per-second sampling rate is set at deploy/code time, not by the agent — pass `video_sampling_rate` to `FilesystemMiddleware` (default `0.5`, i.e. one frame every two seconds).

This requires `ffmpeg` and `ffprobe` to be on `PATH`:

```bash
# macOS
brew install ffmpeg

# Debian/Ubuntu
sudo apt-get install -y ffmpeg

# Verify
ffmpeg -version && ffprobe -version
```

Then install the deepagents `video` extra (currently a placeholder but declares intent for future Python-level video deps):

```bash
pip install 'deepagents[video]'
```

If FFmpeg is missing or extraction fails, `read_file` returns an error string and the agent turn still completes.

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
