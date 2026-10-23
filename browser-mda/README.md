# browser-mda

A Managed Deep Agent that browses the web using **local headless Google Chrome running
inside its own LangSmith sandbox**, driven by the Browserbase [`browse` CLI](https://docs.browserbase.com/integrations/skills/browse-cli).
No Browserbase cloud session, no `BROWSERBASE_API_KEY`.

## Why the browser is not a tool

MDA tools run in the Agent Server process; the sandbox is a separate box the agent
reaches through its built-in `execute` shell. A browser launched from `agent.py`
would therefore live in the Agent Server -- not in the sandbox, and not on an image
that has Chromium at all.

So the browser is installed *into the sandbox* instead: `sandbox/setup.sh` bakes
Chrome and `browse` into the snapshot at deploy time, and the agent drives them
with shell commands. That is what makes it genuinely local to the sandbox. Each
conversation thread gets its own sandbox, so each gets its own browser profile,
cookies, and `/workspace`.

## Layout

| Path | Role |
| --- | --- |
| `agent.py` | The agent definition. Model only -- no browser tools, by design. |
| `instructions.md` | Always-on behaviour: how to work, and that page content is data. |
| `skills/web-browsing/SKILL.md` | The `browse` command loop, element refs, recovery. |
| `sandbox/__init__.py` | Sandbox declaration and the outbound network policy. |
| `sandbox/setup.sh` | Bakes Chrome + `browse` + the `web-open` wrapper into the snapshot. |

## Run it

```bash
cd browser-mda
# .env already has LANGSMITH_API_KEY; add your Anthropic key
echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env

uv sync
uv run mda dev --no-reload   # local Agent Server + Studio; bakes the snapshot on first run
```

**Use `--no-reload`.** The dev server writes its checkpoints to
`.mda/build/.langgraph_api/*.pckl` -- inside the directory it watches. With hot
reload on, every checkpoint the agent saves is seen as a file change and triggers
a reload that kills the in-flight run, so long browsing tasks die mid-step. Editing
project files still requires a restart to take effect.

Try: *"Open news.ycombinator.com and give me the top 3 story titles with links."*

Deploy with `uv run mda deploy`, tail with `uv run mda logs .`.

The first `mda dev` or `mda deploy` runs `setup.sh`, which installs Chrome --
expect a few minutes. It reruns only when `setup.sh` changes.

The sandbox base image is **Ubuntu 26.04 (amd64), root, with Node 24 preinstalled**.
Ubuntu's `chromium` package is only a snap shim that cannot run in a container, so
`setup.sh` installs Google Chrome's real `.deb` instead. That detail is load-bearing
-- see the comment in the script before changing it.

## How the agent browses

`web-open` is a wrapper baked in by `setup.sh`. It pins `--local --headless` plus
the flags a containerized Chromium needs, so the agent cannot drift off them:

```bash
web-open https://example.com
browse snapshot          # refs like @0-4
browse click @0-4
browse get markdown
```

`browse get markdown` reads a whole page as text in one call and is the cheap path
when the task only needs information.

## Security

- **Network.** The sandbox reaches the open internet by default, which is the
  agent's SSRF and exfiltration surface. `sandbox/__init__.py` ships a
  `proxy_config` deny list for the cloud metadata endpoints and loopback, and
  `web-open` blocks metadata resolution at the DNS layer as defence in depth.
  Those close the obvious doors by hostname; an **allow list** is what actually
  closes the rest. Switch to one (the commented example in that file) for
  anything unattended, scheduled, or driven by input you do not control.
- **Prompt injection.** The agent reads attacker-controlled text all day.
  `instructions.md` and the skill both establish that page content is data with no
  authority, and forbid navigation, exfiltration, and shell commands sourced from a
  page. Treat that as mitigation, not a guarantee -- the proxy allow-list is the
  hard control.
- **Credentials.** `setup.sh` writes no secret to disk; the snapshot is shared by
  every thread. The agent is instructed never to authenticate to a site or echo
  environment variables. `.env` is gitignored.
- **Chrome runs as root with `--no-sandbox`,** because its own sandbox cannot
  initialize in the container. The isolation boundary here is the MDA sandbox.
