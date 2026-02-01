# GitHub PR Review Bot

An AI-powered pull request review bot built with [deepagents](https://github.com/langchain-ai/deepagents). When someone mentions `@deepagents-bot` in a PR comment, the bot performs a comprehensive review using specialized subagents for code quality and security.

## Features

- **Mention-triggered**: Just `@deepagents-bot` in a PR comment to get started
- **First-time help**: Shows available commands on first interaction
- **Live status updates**: Edits its comment with progress (like Dependabot) 🔄 📖 🔍 ✅
- **CI-aware**: Reads GitHub Actions workflows, build configs, and checks CI status
- **Context-aware**: Reads style guides, CONTRIBUTING.md, SECURITY.md first
- **Code quality review**: Style compliance, best practices, documentation, code smells
- **Security review**: Vulnerabilities, unsafe patterns, security anti-patterns
- **Parallel subagents**: Code and security reviews run concurrently
- **Can make commits**: Apply reviewer feedback or resolve conflicts (permission-aware)
- **Merge conflict resolution**: Intelligently merges conflicts, not just picking one side

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Webhook                           │
│                   (PR comment event)                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Orchestrator Agent                         │
│                                                             │
│  • Reads .github/workflows/, build configs, .md docs        │
│  • Checks CI status (passing/failing checks)                │
│  • Gathers PR context (diff, commits, comments)             │
│  • Delegates to specialized subagents                       │
│  • Synthesizes and posts final review                       │
└────────────┬────────────────────────────┬───────────────────┘
             │                            │
             ▼                            ▼
┌────────────────────────┐  ┌────────────────────────────────┐
│  Code Review Subagent  │  │    Security Review Subagent    │
│                        │  │                                │
│  • Style compliance    │  │  • Injection vulnerabilities   │
│  • Best practices      │  │  • Auth/authz issues           │
│  • Documentation       │  │  • Data exposure               │
│  • Code smells         │  │  • Crypto weaknesses           │
│  • Test coverage       │  │  • Dependency vulnerabilities  │
└────────────────────────┘  └────────────────────────────────┘
```

## Setup

### 1. Create a GitHub App

1. Go to GitHub Settings → Developer settings → GitHub Apps → New GitHub App
2. Configure the app:
   - **Name**: `deepagents-bot` (or your preferred name)
   - **Homepage URL**: Your server URL
   - **Webhook URL**: `https://your-server.com/webhook`
   - **Webhook secret**: Generate a secure random string
3. Set permissions:
   - **Repository permissions**:
     - Contents: Read & Write (needed for commits)
     - Issues: Read & Write
     - Pull requests: Read & Write
     - Security events: Read (optional, for security alerts)
   - **Subscribe to events**:
     - Issue comment
     - Pull request review comment
4. Generate a private key and download it

### 2. Install Dependencies

```bash
cd examples/github-pr-review
uv sync
```

### 3. Configure Environment

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Required variables:
- `GITHUB_APP_ID`: Your GitHub App ID
- `GITHUB_PRIVATE_KEY_PATH`: Path to your private key PEM file
- `GITHUB_WEBHOOK_SECRET`: The webhook secret you configured
- `ANTHROPIC_API_KEY`: Your Anthropic API key (or `OPENAI_API_KEY`)

### 4. Run the Server

```bash
uv run python server.py
```

For local development with ngrok:

```bash
# In one terminal
uv run python server.py

# In another terminal
ngrok http 8000
```

Update your GitHub App's webhook URL to the ngrok URL.

## Usage

Once installed on a repository, mention the bot in any PR comment:

```
@deepagents-bot
```

**First time?** The bot will show a help message with available commands, then start the review.

**Status updates:** The bot posts a status comment and edits it as work progresses:

```
## 🔄 Reviewing PR

✅ Gathered PR context
📖 Reading repository docs and configuration...
```

### Commands

| Command | Description |
|---------|-------------|
| `@bot` | Full code review |
| `@bot <question>` | Ask anything about the PR (freeform) |
| `@bot /review` | Full code review (explicit) |
| `@bot /security` | Security-focused review |
| `@bot /style` | Code style review |
| `@bot /feedback` | Apply reviewer feedback as commits |
| `@bot /conflict` | Resolve merge conflicts |
| `@bot /help` | Show available commands |

### Examples

```
@deepagents-bot                           # Full code review
@deepagents-bot what does this PR do?     # Ask a question
@deepagents-bot explain the auth changes  # Get an explanation
@deepagents-bot summarize this            # Get a summary
@deepagents-bot /review                   # Explicit full review
@deepagents-bot /security                 # Security-focused review  
@deepagents-bot /style                    # Code style review
@deepagents-bot /feedback                 # Apply reviewer feedback
@deepagents-bot /conflict                 # Resolve merge conflicts
```

### Adding Instructions

All slash commands accept optional instructions:

```
@deepagents-bot /review Focus on error handling

@deepagents-bot /security Check for SQL injection

@deepagents-bot /style Only check Python files

@deepagents-bot /feedback Apply feedback from @reviewer about logging

@deepagents-bot /conflict Prefer changes from main for config files
```

### How Commands Work

**`/security`** - Reads security documentation first:
- SECURITY.md, .github/SECURITY.md
- Security policies in CONTRIBUTING.md  
- Dependabot and CodeQL alerts
- Then analyzes the PR diff for vulnerabilities

**`/style`** - Reads style configuration first:
- pyproject.toml ([tool.ruff], [tool.black], etc.)
- .editorconfig, .prettierrc, .eslintrc
- CONTRIBUTING.md, STYLE.md, CODE_STYLE.md
- Then analyzes the PR diff for style issues

**`/feedback`** and **`/conflict`** - These commands make commits and require approval:

1. **Plan phase**: Bot analyzes and posts a plan of changes
2. **Approval**: React 👍 to approve, or 👎 to request changes
3. **Feedback loop**: If you 👎, the bot will @mention you asking for feedback. Reply with how to modify the plan, and it will revise and ask for approval again
4. **Execution**: Once approved, the bot executes the plan

Example flow:
```
## 📋 Applying feedback - Plan

**Planned changes:**

1. File: src/utils.py - Add docstring to calculate_total function
2. File: src/utils.py - Fix typo 'recieve' -> 'receive' on line 42

---
**React with 👍 to approve, or 👎 to request changes.**
```

After 👎:
```
## 💬 Applying feedback - Changes Requested

**Planned changes:**
...

---
@username Please reply with how you'd like me to modify the plan.

_I'll revise the plan based on your feedback and ask for approval again._
```

### Permission Model

**Enforced in code, not by the LLM:**
- **Users with write access**: Bot commits directly to the PR branch
- **Users with read-only access**: Bot creates a new branch and opens a separate PR

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GITHUB_APP_ID` | GitHub App ID | Yes |
| `GITHUB_PRIVATE_KEY_PATH` | Path to PEM file | Yes* |
| `GITHUB_PRIVATE_KEY_BASE64` | Base64-encoded private key | Yes* |
| `GITHUB_WEBHOOK_SECRET` | Webhook secret | Recommended |
| `ANTHROPIC_API_KEY` | Anthropic API key | Yes** |
| `OPENAI_API_KEY` | OpenAI API key | Yes** |
| `MODEL` | Model to use (default: `anthropic:claude-sonnet-4-5`) | No |
| `BOT_USERNAME` | Bot username to listen for (default: `deepagents-bot`) | No |
| `HOST` | Server host (default: `0.0.0.0`) | No |
| `PORT` | Server port (default: `8000`) | No |

\* One of `GITHUB_PRIVATE_KEY_PATH` or `GITHUB_PRIVATE_KEY_BASE64` is required  
\*\* At least one LLM API key is required

### Customizing the Bot

#### Change the bot mention trigger

Set `BOT_USERNAME` environment variable to match your GitHub App's name.

#### Modify review behavior

Edit `pr_review_agent/prompts.py` to customize:
- `ORCHESTRATOR_PROMPT`: Main agent behavior and workflow
- `CODE_REVIEW_PROMPT`: Code quality review focus areas
- `SECURITY_REVIEW_PROMPT`: Security review focus areas

#### Add new tools

Add tools to `pr_review_agent/tools.py` and register them in `pr_review_agent/agent.py`.

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install uv && uv sync --frozen

EXPOSE 8000
CMD ["uv", "run", "python", "server.py"]
```

### Cloud Run / Railway / Render

1. Set environment variables in your platform's dashboard
2. Deploy with the start command: `uv run python server.py`

### Kubernetes

See the `k8s/` directory for example manifests (coming soon).

## Development

### Run tests

```bash
uv run pytest
```

### Type checking

```bash
uv run mypy pr_review_agent
```

### Linting

```bash
uv run ruff check pr_review_agent
```

## How It Works

1. **Webhook Reception**: GitHub sends a webhook when someone comments on a PR
2. **Mention Detection**: The server checks if the comment mentions `@deepagents-bot`
3. **Permission Check**: The server (not the LLM) checks the requester's repository permissions
4. **Context Gathering**: The orchestrator agent fetches PR details, diff, commits, and repo style guides
5. **Parallel Review**: Code review and security review subagents analyze the PR simultaneously
6. **Synthesis**: The orchestrator combines feedback and posts a structured review comment
7. **Code Changes** (if requested): The bot commits changes following permission rules enforced by the server

## Security Model

Permission decisions are made in **code**, not by the LLM:

- The webhook handler checks user permissions via GitHub API before invoking the agent
- Based on permissions, the agent receives specific instructions about which branch/repo it can commit to
- The LLM cannot override these rules - it only sees the allowed options

This ensures the bot cannot be tricked via prompt injection into committing to unauthorized branches.

## License

MIT
