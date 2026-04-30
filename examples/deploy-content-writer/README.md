# deploy-content-writer

A content writing agent deployed with `deepagents deploy`. It writes blog posts, LinkedIn posts, and tweets — and remembers each user's preferences across sessions using per-user memory scoped by their identity.

This example also demonstrates **custom auth**: adding `[auth] provider = "supabase"` to `deepagents.toml` so that every user's memory is isolated to their account with zero custom code.

## Prerequisites

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | GPT-4.1 model access |
| `LANGSMITH_API_KEY` | Required for deploy |
| `SUPABASE_URL` | Your Supabase project URL (for auth) |
| `SUPABASE_ANON_KEY` | Your Supabase anon/public key (for auth) |

Copy `.env.example` to `.env` and fill in your keys. The Supabase keys are only required if you keep the `[auth]` section in `deepagents.toml`. Remove it to deploy without authentication.

## Deploy

```bash
deepagents deploy
```

On deploy, the `[auth]` section in `deepagents.toml` generates a Supabase token validator and wires it into the deployment automatically — no custom middleware needed.

## How per-user memory works

Each authenticated user gets their own memory files at `/memories/user/`:

- `preferences.md` — the agent reads and updates this to remember tone, topics, and formatting choices
- `context.md` — static context about the user's company and product

Because auth scopes these files by user identity, one deployment serves many users without any bleed between accounts.

## What to try

Once deployed, open the agent in LangSmith and send it prompts like:

- `"Write a blog post about the benefits of AI agents for developer teams"`
- `"Turn this into a LinkedIn post: [paste your content]"`
- `"I prefer a more casual tone — remember that for future posts"`
- `"Draft three tweet variations for our new product launch"`

## Query via SDK

Pass your Supabase JWT in the `Authorization` header — the deployment validates it and infers the user identity automatically:

```python
from langgraph_sdk import get_client

client = get_client(
    url="https://<your-deployment-url>",
    headers={"Authorization": "Bearer <your-supabase-jwt>"},
)
thread = await client.threads.create()

async for chunk in client.runs.stream(
    thread["thread_id"], "agent",
    input={"messages": [{"role": "user", "content": "Write a tweet about AI agents"}]},
    stream_mode="messages",
):
    print(chunk.data, end="", flush=True)
```

Find your deployment URL in LangSmith under **Deployments**. See `test_user_memory.py` for a full example and the [LangGraph SDK docs](https://langchain-ai.github.io/langgraph/concepts/sdk/) for more.

## Structure

```
deploy-content-writer/
├── AGENTS.md              # Agent instructions and memory workflow
├── deepagents.toml        # Deploy config (model, auth)
└── skills/
    ├── blog-post/         # Long-form blog post skill
    └── social-media/      # LinkedIn and tweet skill
```

## Resources

- [deepagents deploy docs](https://docs.langchain.com/deepagents/deploy)
- [Custom auth docs](https://docs.langchain.com/deepagents/auth)
- [Per-user memory docs](https://docs.langchain.com/deepagents/memory)
