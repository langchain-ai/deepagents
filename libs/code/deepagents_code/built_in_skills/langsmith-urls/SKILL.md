---
name: langsmith-urls
description: Produce correct, resolvable LangSmith run/trace/share URLs. Use whenever you emit one or more LangSmith links as a deliverable — a table, a list, or a single link — or when a developer reports that a LangSmith URL you gave them is dead or 404s.
license: MIT
compatibility: designed for deepagents-code
---

# LangSmith URLs

LangSmith share and trace URLs have mandatory path segments. Hand-assembling them from a share token alone produces links that 404. Follow these rules whenever a LangSmith link is part of your output.

## Best Practices

- **Never build a share URL from just a share token.** `https://smith.langchain.com/public/{share_token}/r` is incomplete — the `run_id` segment is mandatory. The canonical share format is `https://smith.langchain.com/public/{share_token}/r/{run_id}`.
- **Prefer the URL the source already returns.** Use the value the LangSmith SDK/CLI hands back when creating or reading a shared run (for example, the return of sharing a run) instead of any string you assemble yourself.
- **When you must build a non-shared trace URL,** use the full documented template with every required segment (host, organization/workspace, project, and run) — do not drop segments.

## Process

1. Obtain each URL from its canonical source — the SDK/CLI return value — rather than concatenating strings.
2. If you have no choice but to construct a URL, include all mandatory segments (`.../r/{run_id}` for share links).
3. Before presenting one or more URLs as a deliverable, validate that at least a sample resolves to a 2xx response (HEAD/GET). If validation is not possible in the environment, state explicitly that the links are unverified rather than implying they work.
4. When a developer reports a link is dead, re-derive it from the canonical source instead of re-emitting the same malformed string.

## Common Pitfalls

- Emitting a table or list of token-only `/public/{share_token}/r` links — every one 404s.
- Presenting unvalidated links as working, forcing the developer to test them and re-ask.
- Re-sending the same broken URL after being told it is dead, instead of re-deriving it.
