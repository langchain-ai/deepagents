# Copilot Instructions — deepagents

## Project

- **Name**: deepagents
- **Organization**: AiFeatures
- **Enterprise**: iAiFy
- **Language**: Python
- **Description**: Agent harness built with LangChain and LangGraph. Equipped with a planning tool, a filesystem backend, and the ability to spawn subagents - well-equipped to handle complex agentic tasks.

## Fork Status

This is a forked repository. Do not contribute back upstream.
Local customizations are preserved in the main branch.
Upstream sync is managed via Ai-road-4-You/fork-sync.

## Conventions

- Use kebab-case for file and directory names
- Use conventional commits (feat:, fix:, chore:, docs:, refactor:, test:)
- All PRs require review before merge
- Branch from main, merge back to main
- All file names in kebab-case

## Shared Infrastructure

- Reusable workflows: Ai-road-4-You/enterprise-ci-cd@v1
- Composite actions: Ai-road-4-You/github-actions@v1
- Governance standards: Ai-road-4-You/governance

## Quality Standards

- Run lint and tests before submitting PRs
- Keep dependencies updated via Dependabot
- No hardcoded secrets — use GitHub Secrets or environment variables
- Follow OWASP Top 10 security practices

## AgentHub Integration
- Skills: `.agents/skills/` in this repo links to shared AgentHub skills
- 14 shared agents available (api, architect, cli, deploy, developer, docker, docs, orchestrator, performance, refactor, reviewer, security, tester, troubleshoot)
- MCP: 12 servers (GitHub, Supabase, Playwright, MongoDB, Notion, HuggingFace, etc.)
