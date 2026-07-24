# CI credentials and GitHub Actions environments

This document defines the names, purposes, target scopes, and minimum expected permissions for non-`GITHUB_TOKEN` credentials used in this repo.

> [!WARNING]
> Never add credential values, key prefixes, short keys, copied settings output, account identifiers, or individual contacts here. Repository workflows remain the source of truth for which credential names code can consume; verify external GitHub and provider configuration separately.

GitHub automatically provides `GITHUB_TOKEN` to each workflow run. It is not a configured repository secret and is outside the inventory below.

## Design principles

- Use non-human service accounts or project credentials for CI when possible. Do not bind CI to a maintainer's personal API key.
- **LangSmith:** Prefer a single-workspace service key over an organization-scoped key.
  - At the time of writing, existing workflows do not select a workspace with `LANGSMITH_WORKSPACE_ID`
- Store credentials in the narrowest GitHub environment that needs them.
- Inject a provider credential only into the runtime step and package job that consumes it.
- Track any temporary exception to these rules explicitly and remove it when its blocker is resolved.

For same-named Actions secrets available to the repository, precedence is environment, then repository, then organization. Audit broader repository and organization scopes before deleting an environment secret; deleting it may expose a same-named broader credential. Organization-secret repository visibility also controls whether fallback is possible.

Repository and organization secrets are read when a workflow run is queued. Environment secrets are read when the referencing job starts. The `vars` and `secrets` namespaces are independent: a same-named variable does not override a secret.

## Target GitHub configuration

The sections below describe target configuration. Selecting `environment:` in workflow YAML does not prove that its secrets, protection rules, or branch policy have been configured.

### Repository scope

| Name | Kind | Purpose |
| --- | --- | --- |
| `ORG_MEMBERSHIP_APP_CLIENT_ID` | Actions variable | Identifies the GitHub App used by repository automation. |
| `ORG_MEMBERSHIP_APP_PRIVATE_KEY` | Secret | Authenticates the GitHub App so workflows can mint short-lived installation tokens. |

### `openwiki`

This environment is selected by `.github/workflows/openwiki-update.yml`.

| Secret | Purpose | Minimum LangSmith permissions |
| --- | --- | --- |
| `LANGSMITH_API_KEY` | Ingest OpenWiki traces into the `openwiki` project. | `runs:create` |
| `LS_GATEWAY_OPENAI_API_KEY` | Invoke the configured model through the workflow's current LangSmith Gateway endpoint. | `gateway:invoke`, `workspaces:read` |

| Actions environment variable | Value | Purpose |
| --- | --- | --- |
| `OPENAI_BASE_URL` | `https://gateway.smith.langchain.com/openai/v1` | Set LangSmith gateway target. |

Use separate workspace-scoped service keys for tracing and Gateway invocation. The environment should not require reviewers because the workflow runs on a schedule. Restrict deployments to `main`.

### `evals`

This environment is used by standard evals, Harbor evals, unified evals, and clbench.

| Secret | Purpose |
| --- | --- |
| `LANGSMITH_API_KEY` | Eval datasets, examples, experiments, traces, feedback, and—temporarily—LangSmith sandbox lifecycle. |
| `ANTHROPIC_API_KEY` | Anthropic models selected by an eval matrix. |
| `BASETEN_API_KEY` | Baseten-hosted models selected by an eval matrix. |
| `FIREWORKS_API_KEY` | Fireworks-hosted models selected by an eval matrix. |
| `GOOGLE_API_KEY` | Google models selected by an eval matrix. |
| `GROQ_API_KEY` | Groq-hosted models selected by an eval matrix. |
| `NVIDIA_API_KEY` | Custom NVIDIA model runs. The generated NVIDIA presets are currently empty. |
| `OLLAMA_API_KEY` | Ollama Cloud models selected by an eval matrix. |
| `OPENAI_API_KEY` | OpenAI models and Harbor judges or verifiers. |
| `OPENROUTER_API_KEY` | OpenRouter-hosted models selected by an eval matrix. |
| `XAI_API_KEY` | xAI models selected by an eval matrix. |

The current `LANGSMITH_API_KEY` needs this practical permission union for the repository's locked eval clients:

```text
datasets:read
datasets:create
datasets:update
projects:read
projects:create
projects:update
runs:create
feedback:create
sandboxes:create
sandboxes:read
sandboxes:delete
```

The sandbox permissions are temporary on this key. See [Harbor sandbox credential isolation](#harbor-sandbox-credential-isolation).

> [!NOTE] Current Harbor security exception
> `_harbor_run.yml` uses this key for host-side result synchronization and passes it to the evaluated agent for tracing. The evaluated agent therefore receives the full permission union above rather than a trace-ingestion-only credential. Until a dedicated agent tracing key exists, run Harbor only against trusted task sources. Separating the agent tracing key is a required least-privilege follow-up.

### `integration-tests`

This environment is intentionally uncredentialed. `.github/workflows/integration_tests.yml` retains package-scoped `secrets.*` references, but absent environment credentials resolve to empty strings.

Will expand as needed.

### `release-dcode`

This environment supports curated Deep Agents Code release-note drafting.

| Secret | Condition | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | `DCODE_RELEASE_MODEL` starts with `openai:` | Draft structured release notes with OpenAI. |

| Actions variable | Value | Type | Purpose |
| --- | --- | --- | --- |
| `DCODE_RELEASE_CLI_VERSION` | Variable | Environment | `deepagents-code` version to use. |
| `DCODE_RELEASE_MODEL` | Variable | Environment | Model used for changelog generation. |
| `DCODE_RELEASE_BOT_ID` | - | Repository | GitHub App bot account user ID. |
| `DCODE_RELEASE_BOT_LOGIN` | `langchain-oss-automated-triage[bot]` | Repository | Login ID. |

Of the model-provider credentials, only the credential for the configured provider is required. Prefer a provider project or service-account key limited to model inference, with model allowlists and spend limits where supported. The workflow also uses the repository GitHub App credentials for repository mutations.

### `release`

The environment is retained for release protections and possible future release-time integration coverage. It intentionally has no provider credentials while the `Run integration tests` step in `.github/workflows/release.yml` is disabled with `if: false`.

The disabled step preserves package-scoped wiring so a future re-enablement has an explicit credential contract:

| Release package | Credentials if re-enabled |
| --- | --- |
| `deepagents` | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `LANGSMITH_API_KEY` |
| `langchain-quickjs` | `ANTHROPIC_API_KEY` |
| `langchain-daytona` | `DAYTONA_API_KEY` |
| `langchain-modal` | `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` |
| `langchain-runloop` | `RUNLOOP_API_KEY` |
| `langchain-vercel-sandbox` | `VERCEL_TOKEN` plus `VERCEL_TEAM_ID` and `VERCEL_PROJECT_ID` Actions variables |

The disabled Vercel step currently references only `VERCEL_TOKEN`; its access-token authentication contract is incomplete. Add the two variables above, or deliberately adopt and document Vercel OIDC, before re-enabling that package's integration test.

## LangSmith permission caveats

The permission lists in this document describe operations required by the currently locked clients. Granular custom roles and service-key role assignment require LangSmith Enterprise RBAC; on other plans these minima are design targets rather than directly assignable roles.

## GitHub App credentials

Repository automation uses `actions/create-github-app-token` with:

```yaml
client-id: ${{ vars.ORG_MEMBERSHIP_APP_CLIENT_ID }}
private-key: ${{ secrets.ORG_MEMBERSHIP_APP_PRIVATE_KEY }}
```

Current workflow operations require the App installation to provide at least:

```text
organization members: read
repository contents: write
issues: write
pull requests: write
```

The App's actual installed permissions are external configuration and must be verified separately. Workflow-level `permissions` restrict `GITHUB_TOKEN`; they do not restrict a separately minted GitHub App installation token.

Several current `actions/create-github-app-token` calls omit explicit `permission-*` inputs and therefore receive the App installation's default permission set. These are known exceptions. Each call should be narrowed to the permissions its job uses.

## Harbor sandbox credential isolation

The intended eval architecture separates:

- `LANGSMITH_API_KEY` for datasets, examples, experiment projects, traces, and feedback.
- `LANGSMITH_SANDBOX_API_KEY` for LangSmith snapshot and sandbox lifecycle.

Deep Agents PR #4723 restored this separation by passing an explicit Harbor environment kwarg. The final tree merged by PR #4745 later removed that wiring, so current workflows again use `LANGSMITH_API_KEY` for both result synchronization and LangSmith sandbox access.

Upstream [harbor-framework/harbor#2344](https://github.com/harbor-framework/harbor/pull/2344) adds safe `${VAR}` resolution for environment and plugin kwargs while keeping the value out of process arguments and serialized job or trial configuration.

Until compatible support ships:

- Harbor uses `LANGSMITH_API_KEY` for both eval tracking and LangSmith sandbox lifecycle.
- Do not configure `LANGSMITH_SANDBOX_API_KEY`; no current workflow consumes it.

Reintroduce the split only after all of these conditions hold:

1. Harbor PR #2344, or equivalent support, is merged.
2. A Harbor release contains the change.
3. This repository's constraint and lock file select that release.

Then:

1. Create a workspace-scoped LangSmith service key with `sandboxes:create`, `sandboxes:read`, and `sandboxes:delete`.
2. Add it to the `evals` environment as `LANGSMITH_SANDBOX_API_KEY`.
3. Pass it only to the Harbor environment provider through:

   ```text
   --environment-kwarg 'langsmith_api_key=${LANGSMITH_SANDBOX_API_KEY}'
   ```

4. Keep it out of dependency installation, agent and verifier environments, the LangSmith results plugin, command-line values, and serialized Harbor configuration.
5. Confirm that general LangSmith client resolution ignores `LANGSMITH_SANDBOX_API_KEY`.
6. Run a one-task LangSmith-sandbox smoke test before removing sandbox permissions from the eval-tracking service key.

Separately, introduce a trace-ingestion-only key with `runs:create` for the evaluated agent. The host-side eval tracking key should not be passed into the agent environment after that migration.
