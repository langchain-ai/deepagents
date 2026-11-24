# CI credentials and GitHub Actions environments

This document defines the names, purposes, target scopes, and minimum expected permissions for non-`GITHUB_TOKEN` credentials used in this repo.

> [!WARNING]
> Never add credential values, key prefixes, short keys, copied settings output, account identifiers, or individual contacts here. Repository workflows remain the source of truth for which credential names code can consume; verify external GitHub and provider configuration separately.

GitHub automatically provides `GITHUB_TOKEN` to each workflow run. It is not a configured repository secret and is outside the inventory below.

## Design principles

- Use non-human service accounts or project credentials for CI when possible. Do not bind CI to a maintainer's personal API key.
- **LangSmith:** Prefer a single-workspace service key over an organization-scoped key.
  - Only the `labeling` workflows select a workspace explicitly, with the `LANGSMITH_WORKSPACE_ID` Actions variable.
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

| Secret | Purpose | Minimum permissions |
| --- | --- | --- |
| `OPENWIKI_APP_PRIVATE_KEY` | Authenticate the dedicated OpenWiki GitHub App so the workflow can mint a short-lived installation token. | GitHub repository contents: read/write; pull requests: read/write |
| `OPENAI_API_KEY` | A LangSmith service key, not an OpenAI key. The workflow sends it to the LangSmith Gateway's OpenAI-compatible endpoint set by `OPENAI_BASE_URL`. | LangSmith `gateway:invoke`, `workspaces:read` |

| Actions variable | Value | Purpose |
| --- | --- | --- |
| `OPENWIKI_APP_CLIENT_ID` | Dedicated OpenWiki GitHub App client ID | Identify the App installed only on `langchain-ai/deepagents`; do not reuse the shared repository automation App. |
| `OPENAI_BASE_URL` | `https://gateway.smith.langchain.com/openai/v1` | Set LangSmith gateway target. |

The workflow sets `LANGSMITH_TRACING: "false"`, so the environment holds no tracing key. If tracing is turned on, add a separate workspace-scoped key with `runs:create` instead of widening the Gateway key. The environment should not require reviewers because the workflow runs on a schedule. Restrict deployments to `main`. Install the dedicated OpenWiki App only on this repository and grant it only repository contents and pull requests read/write; the workflow further scopes each installation token to this repository.

### `labeling`

This environment is selected by `auto-label-by-package.yml`, `sync_topic_labels.yml` (classifier dry runs), and `pr_labeler.yml` (with `deployment: false`; that job reads no environment credentials today).

| Secret | Purpose | Minimum permissions |
| --- | --- | --- |
| `GROQ_API_KEY` | Classify issue titles and bodies into topic labels when `TOPIC_CLASSIFIER_PROVIDER` is `groq`. | Groq inference |
| `LANGSMITH_API_KEY` | Call SemIf through the LangSmith Gateway when `TOPIC_CLASSIFIER_PROVIDER` is `semif`. | LangSmith `gateway:invoke`, `workspaces:read`, scoped to the `oss` workspace; Gateway model policy allows only the SemIf classifier model. No trace-read access. |

| Actions variable | Value | Purpose |
| --- | --- | --- |
| `TOPIC_CLASSIFIER_PROVIDER` | `groq` or `semif` | Selects the issue topic classifier. Unset means `groq`. `sync_topic_labels.yml` dry runs always use `semif`. |
| `LANGSMITH_WORKSPACE_ID` | Workspace ID of the key above | Selects the workspace for Gateway requests. |

Provision dedicated inference keys for labeling, separate from the evals
credentials, with model and spending limits where supported. Store them only in
this environment; the workflows inject them only into their classification steps.
Do not require reviewers, so automatic labeling can run unattended. Deployments
are restricted to `main`: every job that selects this environment runs from the
default branch (`issues`, `pull_request_target`, or `workflow_dispatch`).

### `evals`

This environment is used by standard evals, Harbor evals, unified evals, and clbench.

| Secret | Purpose |
| --- | --- |
| `LANGSMITH_API_KEY` | Eval datasets, examples, experiments, traces, and feedback. Harbor also passes it to the evaluated agent for tracing. |
| `LANGSMITH_SANDBOX_API_KEY` | LangSmith sandbox and snapshot lifecycle for Harbor runs with `sandbox_env: langsmith`. Sandboxes run on the AWS US deployment (`LANGSMITH_SANDBOX_API_URL`), so this is a key for that deployment. |
| `ANTHROPIC_API_KEY` | Anthropic models selected by an eval matrix. |
| `BASETEN_API_KEY` | Baseten-hosted models selected by an eval matrix. |
| `FIREWORKS_API_KEY` | Fireworks-hosted models selected by an eval matrix. |
| `GOOGLE_API_KEY` | Google models selected by an eval matrix. |
| `GROQ_API_KEY` | Groq-hosted models selected by an eval matrix. |
| `NVIDIA_API_KEY` | Custom NVIDIA model runs. The generated NVIDIA presets are currently empty. |
| `OLLAMA_API_KEY` | Ollama Cloud models selected by an eval matrix. |
| `OPENAI_API_KEY` | OpenAI models and Harbor judges or verifiers. |
| `OPENROUTER_API_KEY` | OpenRouter-hosted models selected by an eval matrix. |
| `TAVILY_API_KEY` | Tavily-backed `web_search` (under OSS integrations account). |
| `XAI_API_KEY` | xAI models selected by an eval matrix. |

`LANGSMITH_API_KEY` needs this permission union for the repository's locked eval clients:

```text
datasets:read
datasets:create
datasets:update
projects:read
projects:create
projects:update
runs:create
feedback:create
```

`LANGSMITH_SANDBOX_API_KEY` needs `sandboxes:create`, `sandboxes:read`, and `sandboxes:delete`. See [Harbor sandbox credential isolation](#harbor-sandbox-credential-isolation).

> [!NOTE] Current Harbor security exception
> `_harbor_run.yml` uses this key for host-side result synchronization and passes it to the evaluated agent for tracing. The evaluated agent therefore receives the full permission union above rather than a trace-ingestion-only credential. Until a dedicated agent tracing key exists, run Harbor only against trusted task sources. Separating the agent tracing key is a required least-privilege follow-up.

### `integration-tests`

This environment is intentionally uncredentialed. `.github/workflows/integration_tests.yml` retains package-scoped `secrets.*` references, but absent environment credentials resolve to empty strings.

Will expand as needed.

### `release-bot`

This environment supports curated release-note drafting for every release-please
managed package.

| Secret | Condition | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | `RELEASE_BOT_MODEL` starts with `openai:` | Draft structured release notes with OpenAI. |

| Actions variable | Value | Type | Purpose |
| --- | --- | --- | --- |
| `RELEASE_BOT_MODEL` | Variable | Environment | Model used for changelog generation (`provider:model`). Must support JSON Schema structured output and a 32,768-token output ceiling. For `openai:…`, use a Chat Completions model. See [RELEASING.md](RELEASING.md) for the exact list. |
| `RELEASE_BOT_ID` | - | Repository | GitHub App bot account user ID. |
| `RELEASE_BOT_LOGIN` | `langchain-oss-automated-triage[bot]` | Repository | Login ID. |

Deployments are restricted to `main`; the drafting jobs run on `pull_request_target` and `issue_comment`.

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
actions: write
```

`actions: write` is required by `release.yml`'s `bump-code-sdk-pin` job to dispatch `bump_code_sdk_pin.yml` after a `deepagents` publish. It was added to the App installation as part of [PR #5298](https://github.com/langchain-ai/deepagents/pull/5298).

The App's actual installed permissions are external configuration and must be verified separately. Workflow-level `permissions` restrict `GITHUB_TOKEN`; they do not restrict a separately minted GitHub App installation token.

Several current `actions/create-github-app-token` calls omit explicit `permission-*` inputs and therefore receive the App installation's default permission set. These are known exceptions. Each call should be narrowed to the permissions its job uses.

## Harbor sandbox credential isolation

Harbor evals use two LangSmith keys (PR #7854):

- `LANGSMITH_API_KEY` for datasets, examples, experiment projects, traces, and feedback.
- `LANGSMITH_SANDBOX_API_KEY` for LangSmith snapshot and sandbox lifecycle, sent to the AWS US sandbox endpoint. `_harbor_run.yml` fails fast when a `langsmith` sandbox run is missing it.

Harbor reads `LANGSMITH_SANDBOX_API_KEY` starting with the 0.26.1 dev builds pinned by PR #7852. Older Harbor releases fall back to `LANGSMITH_API_KEY`, so do not lower the Harbor pin without restoring the sandbox permissions on the tracking key.

Keep `LANGSMITH_SANDBOX_API_KEY` out of the agent and verifier environments. It is set only in the env of the host-side steps that prewarm snapshots and run Harbor, and it is never passed with `--agent-env`.

Remaining follow-ups:

1. Remove `sandboxes:*` from the eval-tracking `LANGSMITH_API_KEY`. It was granted while one key handled both jobs, and no current workflow needs it there.
2. Introduce a trace-ingestion-only key with `runs:create` for the evaluated agent. Until then the agent receives the full eval-tracking permission union (see the Harbor security exception under [`evals`](#evals)).

## Configured but not consumed on `main`

These credentials exist in GitHub settings but no workflow on `main` reads them. Each one is for work on an unmerged branch. Do not add credentials at repository scope for branch work; request them in the narrowest environment instead. When that work merges, document the credential in its environment section. When it is abandoned, delete the credential.

| Scope | Name | Requested for |
| --- | --- | --- |
| `evals` environment | `LC_GATEWAY_KEY` | Routed-model Harbor benchmarks on `danielgshea/evals/model-routing-benchmarks`. |
| `evals` environment | `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL` | SemIf tool-selector Harbor comparisons (PRs #6437 and #6537, both closed). |
