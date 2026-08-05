"""Generate Harbor tasks from DRBench enterprise deep-research records.

DRBench ships each task as a company/persona profile, a deep-research question, a
manifest of enterprise documents spread across four apps, and a set of ground-truth
insights. This adapter targets DRBench's *app* mode: each task runs upstream's
per-task container image, which serves the documents from Nextcloud, Mattermost,
Roundcube/IMAP, and a file browser, and the agent researches by navigating those apps
over the network rather than by reading files off disk.

Two compose services per task. `main` is where Harbor installs and runs the agent; it holds
no task data. `drbench` is upstream's image, which boots its own supervisord with this
task's documents already loaded.

The verifier runs in a *third*, separate environment built from the task's `tests/`
directory and started only after the agent's is torn down. That is what lets it install
upstream `drbench` and score with upstream's own metrics: the package ships both the gold
`eval.json` and the whole document corpus, neither of which may exist while the agent is
running. See `README.md` in the generated dataset for the runtime contract.

No task directory is committed. `populate_corpus` builds all 100 from upstream's configs
at `UPSTREAM_SHA`, fetched with a sparse checkout that skips the document corpus. That
keeps ~1,100 generated and copied files -- including the gold `solve.sh` answer keys --
out of this repository, and makes the dataset a function of one commit hash.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import urllib.request
from pathlib import Path

UPSTREAM_REPO = "ServiceNow/drbench"
# Task configs are fetched from this commit at generation time rather than committed.
# A git commit hash is a hash of the content, so this pins the configs as firmly as a
# vendored copy would. The images are pinned separately, by digest, in
# vendor/image_digests.json, because a digest is the only thing a git tree cannot carry.
UPSTREAM_SHA = "0d699ecf6aa96b1de378595b432e9b16a82f0ed9"
UPSTREAM_REMOTE = f"https://github.com/{UPSTREAM_REPO}.git"

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# Fetched with a blobless, depth-1, sparse checkout: the five config files per task are
# ~2.4 MiB, while `drbench/data/tasks/*/files/` is a ~69 MiB document corpus that app
# mode never needs, because upstream's per-task image serves the documents itself.
_SPARSE_PATTERNS = (
    "/drbench/data/tasks/*/config",
    "/drbench/data/tasks/*/*.json",
)

_GIT_TIMEOUT_SEC = 600

IMAGE_REGISTRY = "ghcr.io/mmunozm/drbench-services"
# Upstream publishes the per-task images for arm64 only ("amd64 images are coming
# soon"), so these tasks need an arm64 runner with `sandbox_env: docker`. The generated
# compose file deliberately sets no `platform:` -- see docker-compose.yaml.tmpl.

_MANIFEST_ACCEPT = ",".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)

_TASK_ID_RE = re.compile(r"^(?:DR\d{4}|SANITY\d+)$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

# Apps a DRBench document can be served from, and where the agent reaches each inside
# the compose network. The host is the compose service name, so DNS resolves it with no
# port publishing.
_APP_ENDPOINTS: dict[str, str] = {
    "nextcloud": "http://drbench:8081",
    "mattermost": "http://drbench:8082",
    "email": "imap://drbench:1143",
    "file_system": "http://drbench:8090",
}

# Upstream's built-in per-app logins, before any persona override. These are the
# credentials that actually work for the 85 tasks whose persona carries no password —
# verified by unpacking DR0016's image, whose documents sit under Nextcloud's `admin`
# user and whose mailbox is `current.user`, not the persona.
_APP_DEFAULT_CREDENTIALS: dict[str, dict[str, str]] = {
    "nextcloud": {"username": "admin", "password": "admin_pwd"},
    "mattermost": {"username": "admin@drbench.com", "password": "mm_admin_pwd"},
    "email": {"username": "current.user", "password": "current_user_pwd"},
    "file_system": {"username": "admin", "password": "admin_pwd"},
}

# Per-app access notes for the prompt. `USER` is substituted with the task's login name;
# the password is never written into the prompt -- see `_credential_env` for why.
_APP_GUIDANCE: dict[str, str] = {
    "nextcloud": (
        "the company cloud drive. HTTP Basic auth over WebDAV. List one level with "
        "`curl -u \"$DRBENCH_NEXTCLOUD_USER:$DRBENCH_NEXTCLOUD_PASS\" -X PROPFIND "
        "-H 'Depth: 1' -H 'Host: localhost' "
        "http://drbench:8081/remote.php/dav/files/USER/`, then repeat on any directory "
        "it returns (they end in `/`) to walk deeper, and GET any file path to download "
        "it. Both extra headers matter: this server answers `400 Bad Request` to "
        "`Depth: infinity`, and it only trusts the Host `localhost`, so a request "
        "addressed to `drbench:8081` is rejected without the `Host` override."
    ),
    "mattermost": (
        "team chat. `POST http://drbench:8082/api/v4/users/login` with a JSON body of "
        "`login_id` and `password` returns a session token in the `Token` response "
        "header; send it back as `Authorization: Bearer <token>`."
    ),
    "email": (
        "the mailbox for this login, over IMAP on `drbench:1143` (Python's `imaplib` "
        "works well). The same mail is browsable at `http://drbench:8085`."
    ),
    "file_system": (
        "local and shared drives, exposed through a file browser at `http://drbench:8090`."
    ),
}

# Returns 200 only once every service is up, 503 otherwise.
HEALTH_URL = "http://drbench:8099/health"

# Shared by both Docker build contexts a task generates (`environment/` and `tests/`).
_DOCKERIGNORE = (
    ".env\n.env.*\n*.pem\n*.key\n*.crt\ncredentials.json\n"
    ".git\n__pycache__/\n.venv/\n.DS_Store\n"
)


def _credential_env(credentials: dict[str, dict[str, str]], apps: list[str]) -> dict[str, str]:
    """Map each app's login to the env vars the prompt tells the agent to use.

    The logins are public benchmark fixtures -- upstream publishes them in its repo and
    bakes them into its public images -- so they are not secrets. They are passed through
    the environment rather than written into `instruction.md` for two reasons: a literal
    `-u <user>:<password>` repeated across 100 committed prompts trips secret scanners on
    every push (a false positive that trains people to ignore real alerts), and env
    indirection keeps every value in one clearly-labelled table per task instead of
    scattered through prose.
    """
    env: dict[str, str] = {}
    for app in apps:
        prefix = f"DRBENCH_{app.upper()}"
        env[f"{prefix}_USER"] = credentials[app]["username"]
        env[f"{prefix}_PASS"] = credentials[app]["password"]
    return env

# Upstream `info.json` fields that decide how a task is stratified. Vendored per task in
# task_labels.json so a proportional-sample check can run without building the dataset.
_LABEL_FIELDS = ("difficulty", "industry", "domain")

_INSIGHT_QA_TYPE = "insight"
_DISTRACTOR_QA_TYPE = "distractor"

# Record key -> path of that config inside an upstream task directory. Upstream splits
# them: the three scoring configs live under `config/`, the label file at the task root.
_CONFIG_SOURCES: dict[str, tuple[str, ...]] = {
    "task": ("config", "task.json"),
    "env": ("config", "env.json"),
    "eval": ("config", "eval.json"),
    "info": ("info.json",),
}


def vendor_dir() -> Path:
    """Return the directory containing the vendored DRBench pins.

    Defined as a function (rather than a module-level constant) so tests can
    monkeypatch it to point at a fixture directory.

    Returns:
        Path to the `vendor/` directory shipped alongside this module.
    """
    return Path(__file__).resolve().parent / "vendor"


def _templates_dir() -> Path:
    """Return the directory holding the task-invariant files."""
    return Path(__file__).resolve().parent / "templates"


def upstream_dir() -> Path:
    """Return the directory holding the pinned upstream checkout.

    Override with `DRBENCH_UPSTREAM_DIR` to share one checkout across dataset builds.

    Returns:
        Path to the local checkout, which may not exist yet.
    """
    override = os.environ.get("DRBENCH_UPSTREAM_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent / ".upstream"


def _git(*args: str, cwd: Path | None = None) -> str:
    """Run one git command with a list argv and return its stdout.

    Raises:
        RuntimeError: If git exits non-zero, with its stderr attached.
    """
    command = ["git", *args]
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell, no interpolated input
        command,
        cwd=None if cwd is None else str(cwd),
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_SEC,
        check=False,
    )
    if result.returncode != 0:
        msg = f"`{shlex.join(command)}` failed ({result.returncode}): {result.stderr.strip()}"
        raise RuntimeError(msg)
    return result.stdout


def ensure_upstream_checkout() -> Path:
    """Fetch upstream's task configs at `UPSTREAM_SHA`, and return the checkout root.

    Idempotent: a checkout already at the pinned commit is reused, so only the first
    call in a fresh environment touches the network.

    Returns:
        Path to the checkout root.

    Raises:
        ValueError: If `UPSTREAM_SHA` is not a full 40-character commit hash.
        RuntimeError: If a git command fails.
    """
    if _SHA_RE.match(UPSTREAM_SHA) is None:
        msg = f"UPSTREAM_SHA must be a full 40-character commit hash, got {UPSTREAM_SHA!r}"
        raise ValueError(msg)

    checkout = upstream_dir()
    if (checkout / ".git").is_dir():
        try:
            if _git("rev-parse", "HEAD", cwd=checkout).strip() == UPSTREAM_SHA:
                return checkout
        except RuntimeError:
            # An interrupted fetch can leave a repo with no HEAD; fall through and redo it.
            shutil.rmtree(checkout)

    checkout.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", str(checkout))
    # `config` rather than `remote add`, which fails when a prior attempt already added
    # it -- reached whenever UPSTREAM_SHA is bumped over an existing checkout.
    _git("config", "remote.origin.url", UPSTREAM_REMOTE, cwd=checkout)
    _git("sparse-checkout", "set", "--no-cone", *_SPARSE_PATTERNS, cwd=checkout)
    _git(
        "fetch",
        "-q",
        "--depth",
        "1",
        "--filter=blob:none",
        "origin",
        UPSTREAM_SHA,
        cwd=checkout,
    )
    _git("checkout", "-q", "FETCH_HEAD", cwd=checkout)
    return checkout


def _upstream_tasks_root() -> Path:
    """Return the directory holding upstream's per-task config directories."""
    return ensure_upstream_checkout() / "drbench" / "data" / "tasks"


def parse_task_id(task_id: str) -> str:
    """Validate a DRBench task id.

    Args:
        task_id: Identifier of the form `DR0001` or `SANITY0`.

    Returns:
        The validated `task_id`.

    Raises:
        ValueError: If `task_id` is not a bare DRBench id. Rejecting anything that is
            not a single path component keeps the id safe to join onto an output
            directory and to interpolate into an image tag.
    """
    if _TASK_ID_RE.match(task_id) is None or Path(task_id).name != task_id:
        msg = f"`task_id` {task_id!r} must be a DRBench id such as `DR0001` or `SANITY0`"
        raise ValueError(msg)
    return task_id


def available_task_ids() -> list[str]:
    """Return the benchmark's task ids, sorted.

    Upstream also ships `SANITY0`, a single-document smoke task used to check a local
    install. It is excluded here so `--all` yields exactly the 100 scored DRBench tasks
    and cannot skew a dataset average; generate it explicitly with `--task-ids SANITY0`
    when debugging.

    Read from the vendored copy of upstream's own `val.jsonl` subset rather than by
    listing directories, so the ids resolve with no network and before any task directory
    exists. That is what lets the workflow's prep phase shard the category while the
    dataset is still ungenerated.

    Returns:
        Sorted `DR<nnnn>` task ids.

    Raises:
        FileNotFoundError: If the vendored subset file is missing.
        ValueError: If the subset file is malformed or holds a malformed task id.
    """
    return sorted(read_subset_task_ids(vendor_dir() / "subsets" / "val.jsonl"))


def read_subset_task_ids(path: Path) -> list[str]:
    """Return the task ids listed in one upstream subset file.

    Args:
        path: Path to an upstream `*.jsonl` subset file.

    Returns:
        Task ids in file order.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If a line is not a JSON object, carries no string `task_id`, or the
            id is not a bare DRBench id.
    """
    if not path.is_file():
        msg = f"No DRBench subset file at {path}"
        raise FileNotFoundError(msg)
    task_ids: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        entry = json.loads(line)
        if not isinstance(entry, dict):
            msg = f"{path}:{lineno} must hold a JSON object"
            raise ValueError(msg)
        task_id = entry.get("task_id")
        if not isinstance(task_id, str):
            msg = f"{path}:{lineno} has no string `task_id`"
            raise ValueError(msg)
        task_ids.append(parse_task_id(task_id))
    if not task_ids:
        msg = f"{path} lists no tasks"
        raise ValueError(msg)
    return task_ids


def record_for_task_id(task_id: str) -> dict[str, dict]:
    """Load one task's DRBench config bundle from the pinned upstream checkout.

    Fetches the checkout on first use. `task_id` is validated before it is joined to a
    path, so it can only ever name a direct child of upstream's tasks directory.

    Args:
        task_id: Identifier of the form `DR0001` or `SANITY0`.

    Returns:
        A mapping with `task`, `env`, `eval`, and `info` keys holding the parsed
        upstream config JSON.

    Raises:
        ValueError: If `task_id` is not a bare DRBench id.
        FileNotFoundError: If upstream has no config for `task_id`.
        TypeError: If a config file does not hold a JSON object.
        RuntimeError: If the upstream checkout cannot be fetched.
    """
    parse_task_id(task_id)
    task_root = _upstream_tasks_root() / task_id
    record: dict[str, dict] = {}
    for key, relative in _CONFIG_SOURCES.items():
        path = task_root.joinpath(*relative)
        if not path.is_file():
            msg = f"No upstream DRBench config for {task_id!r} (expected {path})"
            raise FileNotFoundError(msg)
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            msg = f"DRBench config {path} must hold a JSON object"
            raise TypeError(msg)
        record[key] = parsed
    return record


def _labels_path() -> Path:
    """Return the path of the vendored per-task label record."""
    return vendor_dir() / "task_labels.json"


def load_task_labels() -> dict[str, dict[str, str]]:
    """Return the vendored `task_id -> {difficulty, industry, domain}` mapping.

    Committed rather than read from a task config because these labels are what make the
    `research` full profile's 30-task sample checkable as a proportional slice of the
    corpus, and that check has to run in test suites that never build the dataset.

    Returns:
        A mapping of task id to its three upstream labels.

    Raises:
        FileNotFoundError: If the label record is missing.
        ValueError: If the record is malformed, names a non-DRBench id, or holds a
            non-string label.
    """
    path = _labels_path()
    if not path.is_file():
        msg = (
            f"No vendored DRBench task labels at {path}. Generate them with "
            "`python -m harbor_adapters.drbench.main --refresh-labels`."
        )
        raise FileNotFoundError(msg)
    parsed = json.loads(path.read_text(encoding="utf-8"))
    labels = parsed.get("labels") if isinstance(parsed, dict) else None
    if not isinstance(labels, dict):
        msg = f"{path} must hold a `labels` object"
        raise ValueError(msg)
    validated: dict[str, dict[str, str]] = {}
    for task_id, entry in labels.items():
        parse_task_id(task_id)
        if not isinstance(entry, dict):
            msg = f"{path} entry for {task_id!r} must be an object"
            raise ValueError(msg)
        missing = [field for field in _LABEL_FIELDS if not isinstance(entry.get(field), str)]
        if missing:
            msg = f"{path} entry for {task_id!r} is missing string {missing}"
            raise ValueError(msg)
        validated[task_id] = {field: entry[field] for field in _LABEL_FIELDS}
    return validated


def _labels_from_upstream() -> dict[str, dict[str, str]]:
    """Return each task's labels as read from the pinned upstream checkout."""
    return {
        task_id: {
            field: str(record_for_task_id(task_id)["info"].get(field, "")) for field in _LABEL_FIELDS
        }
        for task_id in available_task_ids()
    }


def refresh_task_labels() -> int:
    """Rewrite the vendored label record from the pinned upstream checkout.

    Returns:
        The number of tasks written.

    Raises:
        RuntimeError: If the upstream checkout cannot be fetched.
    """
    labels = _labels_from_upstream()
    _labels_path().write_text(
        json.dumps(
            {
                "_comment": (
                    "Per-task DRBench stratification labels, read from the upstream commit "
                    "below. Regenerate with `python -m harbor_adapters.drbench.main "
                    "--refresh-labels`."
                ),
                "upstream_sha": UPSTREAM_SHA,
                "labels": dict(sorted(labels.items())),
            },
            indent=2,
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return len(labels)


def verify_task_labels() -> list[str]:
    """Return a description of every way the vendored labels disagree with upstream.

    Returns:
        Human-readable mismatches, empty when the record is consistent. Writes nothing.

    Raises:
        RuntimeError: If the upstream checkout cannot be fetched.
    """
    problems: list[str] = []
    recorded_sha = json.loads(_labels_path().read_text(encoding="utf-8")).get("upstream_sha")
    if recorded_sha != UPSTREAM_SHA:
        problems.append(f"records upstream_sha {recorded_sha!r}, expected {UPSTREAM_SHA!r}")
    recorded = load_task_labels()
    upstream = _labels_from_upstream()
    for task_id in sorted(set(recorded) | set(upstream)):
        if task_id not in recorded:
            problems.append(f"{task_id}: missing from the record")
        elif task_id not in upstream:
            problems.append(f"{task_id}: recorded but not in upstream")
        elif recorded[task_id] != upstream[task_id]:
            problems.append(f"{task_id}: {recorded[task_id]} != upstream {upstream[task_id]}")
    return problems


def _digests_path() -> Path:
    """Return the path of the vendored image-digest record."""
    return vendor_dir() / "image_digests.json"


def load_image_digests() -> dict[str, str]:
    """Return the vendored `task_id -> image digest` mapping.

    Returns:
        A mapping of task id to `sha256:...` digest.

    Raises:
        FileNotFoundError: If the digest record is missing.
        ValueError: If the record is malformed or holds a malformed digest.
    """
    path = _digests_path()
    if not path.is_file():
        msg = (
            f"No vendored image digests at {path}. Generate them with "
            "`python -m harbor_adapters.drbench.main --refresh-digests`."
        )
        raise FileNotFoundError(msg)
    digests = json.loads(path.read_text(encoding="utf-8")).get("digests")
    if not isinstance(digests, dict):
        msg = f"{path} must hold a `digests` object"
        raise ValueError(msg)
    for task_id, digest in digests.items():
        if not isinstance(digest, str) or _DIGEST_RE.match(digest) is None:
            msg = f"Malformed image digest for {task_id!r} in {path}: {digest!r}"
            raise ValueError(msg)
    return digests


def image_reference(task_id: str) -> str:
    """Return the digest-pinned image reference for one task.

    Pinning by digest rather than tag matters here: the upstream tags are mutable and
    published under a personal namespace, so a re-push would otherwise change eval
    results with no change on our side.

    Args:
        task_id: Identifier of the form `DR0001`.

    Returns:
        An image reference of the form `<registry>@sha256:...`.

    Raises:
        KeyError: If no digest is vendored for `task_id`.
    """
    digest = load_image_digests().get(task_id)
    if digest is None:
        msg = (
            f"No vendored image digest for {task_id!r}. Run "
            "`python -m harbor_adapters.drbench.main --refresh-digests` to add it."
        )
        raise KeyError(msg)
    return f"{IMAGE_REGISTRY}@{digest}"


def refresh_image_digests(task_ids: list[str] | None = None) -> int:
    """Resolve each task's image tag to an immutable digest and vendor the result.

    Args:
        task_ids: Task ids to resolve. Defaults to every benchmark task.

    Returns:
        The number of digests written.

    Raises:
        RuntimeError: If the registry returns no usable digest for a tag.
    """
    token = _registry_pull_token()
    resolved = {
        parse_task_id(task_id): _resolve_tag_digest(task_id, token)
        for task_id in (task_ids or available_task_ids())
    }
    _digests_path().write_text(
        json.dumps(
            {
                "_comment": (
                    "Per-task DRBench image digests, pinned so a re-push of a mutable "
                    "tag cannot silently change eval results. Regenerate with "
                    "`python -m harbor_adapters.drbench.main --refresh-digests`."
                ),
                "registry": IMAGE_REGISTRY,
                "digests": dict(sorted(resolved.items())),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return len(resolved)


def _registry_repository() -> str:
    """Return the registry path portion of the image reference."""
    return IMAGE_REGISTRY.split("/", 1)[1]


def _registry_pull_token() -> str:
    """Fetch an anonymous pull token for the DRBench image repository."""
    url = (
        f"https://ghcr.io/token?scope=repository:{_registry_repository()}:pull"
        "&service=ghcr.io"
    )
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https host
        token = json.load(response).get("token")
    if not isinstance(token, str) or not token:
        msg = "ghcr.io did not return a pull token"
        raise RuntimeError(msg)
    return token


def _resolve_tag_digest(tag: str, token: str) -> str:
    """Return the immutable digest ghcr.io reports for one already-validated tag."""
    request = urllib.request.Request(  # noqa: S310 - fixed https host, validated tag
        f"https://ghcr.io/v2/{_registry_repository()}/manifests/{tag}",
        headers={"Accept": _MANIFEST_ACCEPT, "Authorization": f"Bearer {token}"},
        method="HEAD",
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        digest = response.headers.get("Docker-Content-Digest")
    if not isinstance(digest, str) or _DIGEST_RE.match(digest) is None:
        msg = f"ghcr.io returned no usable digest for tag {tag!r}: {digest!r}"
        raise RuntimeError(msg)
    return digest


def _env_files(env_config: dict) -> list[dict]:
    """Return a task's validated document manifest."""
    env_files = env_config.get("env_files")
    if not isinstance(env_files, list):
        msg = "DRBench `env.json` must hold an `env_files` list"
        raise TypeError(msg)
    for entry in env_files:
        if not isinstance(entry, dict):
            msg = "Each DRBench `env_files` entry must be a mapping"
            raise TypeError(msg)
    return env_files


def task_apps(env_config: dict) -> list[str]:
    """Return the apps this task's documents are served from, in a stable order.

    Args:
        env_config: Parsed `env.json` for one task.

    Returns:
        Sorted app names, restricted to the apps this task actually uses.

    Raises:
        TypeError: If `env.json` has an unexpected shape.
        ValueError: If an entry names an app with no known endpoint.
    """
    apps = set()
    for entry in _env_files(env_config):
        app = entry.get("app")
        if app not in _APP_ENDPOINTS:
            msg = f"DRBench `env_files` entry declares unknown app {app!r}"
            raise ValueError(msg)
        apps.add(app)
    return sorted(apps)


def document_count(env_config: dict) -> int:
    """Return how many documents this task's manifest declares."""
    return len(_env_files(env_config))


def qa_ground_truth(eval_config: dict, qa_type: str) -> list[dict[str, str]]:
    """Extract one class of ground truth from a task's eval config.

    Mirrors upstream `QASimilarityV2.compute`, which selects entries by `qa_type` and
    scores one LLM judgement per entry. `insight` entries are the findings a report
    should contain; `distractor` entries are the planted facts it should not.

    Args:
        eval_config: Parsed `eval.json` for one task.
        qa_type: Either `insight` or `distractor`.

    Returns:
        One `{id, question, answer, type}` mapping per entry, in upstream order.

    Raises:
        TypeError: If `eval.json` has an unexpected shape.
        ValueError: If `qa_type` is not a known class.
    """
    if qa_type not in {_INSIGHT_QA_TYPE, _DISTRACTOR_QA_TYPE}:
        msg = (
            f"`qa_type` must be `{_INSIGHT_QA_TYPE}` or `{_DISTRACTOR_QA_TYPE}`, "
            f"not {qa_type!r}"
        )
        raise ValueError(msg)
    qa_list = eval_config.get("dr_report_evaluation_qa")
    if not isinstance(qa_list, list):
        msg = "DRBench `eval.json` must hold a `dr_report_evaluation_qa` list"
        raise TypeError(msg)

    entries: list[dict[str, str]] = []
    for qa in qa_list:
        if not isinstance(qa, dict):
            msg = "Each DRBench `dr_report_evaluation_qa` entry must be a mapping"
            raise TypeError(msg)
        if qa.get("qa_type") != qa_type:
            continue
        answer = qa.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            continue
        entries.append(
            {
                "id": str(qa.get("id", "")),
                "question": str(qa.get("question", "")),
                "answer": answer,
                "type": str(qa.get("type", "")),
            }
        )
    return entries


def insight_ground_truth(eval_config: dict) -> list[dict[str, str]]:
    """Return the insight-bearing ground truth recall is scored against."""
    return qa_ground_truth(eval_config, _INSIGHT_QA_TYPE)


def credential_regime(task_config: dict) -> str:
    """Return which credential regime a task falls into.

    The dataset has two, and it is upstream's doing rather than a choice of ours:

    * `persona` — the persona carries a password, so DRBench rewrote every app's login
      to the persona's. 15 of the 100 tasks. DR0001's documents sit under Nextcloud's
      `emily.patel` user.
    * `default` — the persona's `password` is `null`, so DRBench's credential override
      returns early and every app keeps its built-in login. The remaining 85 tasks.
      DR0016's documents sit under Nextcloud's `admin` user and its mailbox is
      `current.user`.

    Args:
        task_config: Parsed `task.json` for one task.

    Returns:
        Either `"persona"` or `"default"`.

    Raises:
        ValueError: If `task.json` holds no persona object.
    """
    persona = task_config.get("persona")
    if not isinstance(persona, dict):
        msg = "DRBench `task.json` must hold a `persona` object"
        raise ValueError(msg)
    password = persona.get("password")
    return "persona" if isinstance(password, str) and password else "default"


def app_credentials(task_config: dict) -> dict[str, dict[str, str]]:
    """Return the per-app login that actually works for this task.

    These are what the agent must use, and what the verifier uses to re-fetch cited
    documents when scoring factuality. They are synthetic benchmark logins baked into a
    public image, not secrets.

    Args:
        task_config: Parsed `task.json` for one task.

    Returns:
        A mapping of app name to `{username, password}`.

    Raises:
        ValueError: If `task.json` holds no persona object.
    """
    if credential_regime(task_config) == "default":
        return {app: dict(creds) for app, creds in _APP_DEFAULT_CREDENTIALS.items()}

    persona = task_config["persona"]
    username = persona.get("username")
    if not username:
        # Upstream's fallback when a persona omits `username`.
        first = persona.get("first_name", "current")
        last = persona.get("last_name", "user")
        username = f"{first}.{last}".lower()
    password = persona["password"]
    return {
        app: {"username": str(username), "password": password}
        for app in _APP_DEFAULT_CREDENTIALS
    }


def generate_task(*, output_dir: Path, task_id: str) -> Path:
    """Generate one self-contained Harbor task from its upstream DRBench record.

    Args:
        output_dir: Dataset directory that will contain the generated task.
        task_id: Identifier of the form `DR0001` or `SANITY0`.

    Returns:
        Path to the generated Harbor task directory.

    Raises:
        ValueError: If `task_id` is not a bare DRBench id, or the record declares an
            unusable manifest or persona.
        FileNotFoundError: If upstream has no config for `task_id`.
        KeyError: If no image digest is vendored for `task_id`.
        TypeError: If an upstream config has an unexpected shape.
    """
    parse_task_id(task_id)
    record = record_for_task_id(task_id)
    image = image_reference(task_id)

    task_dir = output_dir / task_id
    if task_dir.exists():
        # Regenerate cleanly: replace any existing task dir so a rerun overwrites
        # instead of failing on already-created subdirectories. `parse_task_id` proved
        # `task_id` is a single path component; re-check containment so a symlinked task
        # directory cannot redirect the removal outside the dataset.
        if task_dir.resolve().parent != output_dir.resolve():
            msg = f"Refusing to remove {task_dir}: not a direct child of {output_dir}"
            raise ValueError(msg)
        shutil.rmtree(task_dir)
    (task_dir / "environment").mkdir(parents=True)

    _write_task_files(task_dir, record=record, image=image)
    return task_dir


def populate_corpus(dataset_dir: Path) -> int:
    """Build the whole DRBench dataset into `dataset_dir`.

    No task directory is committed: every one is generated here from the configs at
    `UPSTREAM_SHA`, then given the task-invariant files that are single-sourced in
    `templates/`. App mode needs no document corpus on disk — upstream's per-task image
    serves the documents — so the fetch is only the ~2.4 MiB of config JSON.

    The name is retained because it is the contract the workflow's shared dispatcher
    calls (`--populate`), alongside the other local datasets.

    Args:
        dataset_dir: Dataset directory that will contain the generated tasks.

    Returns:
        The number of DRBench task directories built.

    Raises:
        RuntimeError: If the upstream checkout cannot be fetched.
    """
    dataset_root = dataset_dir.resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    built = 0
    for task_id in available_task_ids():
        generate_task(output_dir=dataset_root, task_id=task_id)
        built += 1
    return built


def _copy_environment_invariants(environment_dir: Path) -> None:
    """Copy the task-invariant build inputs into `environment_dir`.

    `main.Dockerfile` and the `extract_text.py` it copies in are identical across every
    task, so they are single-sourced in `templates/` and git-ignored per task.
    """
    environment_dir.mkdir(parents=True, exist_ok=True)
    templates_dir = _templates_dir()
    shutil.copy2(templates_dir / "main.Dockerfile", environment_dir / "main.Dockerfile")
    shutil.copy2(templates_dir / "extract_text.py", environment_dir / "extract_text.py")


def _copy_verifier_invariants(tests_dir: Path) -> None:
    """Write the task-invariant verifier files into `tests_dir`.

    All four are byte-identical across every task, so they are single-sourced in
    `templates/` and git-ignored per task; only `case.json` is committed. `Dockerfile`
    and `docker-compose.yaml` are here rather than under `environment/` because Harbor
    builds a SEPARATE verifier environment from the tests directory, and starts it only
    after the agent environment has been torn down.
    """
    tests_dir.mkdir(parents=True, exist_ok=True)
    templates_dir = _templates_dir()
    shutil.copy2(templates_dir / "test.sh", tests_dir / "test.sh")
    shutil.copy2(templates_dir / "judge.py", tests_dir / "judge.py")
    # This directory is a Docker build context now, so it needs the same exclusions as
    # `environment/`: nothing sensitive should be reachable by a COPY, even accidentally.
    (tests_dir / ".dockerignore").write_text(_DOCKERIGNORE, encoding="utf-8")
    shutil.copy2(
        templates_dir / "verifier-compose.yaml.tmpl", tests_dir / "docker-compose.yaml"
    )
    # The only substitution: pin the upstream commit the metrics, prompts, ground truth,
    # and corpus all come from, so it cannot drift from the fetched task configs.
    dockerfile = (templates_dir / "verifier.Dockerfile").read_text(encoding="utf-8")
    (tests_dir / "Dockerfile").write_text(
        dockerfile.format(drbench_ref=UPSTREAM_SHA), encoding="utf-8"
    )


def _company_brief(company: dict) -> str:
    """Render the company profile block shown to the agent."""
    lines = [f"- **Company:** {company.get('name', 'Unknown')}"]
    for label, key in (
        ("Industry", "industry"),
        ("Headquarters", "headquarters"),
        ("Size", "size"),
        ("Employees", "employee_count"),
        ("Annual revenue", "annual_revenue"),
        ("Market position", "market_position"),
    ):
        value = company.get(key)
        if value:
            lines.append(f"- **{label}:** {value}")
    description = company.get("description")
    if description:
        lines.append(f"- **Description:** {description}")
    for label, key in (
        ("Key products and services", "key_products_services"),
        ("Target markets", "target_markets"),
        ("Compliance certifications", "compliance_certifications"),
    ):
        value = company.get(key)
        if isinstance(value, list) and value:
            lines.append(f"- **{label}:** {'; '.join(str(item) for item in value)}")
    return "\n".join(lines)


def _persona_brief(persona: dict) -> str:
    """Render the persona block shown to the agent."""
    lines = [f"- **Name:** {persona.get('name', 'Unknown')}"]
    for label, key in (
        ("Role", "role"),
        ("Department", "department"),
        ("Seniority", "seniority"),
        ("Email", "email"),
        ("Responsibilities", "responsibilities"),
    ):
        value = persona.get(key)
        if value:
            lines.append(f"- **{label}:** {value}")
    return "\n".join(lines)


def _instruction(
    record: dict[str, dict], apps: list[str], credentials: dict[str, dict[str, str]]
) -> str:
    """Compose the task prompt: persona, company, question, apps, output contract."""
    task_config = record["task"]
    question = str(task_config.get("dr_question", "")).strip()
    date = str(task_config.get("date", "")).strip()
    company = task_config.get("company_info") or {}
    persona = task_config.get("persona") or {}

    # Credentials differ per app, and in the `default` regime they are not the persona's
    # at all, so each line names its own env var pair rather than one shared login.
    app_lines = "\n".join(
        f"- **{app}** (log in as `{credentials[app]['username']}`, password in "
        f"`$DRBENCH_{app.upper()}_PASS`) — "
        + _APP_GUIDANCE[app].replace("USER", credentials[app]["username"])
        for app in apps
    )
    return f"""# Deep research request

{question}

## Who you are

You are working on behalf of:

{_persona_brief(persona)}

## Company context

{_company_brief(company)}

{f"Today's date is {date}." if date else ""}

## Where to research

Your company's systems are running and reachable over the network. **Nothing is on this
machine's filesystem** — you have to query the applications. Each one has its own login,
with its password already exported in the environment:

{app_lines}

Documents are PDF, DOCX, XLSX, PPTX, and JSONL mail exports, so anything you download is
binary. Convert it with `extract-text <path>`. Not every document is relevant — the
systems hold unrelated material alongside what you need.

You also have internet access and a `web_search` tool. Some of what this question needs
is public information that exists nowhere in the company's systems, so research the open
web as well.

If a service seems unreachable, check `{HEALTH_URL}` — it returns 200 only when every
application is up.

## What to deliver

Write a research report to `/app/report.md` as Markdown.

- Ground every factual claim in a source, cited inline with a bracketed number
  (`[1]`, `[2]`, ...).
- End the report with a `## References` section listing each number against its source,
  in one of these exact forms. Scoring resolves each citation back to the source it
  names, and a citation it cannot resolve counts as unsupported no matter how accurate
  the claim is:
  - **a document** — its file name, e.g. `food-safety-compliance.pdf`
  - **a web page** — its full URL
  - **an email** — `RoundCube-<sender address>-<a recipient address>-<Subject>`, e.g.
    `RoundCube-david.lee@example.com-emily.patel@example.com-Re: Q2 Compliance Update`.
    Use the sender's **email address**, not their display name, and copy the subject
    line **exactly** as it appears in the mailbox — both are matched character for
    character.
  - **a chat message** — `MatterMost-<channel>-<team>-<user>`
- Report only what your sources support. Uncited assertions, and claims you cannot trace
  back to a document, email, chat message, or web page, do not count in your favour.
- Cover the question thoroughly: the report is scored on how many of the findings a
  domain expert would consider essential you actually surface, and on whether you kept
  the irrelevant material out.
"""


def _write_task_files(task_dir: Path, *, record: dict[str, dict], image: str) -> None:
    """Write every generated file for one Harbor task."""
    task_config = record["task"]
    info = record["info"]
    task_id = str(task_config["task_id"])

    apps = task_apps(record["env"])
    documents = document_count(record["env"])
    credentials = app_credentials(task_config)
    credential_env_toml = "\n".join(
        f'{name} = {json.dumps(value)}'
        for name, value in sorted(_credential_env(credentials, apps).items())
    )
    regime = credential_regime(task_config)
    insights = insight_ground_truth(record["eval"])
    distractors = qa_ground_truth(record["eval"], _DISTRACTOR_QA_TYPE)

    environment_dir = task_dir / "environment"
    compose_template = (_templates_dir() / "docker-compose.yaml.tmpl").read_text(
        encoding="utf-8"
    )
    (environment_dir / "docker-compose.yaml").write_text(
        compose_template.format(image=image),
        encoding="utf-8",
    )
    (environment_dir / ".dockerignore").write_text(_DOCKERIGNORE, encoding="utf-8")
    _copy_environment_invariants(environment_dir)

    (task_dir / "instruction.md").write_text(
        _instruction(record, apps, credentials),
        encoding="utf-8",
    )

    # The oracle writes every gold insight straight into the report, so a passing oracle
    # run proves the judge wires up (app access, case.json, reward channel) rather than
    # proving anything about research ability.
    solution_dir = task_dir / "solution"
    solution_dir.mkdir()
    oracle_report = "# Reference report\n\n" + "\n\n".join(
        f"{index}. {insight['answer']}" for index, insight in enumerate(insights, 1)
    )
    (solution_dir / "solve.sh").write_text(
        f"#!/bin/sh\nset -eu\nprintf '%s\\n' {shlex.quote(oracle_report)} > /app/report.md\n",
        encoding="utf-8",
    )

    # `case.json` is the only per-task verifier input, so it is committed; the invariant
    # verifier files are single-sourced and git-ignored (regenerated by `populate_corpus`).
    #
    # It carries no ground truth and no app credentials any more. The verifier installs
    # upstream `drbench`, which ships both the gold `eval.json` and the document corpus as
    # package data, so it needs only the task id to look them up -- and it resolves cited
    # sources from that corpus rather than from the running app stack. Keeping the answers
    # out of the task directory entirely is stricter than relying on Harbor to upload
    # `tests/` only after the agent has finished.
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    _copy_verifier_invariants(tests_dir)
    (tests_dir / "case.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                # Recorded so a score can be traced to the exact corpus and ground truth
                # it was computed against; the verifier image pins the same commit.
                "upstream_sha": UPSTREAM_SHA,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    difficulty = str(info.get("difficulty", "medium"))
    industry = str(info.get("industry", ""))
    domain = str(info.get("domain", ""))
    external = sum(1 for insight in insights if insight["type"] == "external_fact")
    (task_dir / "task.toml").write_text(
        f"""version = "1.3"

# Collect the report itself, not just the scores: without it a low score is
# undiagnosable, because a bad report and a broken environment look identical.
# This has to stay above the first table -- in TOML a top-level key written after
# `[metadata]` would belong to that table, and `artifacts` is not a field of any of
# them, so it would validate and then be silently ignored.
artifacts = ["/app/report.md"]

[metadata]
source = "drbench"
mode = "app"
task_id = "{task_id}"
industry = "{industry}"
domain = "{domain}"
difficulty = "{difficulty}"
# Which login the app stack actually accepts. `persona` tasks scope documents to the
# persona's user; `default` tasks (the majority, because upstream leaves their persona
# password null) keep each app's built-in login.
credential_regime = "{regime}"
insight_count = {len(insights)}
external_insight_count = {external}
distractor_count = {len(distractors)}
document_count = {documents}
apps = {json.dumps(apps)}

[environment]
# Public egress, unlike the context-retrieval tasks. DRBench splits its ground truth
# into enterprise facts (served by the app stack) and external facts that exist only on
# the open web, so an allowlist would make the external insights unreachable. It also
# keeps Harbor's egress sidecar out of the picture: that sidecar puts every service into
# one network namespace, which would collide the app stack's ports and drop the UDP DNS
# the services use to find each other.
network_mode = "public"
# Covers pulling the ~1.2 GiB task image, building `main`, and `compose up --wait`, all
# inside this one budget. Harbor's 600s default does not survive a cold pull.
build_timeout_sec = 2400.0

# App logins, passed to the agent through the environment instead of being written into
# `instruction.md`. These are public benchmark fixtures -- upstream publishes them and
# bakes them into its public images -- so they are not secrets; the indirection keeps a
# literal `-u <user>:<password>` out of 100 committed prompts, where it trips secret scanners
# on every push.
[environment.env]
{credential_env_toml}

# `compose up --wait` only waits for containers to be *running*, and this image declares
# no HEALTHCHECK, so it returns long before the app stack is usable. Harbor runs this
# check in `main` before it even installs the agent, so this is the real readiness gate.
[environment.healthcheck]
command = "curl -fsS {HEALTH_URL} >/dev/null"
start_period_sec = 300.0
start_interval_sec = 5.0
interval_sec = 10.0
timeout_sec = 15.0
retries = 5

[agent]
# Deep research across four applications plus open-web search; the scorecard scales this
# with `agent_timeout_multiplier`.
timeout_sec = 3600.0

[verifier]
# Four metrics: one claim-split call, one call per gold insight and per distractor, plus
# a document read and embedding pass for factuality. Includes building the verifier image
# on the first trial of a run.
timeout_sec = 2400.0
# The verifier runs in its OWN environment, built from `tests/` and started only after the
# agent environment has been stopped. That is what lets it install upstream `drbench`,
# which ships the gold `eval.json` and the whole document corpus as package data: none of
# it is ever present while the agent is running. Without this the answers would sit in the
# agent's image for the length of the run.
environment_mode = "separate"

[verifier.environment]
# Must be declared, not inherited: with this table absent Harbor deep-copies
# `[environment]`, which would boot the entire app stack a second time and then fail its
# `drbench:8099` healthcheck, since the verifier compose file has no such service.
#
# Egress is required for the judge API and for re-fetching cited web pages.
network_mode = "public"
# First trial on a runner builds the image (pandas, faiss, pymupdf, scikit-learn);
# BuildKit caches it for the rest of the run, since the file is identical across tasks.
build_timeout_sec = 1800.0
""",
        encoding="utf-8",
    )
