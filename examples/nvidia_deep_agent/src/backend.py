"""Backend configuration: Modal sandbox with skills/memory uploaded on creation."""

from pathlib import Path

import modal
from langchain_modal import ModalSandbox

# --- Sandbox ---
# Modal sandbox with NVIDIA RAPIDS image.
# Authenticate first: `modal setup`
#
# Sandbox type (gpu/cpu) is controlled at runtime via context_schema.
# Pass context={"sandbox_type": "cpu"} to run without GPU (cuDF falls back to pandas).
# Default is "gpu" for backward compatibility.

MODAL_SANDBOX_NAME = "nemotron-deep-agent"
modal_app = modal.App.lookup(name=MODAL_SANDBOX_NAME, create_if_missing=True)
rapids_image = (
    modal.Image.from_registry("nvcr.io/nvidia/rapidsai/base:25.02-cuda12.8-py3.12")
    # RAPIDS 25.02 ships numba-cuda 0.2.0 which has a broken device enumeration
    # that causes .to_pandas() and .describe() to crash with IndexError.
    # Upgrading to 0.28+ fixes it.
    .pip_install("numba-cuda>=0.28", "matplotlib", "seaborn")
)
cpu_image = modal.Image.debian_slim().pip_install(
    "pandas", "numpy", "scipy", "scikit-learn", "matplotlib", "seaborn"
)

SKILLS_DIR = Path("skills")
MEMORY_FILE = Path("src/AGENTS.md")


# --- Helpers ---

def _seed_sandbox(backend: ModalSandbox) -> None:
    """Upload local skill and memory files into a freshly created sandbox.

    In production, replace the local file reads with your storage layer
    (S3, database, etc.).
    """
    files: list[tuple[str, bytes]] = []

    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        files.append(
            (f"/skills/{skill_dir.name}/SKILL.md", skill_md.read_bytes())
        )

    if MEMORY_FILE.exists():
        files.append(("/memory/AGENTS.md", MEMORY_FILE.read_bytes()))

    if not files:
        return

    # Create parent directories inside the sandbox, then upload
    dirs = sorted({str(Path(p).parent) for p, _ in files})
    backend.execute(f"mkdir -p {' '.join(dirs)}")
    backend.upload_files(files)


# --- Backend Factory ---

def create_backend(runtime):
    """Create a ModalSandbox backend with skills and memory pre-loaded.

    On first sandbox creation, skill and memory files are uploaded from the
    local filesystem into the sandbox. The agent reads and edits them directly
    inside the sandbox; changes persist for the sandbox's lifetime.

    In production, swap the local file reads in `_seed_sandbox` for your
    storage layer (S3, database, etc.).
    """
    ctx = runtime.context or {}
    sandbox_type = ctx.get("sandbox_type", "gpu")
    use_gpu = sandbox_type == "gpu"
    sandbox_name = f"{MODAL_SANDBOX_NAME}-{sandbox_type}"

    created = False
    try:
        sandbox = modal.Sandbox.from_name(MODAL_SANDBOX_NAME, sandbox_name)
    except modal.exception.NotFoundError:
        create_kwargs = dict(
            app=modal_app,
            workdir="/workspace",
            name=sandbox_name,
            timeout=3600,       # 1 hour max lifetime
            idle_timeout=1800,  # 30 min idle before auto-terminate
        )
        if use_gpu:
            create_kwargs["image"] = rapids_image
            create_kwargs["gpu"] = "A10G"
        else:
            create_kwargs["image"] = cpu_image
        sandbox = modal.Sandbox.create(**create_kwargs)
        created = True

    backend = ModalSandbox(sandbox=sandbox)
    if created:
        _seed_sandbox(backend)
    return backend
