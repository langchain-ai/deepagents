"""Backend configuration: Modal sandbox with uploaded memory and skills."""

from __future__ import annotations

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

# --- Local assets to upload into the sandbox ---
_EXAMPLE_DIR = Path(__file__).resolve().parent.parent


def _collect_uploads() -> list[tuple[str, bytes]]:
    """Read local memory and skills files to upload into the sandbox."""
    files: list[tuple[str, bytes]] = []

    # Memory: src/AGENTS.md → /memory/AGENTS.md
    agents_md = _EXAMPLE_DIR / "src" / "AGENTS.md"
    if agents_md.exists():
        files.append(("/memory/AGENTS.md", agents_md.read_bytes()))

    # Skills: skills/**/* → /skills/**/*
    skills_dir = _EXAMPLE_DIR / "skills"
    if skills_dir.is_dir():
        for path in sorted(skills_dir.rglob("*")):
            if path.is_file():
                rel = path.relative_to(skills_dir)
                files.append((f"/skills/{rel}", path.read_bytes()))

    return files


# --- Backend Factory ---


def create_backend(runtime):
    """Create a ModalSandbox backend with uploaded memory and skills.

    Local memory and skills files are uploaded into the sandbox so the
    agent reads them remotely instead of accessing the host filesystem.
    """
    ctx = runtime.context or {}
    sandbox_type = ctx.get("sandbox_type", "gpu")
    use_gpu = sandbox_type == "gpu"
    sandbox_name = f"{MODAL_SANDBOX_NAME}-{sandbox_type}"

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

    backend = ModalSandbox(sandbox=sandbox)

    # Upload local memory and skills into the sandbox
    files = _collect_uploads()
    if files:
        dirs = sorted({str(Path(p).parent) for p, _ in files})
        backend.execute(f"mkdir -p {' '.join(dirs)}")
        backend.upload_files(files)

    return backend
