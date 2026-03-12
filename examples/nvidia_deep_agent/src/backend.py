"""Backend configuration: Modal sandbox + FilesystemBackend for memory/skills."""

import modal
from deepagents.backends import CompositeBackend, FilesystemBackend
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


# --- Backend Factory ---

def create_backend(runtime):
    """Create a CompositeBackend: Modal sandbox + filesystem for memory/skills.

    Memory and skills are stored on the local filesystem via FilesystemBackend.
    The agent can read and edit these files directly, and changes persist
    across restarts (useful for committing agent improvements back to git).
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

    return CompositeBackend(
        default=ModalSandbox(sandbox=sandbox),
        routes={
            "/memory/": FilesystemBackend(root_dir="./src", virtual_mode=True),
            "/skills/": FilesystemBackend(root_dir="./skills", virtual_mode=True),
        },
    )
