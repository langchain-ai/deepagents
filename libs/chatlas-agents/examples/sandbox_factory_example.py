"""Example demonstrating sandbox factory functions.

This example shows how to use the new create_docker_sandbox and create_apptainer_sandbox
factory functions that follow the deepagents-cli pattern for lifecycle management.
"""

import os
from chatlas_agents.sandbox import create_docker_sandbox, create_apptainer_sandbox
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model


def docker_factory_example():
    """Example using create_docker_sandbox factory."""
    
    print("=" * 60)
    print("Docker Sandbox Factory Example")
    print("=" * 60)
    
    # Using the factory function with context manager
    # This automatically handles cleanup when done
    with create_docker_sandbox(
        image="python:3.13-slim",
        working_dir="/workspace",
        auto_remove=True,
    ) as backend:
        # Backend is ready to use
        print(f"✓ Sandbox ID: {backend.id[:12]}")
        
        # Create agent with the sandbox backend
        # Initialize model using init_chat_model to avoid AttributeError
        model = init_chat_model("anthropic:claude-sonnet-4-5-20250929")
        
        agent = create_deep_agent(
            model=model,
            backend=backend,
        )
        
        # Run a simple task
        agent.invoke({
            "messages": [{"role": "user", "content": "Echo 'Hello from Docker!' to a file"}]
        })
        
        print("Agent completed task")
    
    print("✓ Sandbox cleaned up automatically")


def apptainer_factory_example():
    """Example using create_apptainer_sandbox factory."""
    
    print("\n" + "=" * 60)
    print("Apptainer Sandbox Factory Example")
    print("=" * 60)
    
    # Using the factory function with context manager
    # This is ideal for HPC environments like CERN lxplus
    with create_apptainer_sandbox(
        image="docker://python:3.13-slim",
        working_dir="/workspace",
        auto_remove=True,
    ) as backend:
        # Backend is ready to use
        print(f"✓ Sandbox ID: {backend.id}")
        
        # Create agent with the sandbox backend
        # Initialize model using init_chat_model to avoid AttributeError
        model = init_chat_model("anthropic:claude-sonnet-4-5-20250929")
        
        agent = create_deep_agent(
            model=model,
            backend=backend,
        )
        
        # Run a simple task
        agent.invoke({
            "messages": [{"role": "user", "content": "Create a Python script that prints system info"}]
        })
        
        print("Agent completed task")
    
    print("✓ Sandbox cleaned up automatically")


def with_setup_script_example():
    """Example using a setup script to configure the sandbox."""
    
    print("\n" + "=" * 60)
    print("Sandbox with Setup Script Example")
    print("=" * 60)
    
    # Create a temporary setup script
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write("""#!/bin/bash
# Setup script example
pip install numpy pandas
echo "Setup complete!"
""")
        setup_script = f.name
    
    try:
        with create_docker_sandbox(
            image="python:3.13-slim",
            setup_script_path=setup_script,
        ) as backend:
            print(f"✓ Sandbox ready with packages installed")
            
            # Verify packages are installed
            result = backend.execute("pip list | grep numpy")
            print(f"Verification: {result.output.strip()}")
    finally:
        os.unlink(setup_script)


def atlas_container_example():
    """Example for ATLAS physics analysis on lxplus."""
    
    print("\n" + "=" * 60)
    print("ATLAS Container Example (for lxplus)")
    print("=" * 60)
    
    # This example shows how to use ATLAS software containers
    # Note: Requires Apptainer and access to ATLAS container images
    try:
        with create_apptainer_sandbox(
            # ATLAS Athanalysis container (example - use actual ATLAS image)
            image="docker://atlas/athanalysis:latest",
            working_dir="/workspace",
        ) as backend:
            print(f"✓ ATLAS sandbox ready: {backend.id}")
            
            # Now you can use ATLAS software in the agent
            # Initialize model using init_chat_model to avoid AttributeError
            model = init_chat_model("anthropic:claude-sonnet-4-5-20250929")
            
            agent = create_deep_agent(
                model=model,
                backend=backend,
                system_prompt=(
                    "You are an ATLAS physics analysis assistant. "
                    "You have access to the ATLAS software environment."
                )
            )
            
            # Agent can now run ATLAS commands
            agent.invoke({
                "messages": [{
                    "role": "user",
                    "content": "Check what ATLAS release is available"
                }]
            })
            
            print("Agent completed ATLAS task")
    except Exception as e:
        print(f"Note: ATLAS container not available in this environment: {e}")


if __name__ == "__main__":
    # Run Docker example if Docker is available
    import shutil
    if shutil.which('docker'):
        print("\n🐳 Running Docker examples...")
        docker_factory_example()
        with_setup_script_example()
    else:
        print("⚠️  Docker not available, skipping Docker examples")
    
    # Run Apptainer example if Apptainer is available
    if shutil.which('apptainer'):
        print("\n🔷 Running Apptainer examples...")
        apptainer_factory_example()
        atlas_container_example()
    else:
        print("⚠️  Apptainer not available, skipping Apptainer examples")
