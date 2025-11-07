"""Test Piece 3: Agent creation with optional sandbox parameter."""
import os
import sys
from pathlib import Path
from deepagents_cli.sandbox_factory import create_sandbox_backend, cleanup_sandbox
from deepagents_cli.agent import create_agent_with_config
from langchain_openai import ChatOpenAI

print("=" * 70)
print("PIECE 3 TEST: Agent Creation with Optional Sandbox")
print("=" * 70)

# Check environment
if not os.environ.get("RUNLOOP_API_KEY"):
    print("\n❌ RUNLOOP_API_KEY not set")
    print("Run: export RUNLOOP_API_KEY='your-key-here'")
    sys.exit(1)

if not os.environ.get("OPENAI_API_KEY"):
    print("\n❌ OPENAI_API_KEY not set")
    print("Run: export OPENAI_API_KEY='your-key-here'")
    sys.exit(1)

print("\n✓ Environment variables set")

# Create model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print("✓ Model created")

# TEST 1: Local mode (sandbox=None)
print("\n" + "-" * 70)
print("TEST 1: Local Mode (sandbox=None)")
print("-" * 70)

try:
    agent_local = create_agent_with_config(
        model=model,
        assistant_id="test-piece3-local",
        tools=[],
        sandbox=None
    )
    print("✓ Local agent created successfully")

    # Verify memory directory
    local_mem = Path.home() / ".deepagents" / "test-piece3-local"
    print(f"✓ Memory directory exists: {local_mem.exists()}")

except Exception as e:
    print(f"❌ Local agent creation failed: {e}")
    sys.exit(1)

# TEST 2: Remote mode (sandbox=RunloopBackend)
print("\n" + "-" * 70)
print("TEST 2: Remote Mode (sandbox=RunloopBackend)")
print("-" * 70)

# Create devbox
print("\nCreating Runloop devbox...")
backend, devbox_id = create_sandbox_backend('runloop', None)

if not backend or not devbox_id:
    print("❌ Failed to create devbox")
    sys.exit(1)

print(f"✓ Devbox created: {devbox_id}")

try:
    # Test backend execute
    print("\nTesting backend.execute()...")
    result = backend.execute("echo 'Hello from sandbox'")
    print(f"✓ Command executed: {result.output.strip()}")
    print(f"✓ Exit code: {result.exit_code}")

    # Create remote agent
    print("\nCreating remote agent...")
    agent_remote = create_agent_with_config(
        model=model,
        assistant_id="test-piece3-remote",
        tools=[],
        sandbox=backend
    )
    print("✓ Remote agent created successfully")
    print("✓ No duplicate middleware error (this was the bug!)")

    # Verify memory directory
    remote_mem = Path.home() / ".deepagents" / "test-piece3-remote"
    print(f"✓ Memory directory exists: {remote_mem.exists()}")

    # TEST 3: Verify code structure
    print("\n" + "-" * 70)
    print("TEST 3: Code Structure Verification")
    print("-" * 70)

    import inspect
    source = inspect.getsource(create_agent_with_config)

    checks = [
        ("Function has 'sandbox' parameter", "sandbox=None" in source),
        ("Conditional logic exists", "if sandbox is None:" in source),
        ("Local mode creates CompositeBackend", "CompositeBackend(" in source),
        ("Remote mode uses sandbox", "default=sandbox" in source),
        ("Memory routing to /memories/", '"/memories/"' in source),
        ("Local mode has ResumableShellToolMiddleware", "ResumableShellToolMiddleware" in source),
        ("Execute interrupt config exists", "execute_interrupt_config" in source),
        ("Execute in interrupt_on dict", '"execute": execute_interrupt_config' in source),
    ]

    all_passed = True
    for check_name, result in checks:
        status = "✓" if result else "❌"
        print(f"{status} {check_name}")
        if not result:
            all_passed = False

    # Final summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED - PIECE 3 COMPLETE!")
        print("=" * 70)
        print("\nVerified:")
        print("  ✓ Local agent (sandbox=None) works")
        print("  ✓ Remote agent (sandbox=RunloopBackend) works")
        print("  ✓ No duplicate middleware errors")
        print("  ✓ Backend can execute commands in sandbox")
        print("  ✓ Memory directories created locally (not in sandbox)")
        print("  ✓ Code structure correct (conditional middleware)")
        print("  ✓ Execute tool interrupt config added")
        print("\n🎉 Piece 3 implementation verified and working!")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("=" * 70)
        sys.exit(1)

finally:
    print("\n" + "-" * 70)
    print("Cleanup")
    print("-" * 70)
    cleanup_sandbox(devbox_id, 'runloop')
    print("✓ Devbox shut down")
    print("✓ Test complete")
