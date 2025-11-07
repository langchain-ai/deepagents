"""Test factory can create Runloop devbox."""

import os

from deepagents_cli.sandbox_factory import cleanup_sandbox, create_sandbox_backend

# Verify API key is set
api_key = os.environ.get("RUNLOOP_API_KEY")
if not api_key:
    print("❌ RUNLOOP_API_KEY not set!")
    print("Run: export RUNLOOP_API_KEY='your-key-here'")
    exit(1)

print("✓ RUNLOOP_API_KEY is set")
print("\n--- Creating Runloop devbox via factory ---")

# Create devbox
backend, devbox_id = create_sandbox_backend("runloop", None)

if not backend or not devbox_id:
    print("❌ Failed to create devbox")
    exit(1)

print(f"✓ Backend type: {type(backend).__name__}")
print(f"✓ Devbox ID: {devbox_id}")

try:
    # Test execute command
    print("\n--- Testing backend.execute() ---")
    result = backend.execute("echo 'Hello from Runloop!'")
    print("✓ Command executed")
    print(f"  Output: {result.output.strip()}")
    print(f"  Exit code: {result.exit_code}")

    if result.exit_code == 0 and "Hello from Runloop!" in result.output:
        print("\n✅ Factory test PASSED!")
    else:
        print("\n⚠️ Unexpected result")

finally:
    print("\n--- Cleaning up ---")
    cleanup_sandbox(devbox_id, "runloop")
    print("✓ Cleanup complete")
