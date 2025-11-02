"""Integration test for Ollama with qwen2.5-coder:14b."""

import os

# Set Ollama environment variables
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
os.environ["OLLAMA_MODEL"] = "qwen2.5-coder:14b"

# Clear other model API keys to ensure Ollama is used
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)

import sys
sys.path.insert(0, "libs/deepagents-cli")

from deepagents_cli.config import create_model

print("Creating Ollama model...")
model = create_model()

print(f"\nModel type: {type(model).__name__}")
print(f"Model name: {model.model}")
print(f"Base URL: {model.base_url}")
print(f"Temperature: {model.temperature}")

print("\n✅ Model created successfully!")

# Test a simple invocation
print("\nTesting model invocation with a simple code question...")
response = model.invoke("Write a Python function to calculate fibonacci numbers. Keep it very brief.")
print(f"\nResponse: {response.content[:200]}...")

print("\n✅ Integration test passed! Ollama with qwen2.5-coder:14b is working!")
