#!/usr/bin/env python3
"""Run the GitHub PR Review Bot webhook server."""

import os

import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def main():
    """Start the webhook server."""
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))

    print(f"Starting PR Review Bot on {host}:{port}")
    print("Webhook endpoint: POST /webhook")
    print("Health check: GET /health")

    uvicorn.run(
        "pr_review_agent.webhook:app",
        host=host,
        port=port,
        reload=os.environ.get("DEBUG", "").lower() == "true",
    )


if __name__ == "__main__":
    main()
