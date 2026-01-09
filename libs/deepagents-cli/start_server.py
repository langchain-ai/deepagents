"""Start textual-serve for the mock app."""
import os
os.chdir("/Users/vivektrivedy/Documents/da5/deepagents/libs/deepagents-cli")

from textual_serve.server import Server

server = Server(
    command="uv run python serve_mock.py",
    host="127.0.0.1",
    port=8000,
)
server.serve()
