"""Setup script for chatlas-agents."""

from setuptools import setup, find_packages

setup(
    name="chatlas-agents",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "deepagents>=0.3.0",
        "langchain>=0.3.0",
        "langchain-openai>=0.2.0",
        "langchain-anthropic>=0.2.0",
        "langchain-groq>=0.2.0",
        "langgraph>=1.0.0",
        "langgraph-checkpoint>=2.0.0",
        "httpx>=0.27.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.0",
        "typer>=0.12.0",
        "rich>=13.0.0",
        "deepagents-cli>=0.0.3",
        "langchain-mcp-adapters>=0.2.1",
    ],
    entry_points={
        "console_scripts": [
            "chatlas=chatlas_agents.cli:main",
        ],
    },
    python_requires=">=3.10",
)
