"""Test runloop integration"""

from deepagents_cli.integrations.sandbox_factory import create_sandbox


class TestRunLoopIntegration:
    def test_sandbox_creation(self) -> None:
        with create_sandbox("runloop") as sandbox:
            assert sandbox.id is not None
            result = sandbox.execute("echo 'hello'")
            assert result.output.strip() == "hello"


class TestDaytonaIntegration:
    def test_sandbox_creation(self) -> None:
        with create_sandbox("daytona") as sandbox:
            assert sandbox.id is not None
            result = sandbox.execute("echo 'hello'")
            assert result.output.strip() == "hello"


class TestModalIntegration:
    def test_sandbox_creation(self) -> None:
        with create_sandbox("modal") as sandbox:
            assert sandbox.id is not None
            result = sandbox.execute("echo 'hello'")
            assert result.output.strip() == "hello"
