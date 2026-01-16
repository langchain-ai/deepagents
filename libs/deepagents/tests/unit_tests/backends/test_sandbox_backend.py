import subprocess
from pathlib import Path

from deepagents.backends.sandbox import BaseSandbox
from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse


class TestSandbox(BaseSandbox):
    """Test sandbox implementation that executes commands locally using subprocess."""
    
    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        try:
            # Execute the command using subprocess with shell=True for bash compatibility
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            # Combine stdout and stderr as expected by ExecuteResponse
            combined_output = ""
            if result.stdout:
                combined_output += result.stdout
            if result.stderr:
                if combined_output:
                    combined_output += "\n"
                combined_output += result.stderr
            
            return ExecuteResponse(
                output=combined_output,
                exit_code=result.returncode,
                truncated=False
            )
            
        except Exception as exc:
            return ExecuteResponse(
                output=str(exc),
                exit_code=1,
                truncated=False
            )

    def id(self) -> str:
        return ""

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return []

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return []


def test_test_backend_normal_mode(tmp_path: Path):
    be = TestSandbox()

    root = tmp_path
    f1 = root / "a.txt"
    f2 = root / "dir" / "b.py"
    be.write(f1, "hello fs")
    be.write(f2, "print('x')\nhello")

    # ls_info absolute path - should only list files in root, not subdirectories
    infos = be.ls_info(str(root))
    paths = {i["path"] for i in infos}
    assert str(f1) in paths  # File in root should be listed
    assert str(f2) not in paths  # File in subdirectory should NOT be listed
    assert (str(root) + "/dir") in paths  # Directory should be listed
