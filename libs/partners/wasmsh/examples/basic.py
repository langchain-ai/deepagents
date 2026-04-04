"""Basic langchain-wasmsh example."""

from langchain_wasmsh import WasmshSandbox


def main() -> None:
    """Run a basic wasmsh sandbox example."""
    backend = WasmshSandbox(
        initial_files={"/workspace/data.txt": b"hello from wasmsh\n"}
    )
    try:
        cmd = (
            "cat data.txt && python3 -c "
            "\"print(open('/workspace/data.txt').read().strip())\""
        )
        result = backend.execute(cmd)
        print(result.output)  # noqa: T201
    finally:
        backend.close()


if __name__ == "__main__":
    main()
