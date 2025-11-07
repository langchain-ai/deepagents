# deepagents cli

This is the CLI for deepagents

## Development

### Running Tests

To run the test suite:

```bash
# Install test dependencies
uv pip install -e ".[test]"

# Run all tests
pytest

# Run specific test file
pytest tests/test_tools.py

# Run with coverage
pytest --cov=deepagents_cli --cov-report=html
```

### Test Dependencies

The test suite uses:
- `pytest` - Test framework
- `responses` - Mock HTTP requests for testing
- `pytest-cov` - Coverage reporting
- `pytest-asyncio` - Async test support
