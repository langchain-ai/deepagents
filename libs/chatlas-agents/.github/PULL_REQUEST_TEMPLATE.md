# Pull Request Template

## Description

Please include a summary of the changes and related context. Explain the motivation for this PR.

### Type of Change

- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Changes Made

- Change 1
- Change 2
- Change 3

## Testing

Describe the tests you've run to verify your changes:

```bash
# Example test commands
uv run pytest tests/

# Manual testing
uv run python -m chatlas_agents.cli run --input "test query"
```

### Test Results

- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed
- [ ] Verified with `--verbose` flag

## Verification Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated (if needed)
- [ ] No breaking changes (or documented if intentional)
- [ ] Tested with multiple queries/scenarios

## Environment Tested

- Python: [e.g., 3.11, 3.13]
- OS: [e.g., Linux, macOS]
- Configuration: [any special setup needed]

## Related Issues

Closes #(issue number) or References #(issue number)

## Screenshots/Examples

If applicable, include screenshots or example output:

```
[Paste example output here]
```

## Additional Notes

Any additional information that reviewers should know.
