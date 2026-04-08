# Provider Plugin Starter (Wave 4 P1)

A typed scaffold for adding a new model/tool provider to deepagents.
Copy this directory, rename, and fill in the four required functions.

## Layout

```
starter_template/
  README.md            <- this file
  provider.py          <- minimal Provider implementation
  test_provider.py     <- smoke tests you should run before opening a PR
```

## Required surface

A provider must implement four things:

1. `name` — string identifier (lowercase, hyphenated)
2. `capabilities()` — returns a dict describing supported features
3. `invoke(prompt, **kwargs)` — synchronous call returning a `dict`
4. `health()` — returns `{ "ok": bool, "reason": str | None }`

See `provider.py` for the canonical signatures and inline docstrings.

## Acceptance

- `pytest test_provider.py` passes
- `python -c "from provider import StarterProvider; print(StarterProvider().capabilities())"`
  prints a dict
- The provider is registered in `deepagents/registry.py` (out of scope
  for this starter — see existing partners for the pattern)
