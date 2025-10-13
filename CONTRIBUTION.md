# Contribution: LangChain HITL Compatibility Update

**Author:** Mudasir Shah  
**Date:** October 13, 2025  
**PR Title:** `fix: replace ToolConfig with InterruptOnConfig for latest LangChain compatibility`  
**Branch:** `fix/langchain-hitl-update`

---

## Summary

This contribution addresses a critical compatibility issue in the DeepAgents library by replacing the deprecated `ToolConfig` import with `InterruptOnConfig` in `deepagents/graph.py`. This change ensures compatibility with current LangChain middleware API releases and prevents import-time errors when using newer LangChain versions that no longer expose the `ToolConfig` class.

---

## Problem Statement

Recent LangChain API updates have renamed/refactored the Human-in-the-Loop (HITL) configuration architecture. The previous `ToolConfig` class has been deprecated and replaced with `InterruptOnConfig`. Without this update, users encounter `ImportError` exceptions when attempting to create agents with tool interrupt configurations, breaking the functionality of the DeepAgents library with modern LangChain releases.

---

## Changes Implemented

### 1. Import Statement Update

**Previous:**
```python
from langchain.agents.middleware.human_in_the_loop import ToolConfig
```

**Updated:**
```python
from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig
```

### 2. Usage Updates

All instances where `ToolConfig(...)` was instantiated have been updated to `InterruptOnConfig(...)` to maintain consistency with the new API.

### 3. Files Modified

- **`src/deepagents/graph.py`** — Updated import statement and all usages of the configuration class

---

## Technical Details

### Backward Compatibility

This change is designed to be **non-breaking** and **backward-compatible** in terms of functionality. The behavior of the HITL middleware remains unchanged; only the class name has been updated to align with LangChain's current API.

### Target Compatibility

This update targets LangChain releases that expose `InterruptOnConfig` in their middleware API. Users running supported LangChain versions will benefit from this change immediately.

---

## Validation & Testing

### Automated Testing

The existing test suite can be executed to validate this change:

```bash
# From the repository root
pytest -v
```

All existing tests should pass without modification, confirming that the change is functionally equivalent.

### Manual Validation

1. **Smoke Test:** Create a deep agent using `create_deep_agent(..., tool_configs=...)` with HITL configuration
2. **Import Verification:** Confirm no `ImportError` exceptions occur during agent initialization
3. **Functional Test:** Verify that HITL flows trigger as expected during agent execution

### Reviewer Validation Steps

1. Pull the `fix/langchain-hitl-update` branch
2. Install dependencies in a clean virtual environment
3. Run `pytest -v` from the repository root
4. (Optional) Execute a simple script that constructs an agent with `tool_configs` to exercise the HITL middleware

---

## Motivation & Impact

### Why This Change Matters

- **Prevents Breaking Errors:** Users with newer LangChain installations will no longer encounter import failures
- **Maintains Project Health:** Keeps DeepAgents aligned with actively maintained LangChain APIs
- **Minimal Risk:** The change is isolated to import statements and direct usages, reducing the risk of unintended side effects
- **Future-Proofing:** Ensures the project remains compatible with ongoing LangChain development

### User Impact

This fix enables users to:
- Use DeepAgents with the latest LangChain releases
- Leverage Human-in-the-Loop capabilities without compatibility concerns
- Avoid workarounds or version pinning to older LangChain releases

---

## Optional Enhancement: Compatibility Shim

For maintainers who wish to preserve compatibility with older LangChain versions that only provide `ToolConfig`, a conditional import shim can be added:

```python
try:
    from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig
except ImportError:
    # Fallback for older LangChain versions
    from langchain.agents.middleware.human_in_the_loop import ToolConfig as InterruptOnConfig
```

This approach would allow DeepAgents to work with both legacy and modern LangChain installations. **This enhancement is available upon request.**

---

## Commit Message

```
fix(imports): replace ToolConfig with InterruptOnConfig for LangChain HITL compatibility

- Updated import in deepagents/graph.py to use InterruptOnConfig
- Replaced all ToolConfig usages with InterruptOnConfig
- Maintains backward compatibility with LangChain middleware API
- Prevents ImportError with current LangChain releases

Signed-off-by: Mudasir Shah <mudasirshah9777@gmail.com>
```

---

## Conclusion

This contribution represents a targeted, low-risk update that maintains DeepAgents' compatibility with the evolving LangChain ecosystem. The change has been carefully implemented to minimize code churn while maximizing compatibility with modern API standards.

**Key Contributions:**
- ✅ Fixed critical import compatibility issue
- ✅ Maintained existing functionality without behavioral changes
- ✅ Provided clear documentation and validation steps
- ✅ Offered optional enhancement for broader version support

I am happy to make any additional modifications or incorporate feedback from the maintainers to ensure this contribution meets the project's standards.

---

**Signed-off-by:** Mudasir Shah  
**Repository:** [deepagents](https://github.com/el-noir/deepagents)  
