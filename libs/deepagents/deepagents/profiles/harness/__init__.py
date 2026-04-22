"""Harness profile package: built-in `HarnessProfile` registrations.

Individual modules register their profiles as a top-level import side effect.
The `_builtin_profiles` bootstrap imports them once at package load.
"""
