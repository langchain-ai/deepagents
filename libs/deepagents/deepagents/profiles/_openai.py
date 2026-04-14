"""OpenAI provider harness profile.

!!! warning

    This is an internal API subject to change without deprecation. It is not
    intended for external use or consumption.
"""

from deepagents.profiles._harness_profiles import _HarnessProfile, _register_harness_profile

_register_harness_profile(
    "openai",
    _HarnessProfile(init_kwargs={"use_responses_api": True}),
)
