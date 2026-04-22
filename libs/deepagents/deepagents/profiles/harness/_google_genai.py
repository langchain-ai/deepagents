"""Built-in Google GenAI harness profile.

Appends a short system-prompt suffix that reminds the model to batch
independent tool calls, which Gemini supports natively. Users may layer
additional harness fields on top via `register_harness_profile("google_genai",
...)`.
"""

from deepagents.profiles.harness_profiles import HarnessProfile, register_harness_profile

_PARALLEL_TOOLS_SUFFIX = "Gemini supports parallel tool calls; emit independent calls together to minimize round-trips."


def register() -> None:
    """Register the built-in Google GenAI harness profile."""
    register_harness_profile(
        "google_genai",
        HarnessProfile(system_prompt_suffix=_PARALLEL_TOOLS_SUFFIX),
    )
