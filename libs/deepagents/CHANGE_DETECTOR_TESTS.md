# Change-detector test audit

This file tracks tests in `libs/deepagents` that match the definition in
[Change-Detector Tests Considered Harmful](https://testing.googleblog.com/2015/01/testing-on-toilet-change-detector-tests.html):
they primarily restate the current implementation or assert incidental internal
interactions, so a behavior-preserving refactor can make them fail without proving
that behavior regressed.

## Audit scope

- Baseline: `1a787808ac81a7072b54cf2453a8c04fa4dd32f5` (`main`)
- Test files reviewed: 79 Python files under `tests/`
- Test functions reviewed: 2,463
- Tests tracked below: 49
- Integration tests and benchmarks tracked below: 0

The audit is intentionally narrower than a general test-quality review. A test is
not listed merely because it uses mocks, accesses a private helper, is redundant,
is brittle, or has weak assertions. Snapshot tests for model-facing output,
performance tests, security invariants, public API-shape tests, and assertions at
external protocol boundaries are also excluded when they verify a meaningful
contract.

`Rewrite` means preserve the intended requirement but prove it through observable
behavior. `Delete` means equivalent behavioral coverage already exists or the
asserted implementation detail is not a contract. All entries are open until their
source test is rewritten or removed.

## Backends

| Status | Test | Why it is a change detector | Action |
| --- | --- | --- | --- |
| Open | `tests/unit_tests/backends/test_protocol.py::TestAgrepTimeout::test_agrep_timeout_exceeds_two_sync_grep_phases` | Compares two timeout constants instead of exercising the timeout behavior; changing the search implementation or budget model can preserve behavior and fail this inequality. | Rewrite |
| Open | `tests/unit_tests/backends/test_file_format.py::test_compile_glob_is_cached` | Uses object identity to pin `lru_cache` as the memoization mechanism rather than measuring matching behavior or performance. | Delete |
| Open | `tests/unit_tests/backends/test_timeout_compat.py::TestExecuteAcceptsTimeout::test_result_is_cached` | Reads `cache_info().hits`, directly coupling the test to the `lru_cache` wrapper. | Delete |
| Open | `tests/unit_tests/backends/test_composite_backend.py::test_composite_backend_supports_execution_check` | Asserts only that the class has an `execute` attribute, which restates its static structure and does not establish execution support. | Delete |
| Open | `tests/unit_tests/backends/test_langsmith_sandbox.py::test_max_binary_bytes_constant_matches_template` | Searches a private command template for a copied constant literal; generating or sharing the value differently would preserve the size limit and fail the test. | Rewrite |
| Open | `tests/unit_tests/backends/test_langsmith_sandbox.py::test_max_output_bytes_constant_matches_template` | Searches a private command template for a copied constant literal rather than exercising the output cap. | Rewrite |
| Open | `tests/unit_tests/backends/test_local_shell_backend.py::test_local_shell_backend_execute_starts_new_session` | Verifies the exact `subprocess.run(start_new_session=True)` mechanism; the sibling controlling-terminal probe already proves the security property behaviorally. | Delete |
| Open | `tests/unit_tests/backends/test_filesystem_backend.py::test_glob_backend_budget_below_middleware_deadline` | Pins a relationship between private timeout constants; a different implementation can still return partial results before the outer deadline. | Rewrite |
| Open | `tests/unit_tests/backends/test_sandbox_backend.py::test_grep_path_glob_template_strips_leading_slash` | Greps private template source for `lstrip`, variable names, and a particular `glob.glob` call instead of testing anchored-glob results. | Delete |
| Open | `tests/unit_tests/backends/test_sandbox_backend.py::test_grep_path_glob_template_terminates_each_record` | Greps private template source for `rstrip` and `line.rstrip`, so an equivalent record-framing implementation fails it. | Delete |
| Open | `tests/unit_tests/backends/test_sandbox_backend.py::test_edit_command_template_ends_with_newline` | Checks the final character of a private heredoc template; an implementation that avoids the heredoc can preserve edit behavior and fail this test. | Rewrite |

## Middleware

| Status | Test | Why it is a change detector | Action |
| --- | --- | --- | --- |
| Open | `tests/unit_tests/middleware/test_compact_tool.py::TestCompactBackendUsage::test_static_backend_is_passed_to_offload` | Patches the private offload helper and verifies its exact backend argument instead of observing where compacted history is written. | Rewrite |
| Open | `tests/unit_tests/middleware/test_compact_tool.py::TestIsEligibleForCompaction::test_dict_trigger_constructs_langchain_trigger_clauses` | Asserts a private LangChain helper's `_trigger_clauses` representation; neighboring tests already exercise the trigger semantics. | Delete |
| Open | `tests/unit_tests/middleware/test_memory_middleware.py::test_memory_middleware_with_state_backend` | Reads back the private backend type and constructor arguments without loading or rendering memory. | Rewrite |
| Open | `tests/unit_tests/middleware/test_memory_middleware.py::test_memory_middleware_with_store_backend_instance` | Reads back the private backend type and constructor arguments without exercising store-backed memory. | Rewrite |
| Open | `tests/unit_tests/middleware/test_rubric_middleware.py::TestConstruction::test_defaults` | Primarily mirrors constructor assignments through private `_model`, `_tools`, and `_system_prompt` attributes. | Rewrite |
| Open | `tests/unit_tests/middleware/test_rubric_middleware.py::TestConstruction::test_tools_default_to_empty` | Asserts only the private `_tools` storage representation. | Delete |
| Open | `tests/unit_tests/middleware/test_rubric_middleware.py::TestConstruction::test_tools_propagated` | Verifies that a constructor argument was copied into `_tools`, not that the grader can use the configured tool. | Rewrite |
| Open | `tests/unit_tests/middleware/test_rubric_middleware.py::TestConstruction::test_custom_system_prompt_stored` | Verifies private prompt storage rather than the prompt sent to the grader model. | Rewrite |
| Open | `tests/unit_tests/middleware/test_skills_middleware.py::test_skills_middleware_with_state_backend` | Reads back private backend type and source-list storage without loading any skills. | Rewrite |
| Open | `tests/unit_tests/middleware/test_skills_middleware.py::test_skills_middleware_with_store_backend_instance` | Reads back private backend type and source-list storage without exercising store-backed skills. | Rewrite |
| Open | `tests/unit_tests/middleware/test_subagent_middleware_init.py::TestSubagentMiddlewareInit::test_middleware_delegates_to_create_sub_agent` | Replaces an internal helper and verifies the exact delegation and argument instead of invoking the resulting subagent. | Rewrite |
| Open | `tests/unit_tests/middleware/test_summarization_factory.py::test_factory_uses_profile_based_defaults` | Reads private helper configuration produced by the factory instead of showing when summarization and argument truncation occur. | Rewrite |
| Open | `tests/unit_tests/middleware/test_summarization_factory.py::test_factory_uses_fallback_defaults_without_profile` | Reads private helper configuration produced by the fallback branch instead of exercising the thresholds. | Rewrite |
| Open | `tests/unit_tests/middleware/test_summarization_factory.py::test_factory_surfaces_summarization_knobs` | Verifies that public arguments were copied into private helper attributes, not that the custom prompt, trim limit, and counter affect summarization. | Rewrite |
| Open | `tests/unit_tests/middleware/test_summarization_middleware.py::TestSummarizationMiddlewareInit::test_init_with_backend` | Asserts private backend identity and a derived private path prefix without performing an offload. | Rewrite |
| Open | `tests/unit_tests/middleware/test_summarization_middleware.py::TestTokenCountingEfficiency::test_token_counter_called_once_per_model_call` | Pins the exact invocation count of an internal token-counting path without measuring the intended efficiency property. | Rewrite |
| Open | `tests/unit_tests/middleware/test_summarization_middleware.py::TestTokenCountingEfficiency::test_token_counter_called_once_per_model_call_async` | Async equivalent of the exact internal call-count assertion. | Rewrite |
| Open | `tests/unit_tests/middleware/test_summarization_middleware.py::TestTokenCountingEfficiency::test_token_counter_recounts_when_truncation_modifies_messages` | Its exact two-call assertion pins internal sequencing; retain the truncation assertion but replace the call-count check with a behavioral result. | Rewrite |
| Open | `tests/unit_tests/middleware/test_summarization_middleware.py::TestTokenCountingEfficiency::test_token_counter_recounts_when_truncation_modifies_messages_async` | Async equivalent of the exact internal recount-sequence assertion. | Rewrite |

## Top-level unit tests

| Status | Test | Why it is a change detector | Action |
| --- | --- | --- | --- |
| Open | `tests/unit_tests/test_artifacts_root.py::TestFilesystemMiddlewareArtifactsRoot::test_default_prefixes` | Reads two private cached path prefixes rather than evicting and retrieving an artifact. | Delete |
| Open | `tests/unit_tests/test_artifacts_root.py::TestFilesystemMiddlewareArtifactsRoot::test_custom_artifacts_root_from_composite_backend` | Reads private derived prefixes; custom-root eviction is already covered behaviorally in the same file. | Delete |
| Open | `tests/unit_tests/test_artifacts_root.py::TestFilesystemMiddlewareArtifactsRoot::test_trailing_slash_normalized` | Pins private string normalization instead of proving that a trailing-slash root produces usable artifact paths. | Rewrite |
| Open | `tests/unit_tests/test_artifacts_root.py::TestFilesystemMiddlewareArtifactsRoot::test_root_slash_no_double_slash` | Duplicates the private default-prefix readback. | Delete |
| Open | `tests/unit_tests/test_artifacts_root.py::TestCreateSummarizationMiddlewareArtifactsRoot::test_default_history_path_prefix` | Reads a private history prefix without offloading conversation history. | Rewrite |
| Open | `tests/unit_tests/test_artifacts_root.py::TestCreateSummarizationMiddlewareArtifactsRoot::test_custom_artifacts_root_from_composite_backend` | Reads a private derived prefix; custom-root history offload has end-to-end coverage elsewhere. | Delete |
| Open | `tests/unit_tests/test_artifacts_root.py::TestCreateSummarizationMiddlewareArtifactsRoot::test_trailing_slash_normalized` | Pins private path-normalization storage rather than the history file's observable location. | Rewrite |
| Open | `tests/unit_tests/test_artifacts_root.py::TestCreateSummarizationMiddlewareArtifactsRoot::test_root_slash_no_double_slash` | Reads the private default prefix rather than exercising a root-path history offload. | Rewrite |
| Open | `tests/unit_tests/test_artifacts_root.py::TestCompositeBackendEvictionArtifactsRoot::test_summarization_history_prefix` | Duplicates the private custom-root prefix readback. | Delete |
| Open | `tests/unit_tests/test_local_shell.py::TestDefaultTimeoutConstant::test_default_timeout_uses_constant` | Reads private timeout storage; applying the same default directly during execution would preserve behavior and fail the test. | Rewrite |
| Open | `tests/unit_tests/test_local_shell.py::TestInitTimeoutValidation::test_custom_timeout_accepted` | Reads private timeout storage instead of demonstrating that the custom timeout governs a command. | Rewrite |
| Open | `tests/unit_tests/test_middleware.py::TestFilesystemMiddleware::test_init_default` | Mirrors constructor structure through backend type, private prompt storage, and an exact tool count. | Rewrite |
| Open | `tests/unit_tests/test_middleware.py::TestFilesystemMiddleware::test_init_with_composite_backend` | Mirrors constructor structure through backend type, private prompt storage, and an exact tool count. | Rewrite |
| Open | `tests/unit_tests/test_middleware.py::TestFilesystemMiddleware::test_init_custom_system_prompt_default` | Reads the private prompt field instead of inspecting the prompt sent to a model request. | Rewrite |
| Open | `tests/unit_tests/test_middleware.py::TestFilesystemMiddleware::test_init_custom_system_prompt_with_composite` | Reads the private prompt field instead of inspecting the prompt sent to a model request. | Rewrite |
| Open | `tests/unit_tests/test_models.py::TestBuiltInProfiles::test_nvidia_provider_profile_has_attribution_factory` | Requires identity with a specific private factory function even though another factory could produce identical attribution kwargs. | Rewrite |
| Open | `tests/unit_tests/test_nemotron_ultra_profile.py::test_register_adds_ultra3_profiles_for_supported_providers` | The exact ordered list of 12 internal middleware class names fails if equivalent middleware are reordered, combined, or split; retain the profile behavior assertions. | Rewrite |
| Open | `tests/unit_tests/test_rubric_example.py::test_project_is_resolved_after_dotenv_load` | The embedded fake requires an exact positional `load_dotenv("settings")` call; keyword use or an equivalent loading sequence preserves the resolved project and fails the test. | Rewrite |
| Open | `tests/unit_tests/test_version.py::TestLcVersion::test_caches_editable_install_lookup` | Uses a private helper's exact call count as a proxy for caching, coupling the test to where memoization occurs rather than version stability or measured lookup cost. | Rewrite |

## Maintenance

When an entry is resolved, remove its row in the same change. Re-run the complete
`tests/` audit when test architecture changes substantially; otherwise add newly
introduced change detectors as they are found. The counts above describe the
baseline commit and should be updated only by a complete re-audit.
