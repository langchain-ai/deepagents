"""Unit tests for `deepagents_evals.trial_summary`."""

from __future__ import annotations

from deepagents_evals.trial_summary import render_per_trial_category_matrix


class TestRenderPerTrialCategoryMatrix:
    """The matrix renders one row per trial and one column per category."""

    def test_returns_empty_when_no_trials(self) -> None:
        assert render_per_trial_category_matrix([], ["memory"], {}) == []

    def test_returns_empty_when_no_categories(self) -> None:
        trials = [{"trial_index": 0, "category_scores": {"memory": 1.0}}]
        assert render_per_trial_category_matrix(trials, [], {}) == []

    def test_renders_header_separator_and_rows(self) -> None:
        trials = [
            {"trial_index": 0, "category_scores": {"memory": 0.875, "tool_use": 0.5}},
            {"trial_index": 1, "category_scores": {"memory": 1.0, "tool_use": 0.25}},
        ]
        cat_keys = ["memory", "tool_use"]
        labels = {"memory": "Memory", "tool_use": "Tool use"}

        lines = render_per_trial_category_matrix(trials, cat_keys, labels)

        assert lines == [
            "",
            "### Per-trial correctness by category",
            "",
            "| # | Memory | Tool use |",
            "|---:|---:|---:|",
            "| 0 | 0.875 | 0.500 |",
            "| 1 | 1.000 | 0.250 |",
        ]

    def test_falls_back_to_raw_key_when_label_missing(self) -> None:
        trials = [{"trial_index": 0, "category_scores": {"memory": 1.0}}]
        lines = render_per_trial_category_matrix(trials, ["memory"], labels=None)
        # Header uses the raw key when no label dict is provided.
        assert "| memory |" in lines[3]

    def test_renders_dash_for_missing_score(self) -> None:
        # `None` distinguishes "category did not run for this trial" from
        # an actual 0.0 score — pytest_reporter only emits a category when
        # at least one test ran.
        trials = [
            {"trial_index": 0, "category_scores": {"memory": 0.0}},
            {"trial_index": 1, "category_scores": {}},
        ]
        lines = render_per_trial_category_matrix(trials, ["memory"], {})
        assert lines[-2] == "| 0 | 0.000 |"
        assert lines[-1] == "| 1 | - |"

    def test_renders_dash_when_category_scores_is_none(self) -> None:
        trials = [{"trial_index": 0, "category_scores": None}]
        lines = render_per_trial_category_matrix(trials, ["memory"], {})
        assert lines[-1] == "| 0 | - |"

    def test_escapes_pipe_in_label(self) -> None:
        trials = [{"trial_index": 0, "category_scores": {"weird": 1.0}}]
        labels = {"weird": "Has | pipe"}
        lines = render_per_trial_category_matrix(trials, ["weird"], labels)
        assert "Has \\| pipe" in lines[3]

    def test_escapes_newline_in_label(self) -> None:
        trials = [{"trial_index": 0, "category_scores": {"weird": 1.0}}]
        labels = {"weird": "Line one\nline two"}
        lines = render_per_trial_category_matrix(trials, ["weird"], labels)
        # Newline collapsed to a single space; the row stays one line.
        assert "Line one line two" in lines[3]
        assert "\n" not in lines[3]

    def test_escapes_backslash_in_label(self) -> None:
        trials = [{"trial_index": 0, "category_scores": {"weird": 1.0}}]
        labels = {"weird": "back\\slash"}
        lines = render_per_trial_category_matrix(trials, ["weird"], labels)
        # Backslash is doubled before the pipe escape pass so a malicious
        # label can't smuggle a real `\|` through.
        assert "back\\\\slash" in lines[3]

    def test_places_parameter_controls_precision(self) -> None:
        trials = [{"trial_index": 0, "category_scores": {"memory": 0.123456}}]
        lines = render_per_trial_category_matrix(trials, ["memory"], {}, places=2)
        assert lines[-1] == "| 0 | 0.12 |"

    def test_columns_render_in_provided_order(self) -> None:
        # The caller controls column order via `cat_keys`; the helper must
        # not re-sort or it would break alignment with the header.
        trials = [{"trial_index": 0, "category_scores": {"a": 0.1, "b": 0.2}}]
        forward = render_per_trial_category_matrix(trials, ["a", "b"], {})
        reverse = render_per_trial_category_matrix(trials, ["b", "a"], {})
        assert forward[-1] == "| 0 | 0.100 | 0.200 |"
        assert reverse[-1] == "| 0 | 0.200 | 0.100 |"
