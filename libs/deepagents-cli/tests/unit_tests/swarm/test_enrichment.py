"""Unit tests for CSV enrichment module."""

import pytest

from deepagents_cli.swarm.enrichment import (
    EnrichmentError,
    create_enrichment_tasks,
    merge_enrichment_results,
    parse_csv_for_enrichment,
    parse_enrichment_output,
    write_enriched_csv,
)


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV with some empty columns."""
    path = tmp_path / "companies.csv"
    content = """\
ticker,company,ceo,cto,market_cap
AAPL,Apple,,,
MSFT,Microsoft,Satya Nadella,,
GOOGL,Alphabet,,,
"""
    path.write_text(content)
    return path


class TestParseCsvForEnrichment:
    def test_parses_headers_and_rows(self, sample_csv):
        headers, rows = parse_csv_for_enrichment(sample_csv)

        assert headers == ["ticker", "company", "ceo", "cto", "market_cap"]
        assert len(rows) == 3
        assert rows[0]["ticker"] == "AAPL"
        assert rows[1]["ceo"] == "Satya Nadella"

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_csv_for_enrichment(tmp_path / "nonexistent.csv")

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("ticker,company\n")  # Headers only

        with pytest.raises(EnrichmentError, match="no data rows"):
            parse_csv_for_enrichment(path)


class TestCreateEnrichmentTasks:
    def test_creates_tasks_for_rows_with_empty_columns(self, sample_csv):
        headers, rows = parse_csv_for_enrichment(sample_csv)
        tasks = create_enrichment_tasks(headers, rows)

        # All 3 rows have empty columns
        assert len(tasks) == 3

    def test_task_ids_default_to_row_index(self, sample_csv):
        headers, rows = parse_csv_for_enrichment(sample_csv)
        tasks = create_enrichment_tasks(headers, rows)

        task_ids = [t["id"] for t in tasks]
        assert task_ids == ["1", "2", "3"]

    def test_task_ids_from_column(self, sample_csv):
        headers, rows = parse_csv_for_enrichment(sample_csv)
        tasks = create_enrichment_tasks(headers, rows, id_column="ticker")

        task_ids = [t["id"] for t in tasks]
        assert task_ids == ["AAPL", "MSFT", "GOOGL"]

    def test_task_description_includes_context(self, sample_csv):
        headers, rows = parse_csv_for_enrichment(sample_csv)
        tasks = create_enrichment_tasks(headers, rows)

        # First task (AAPL) should have ticker and company as context
        desc = tasks[0]["description"]
        assert "AAPL" in desc
        assert "Apple" in desc

    def test_task_description_lists_empty_columns(self, sample_csv):
        headers, rows = parse_csv_for_enrichment(sample_csv)
        tasks = create_enrichment_tasks(headers, rows)

        # First task needs ceo, cto, market_cap
        desc = tasks[0]["description"]
        assert "ceo" in desc
        assert "cto" in desc
        assert "market_cap" in desc

    def test_skips_rows_with_no_empty_columns(self, tmp_path):
        path = tmp_path / "complete.csv"
        path.write_text("a,b\n1,2\n3,4\n")

        headers, rows = parse_csv_for_enrichment(path)
        tasks = create_enrichment_tasks(headers, rows)

        assert len(tasks) == 0

    def test_metadata_includes_row_info(self, sample_csv):
        headers, rows = parse_csv_for_enrichment(sample_csv)
        tasks = create_enrichment_tasks(headers, rows)

        meta = tasks[0]["metadata"]
        assert meta["row_index"] == 0
        assert "ceo" in meta["empty_columns"]
        assert meta["original_row"]["ticker"] == "AAPL"


class TestParseEnrichmentOutput:
    def test_parses_json_object(self):
        output = '{"ceo": "Tim Cook", "cto": "Craig Federighi"}'
        result = parse_enrichment_output(output)

        assert result["ceo"] == "Tim Cook"
        assert result["cto"] == "Craig Federighi"

    def test_parses_json_in_code_block(self):
        output = """Here's the information:
```json
{"ceo": "Tim Cook", "market_cap": "$3T"}
```
"""
        result = parse_enrichment_output(output)

        assert result["ceo"] == "Tim Cook"
        assert result["market_cap"] == "$3T"

    def test_parses_json_with_surrounding_text(self):
        output = 'The CEO is {"ceo": "Tim Cook"} based on my research.'
        result = parse_enrichment_output(output)

        assert result["ceo"] == "Tim Cook"

    def test_handles_null_values(self):
        output = '{"ceo": "Tim Cook", "cto": null}'
        result = parse_enrichment_output(output)

        assert result["ceo"] == "Tim Cook"
        assert result["cto"] == ""

    def test_returns_empty_dict_for_invalid_json(self):
        output = "I couldn't find the information"
        result = parse_enrichment_output(output)

        assert result == {}

    def test_returns_empty_dict_for_empty_output(self):
        assert parse_enrichment_output("") == {}
        assert parse_enrichment_output(None) == {}


class TestMergeEnrichmentResults:
    def test_merges_results_into_empty_columns(self):
        headers = ["ticker", "company", "ceo"]
        rows = [
            {"ticker": "AAPL", "company": "Apple", "ceo": ""},
            {"ticker": "MSFT", "company": "Microsoft", "ceo": ""},
        ]
        results = {
            "1": {"ceo": "Tim Cook"},
            "2": {"ceo": "Satya Nadella"},
        }

        enriched = merge_enrichment_results(headers, rows, results)

        assert enriched[0]["ceo"] == "Tim Cook"
        assert enriched[1]["ceo"] == "Satya Nadella"

    def test_preserves_existing_values(self):
        headers = ["ticker", "ceo"]
        rows = [{"ticker": "AAPL", "ceo": "Tim Cook"}]
        results = {"1": {"ceo": "Wrong Person"}}

        enriched = merge_enrichment_results(headers, rows, results)

        # Should keep original value
        assert enriched[0]["ceo"] == "Tim Cook"

    def test_handles_missing_results(self):
        headers = ["ticker", "ceo"]
        rows = [{"ticker": "AAPL", "ceo": ""}]
        results = {}  # No results

        enriched = merge_enrichment_results(headers, rows, results)

        assert enriched[0]["ceo"] == ""


class TestWriteEnrichedCsv:
    def test_writes_csv_file(self, tmp_path):
        path = tmp_path / "output.csv"
        headers = ["a", "b"]
        rows = [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]

        write_enriched_csv(path, headers, rows)

        content = path.read_text()
        assert "a,b" in content
        assert "1,2" in content
        assert "3,4" in content

    def test_returns_path(self, tmp_path):
        path = tmp_path / "output.csv"
        result = write_enriched_csv(path, ["a"], [{"a": "1"}])

        assert result == path
