"""Tests for inline citation functionality in the Gemini client."""
import pytest
from research.services.gemini import (
    GeminiClient,
    WebSource,
    GroundingMetadata,
    _strip_invalid_citations,
)


class TestStripInvalidCitations:
    """Tests for the _strip_invalid_citations module-level helper."""

    def test_removes_out_of_range_citation(self):
        """[5] with max_n=3 should be stripped; [2] should be preserved."""
        data = {"overview": "Fact one [2] and fact two [5]."}
        result = _strip_invalid_citations(data, max_n=3)
        assert "[2]" in result["overview"]
        assert "[5]" not in result["overview"]

    def test_preserves_in_range_citations(self):
        """All citations within range should be preserved."""
        data = {"overview": "First [1], second [2], third [3]."}
        result = _strip_invalid_citations(data, max_n=3)
        assert result["overview"] == "First [1], second [2], third [3]."

    def test_noop_when_no_citations(self):
        """Plain text without any [N] markers should be unchanged."""
        data = {"overview": "A simple company with no citations."}
        result = _strip_invalid_citations(data, max_n=5)
        assert result["overview"] == "A simple company with no citations."

    def test_handles_list_values(self):
        """Citations in list items should also be cleaned."""
        data = {
            "pain_points": [
                "Legacy systems [1]",
                "Data silos [10]",
                "High costs [3]",
            ]
        }
        result = _strip_invalid_citations(data, max_n=3)
        assert "[1]" in result["pain_points"][0]
        assert "[10]" not in result["pain_points"][1]
        assert "[3]" in result["pain_points"][2]

    def test_strips_only_out_of_range_boundary(self):
        """[N] exactly at max_n should be preserved; [N+1] should be stripped."""
        data = {"overview": "Exact [5] and over [6]."}
        result = _strip_invalid_citations(data, max_n=5)
        assert "[5]" in result["overview"]
        assert "[6]" not in result["overview"]

    def test_non_string_values_passed_through(self):
        """Non-string, non-list values should not be modified."""
        data = {
            "founded_year": 2010,
            "confidence": 0.9,
            "overview": "Text [1]",
        }
        result = _strip_invalid_citations(data, max_n=1)
        assert result["founded_year"] == 2010
        assert result["confidence"] == 0.9
        assert "[1]" in result["overview"]

    def test_multiple_out_of_range_in_one_field(self):
        """Multiple out-of-range markers in one field are all stripped."""
        data = {"overview": "Fact [99] and fact [100] and fact [1]."}
        result = _strip_invalid_citations(data, max_n=3)
        assert "[99]" not in result["overview"]
        assert "[100]" not in result["overview"]
        assert "[1]" in result["overview"]


class TestSynthesisPromptSourceList:
    """Tests that the synthesis prompt includes source list when sources are available."""

    def test_synthesis_prompt_includes_source_list_text(self):
        """When sources exist, formatted prompt should contain numbered source list."""
        import unittest.mock as mock

        sources = [
            WebSource(uri="https://example.com/1", title="Example Article"),
            WebSource(uri="https://news.com/2", title="News Story"),
        ]
        metadata = GroundingMetadata(web_sources=sources, search_queries=[])

        client = GeminiClient(api_key="test-key")

        # Build what the source_list_text would be
        source_list_text = "\n".join(
            f"{i+1}. {s.title or 'Source'} — {s.uri}"
            for i, s in enumerate(sources)
        )

        prompt = client.SYNTHESIS_PROMPT.format(
            client_name="TestCo",
            prompt="No specific research directive provided.",
            sales_history="No prior sales history provided.",
            profile="Profile text.",
            news="News text.",
            leadership="Leadership text.",
            technology="Tech text.",
            cloud_infrastructure="Cloud text.",
            cybersecurity_compliance="Security text.",
            data_analytics="Data text.",
            financial_filings="Financial text.",
            source_list=source_list_text,
        )

        assert "1. Example Article — https://example.com/1" in prompt
        assert "2. News Story — https://news.com/2" in prompt
        assert "CITATION RULES" in prompt

    def test_synthesis_prompt_works_with_empty_source_list(self):
        """Prompt should render without error when no sources (empty string)."""
        client = GeminiClient(api_key="test-key")

        prompt = client.SYNTHESIS_PROMPT.format(
            client_name="TestCo",
            prompt="No specific research directive provided.",
            sales_history="No prior sales history provided.",
            profile="Profile text.",
            news="News text.",
            leadership="Leadership text.",
            technology="Tech text.",
            cloud_infrastructure="Cloud text.",
            cybersecurity_compliance="Security text.",
            data_analytics="Data text.",
            financial_filings="Financial text.",
            source_list="",
        )

        assert "TestCo" in prompt
        assert "CITATION RULES" in prompt

    def test_json_format_prompt_includes_preservation_instruction(self):
        """JSON format prompt should include the citation preservation instruction."""
        client = GeminiClient(api_key="test-key")

        prompt = client.JSON_FORMAT_PROMPT.format(research_text="Some research.")

        assert "CITATION PRESERVATION" in prompt
        assert "Preserve these EXACTLY" in prompt
