"""Tests for VerticalClassifier (AGE-11)."""
import pytest
from unittest.mock import Mock, patch

from research.services.classifier import VerticalClassifier
from research.constants import Vertical


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_gemini_client():
    client = Mock()
    client.classify_vertical.return_value = Vertical.RETAIL.value
    return client


# ---------------------------------------------------------------------------
# Unit Tests: VerticalClassifier
# ---------------------------------------------------------------------------

class TestVerticalClassifier:

    def test_classify_with_llm_returns_valid_vertical(self, mock_gemini_client):
        classifier = VerticalClassifier(gemini_client=mock_gemini_client)
        result = classifier.classify(
            client_name="RetailCo",
            company_overview="A leading retail chain",
            use_llm=True,
        )

        assert result == Vertical.RETAIL.value
        mock_gemini_client.classify_vertical.assert_called_once()

    def test_classify_falls_back_to_keyword_matching_when_llm_fails(self, mock_gemini_client):
        mock_gemini_client.classify_vertical.side_effect = Exception("LLM error")

        classifier = VerticalClassifier(gemini_client=mock_gemini_client)
        result = classifier.classify(
            client_name="RetailCo",
            company_overview="A retail store selling consumer goods",
            use_llm=True,
        )

        assert result == Vertical.RETAIL.value

    def test_classify_by_keywords_healthcare(self):
        classifier = VerticalClassifier()
        result = classifier.classify(
            client_name="HealthFirst Hospital",
            company_overview="A hospital providing medical services",
            use_llm=False,
        )

        assert result == Vertical.HEALTHCARE.value

    def test_classify_by_keywords_finance(self):
        classifier = VerticalClassifier()
        result = classifier.classify(
            client_name="Capital Bank",
            company_overview="Investment banking and financial services",
            use_llm=False,
        )

        assert result == Vertical.FINANCE.value

    def test_classify_by_keywords_technology(self):
        classifier = VerticalClassifier()
        result = classifier.classify(
            client_name="CloudSoft",
            company_overview="SaaS platform for cloud analytics and data processing",
            use_llm=False,
        )

        assert result == Vertical.TECHNOLOGY.value

    def test_classify_defaults_to_other_on_unknown_input(self):
        classifier = VerticalClassifier()
        result = classifier.classify(
            client_name="XYZ Corp",
            company_overview="",
            use_llm=False,
        )

        assert result == Vertical.OTHER.value

    def test_classify_without_gemini_client_uses_keywords(self):
        classifier = VerticalClassifier(gemini_client=None)
        result = classifier.classify(
            client_name="Retail Giant",
            company_overview="Large retail marketplace",
            use_llm=True,  # Should fall back to keywords when no client
        )

        assert result == Vertical.RETAIL.value

    def test_keyword_matching_scores_best_match(self):
        """Multiple vertical keywords present — highest score should win."""
        classifier = VerticalClassifier()
        # Contains many healthcare keywords, only one retail keyword
        result = classifier._classify_by_keywords(
            client_name="General Hospital",
            company_overview="hospital medical healthcare patient clinic diagnostics therapeutic"
        )

        assert result == Vertical.HEALTHCARE.value


# ---------------------------------------------------------------------------
# Workflow Node Tests
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestClassifyVerticalNode:

    @patch('research.services.classifier.VerticalClassifier.classify')
    def test_classify_vertical_node_success(self, mock_classify):
        from research.graph.nodes import classify_vertical

        mock_classify.return_value = Vertical.RETAIL.value

        state = {
            'client_name': 'RetailCo',
            'research_report': {'company_overview': 'A retail chain'},
            'status': 'classifying',
        }

        result = classify_vertical(state)

        assert result['status'] == 'competitor_search'
        assert result['vertical'] == Vertical.RETAIL.value

    @patch('research.services.classifier.VerticalClassifier.classify')
    def test_classify_vertical_node_non_fatal_on_error(self, mock_classify):
        from research.graph.nodes import classify_vertical

        mock_classify.side_effect = Exception("LLM error")

        state = {
            'client_name': 'UnknownCo',
            'research_report': {},
            'status': 'classifying',
        }

        result = classify_vertical(state)

        assert result['status'] == 'competitor_search'
        assert result['vertical'] == 'other'

    def test_classify_vertical_node_skips_when_failed(self):
        from research.graph.nodes import classify_vertical

        state = {'status': 'failed', 'error': 'Prior failure', 'client_name': 'TestCo'}
        result = classify_vertical(state)

        assert result['status'] == 'failed'
