"""Tests for CompetitorSearchService (AGE-12)."""
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from research.services.competitor import CompetitorSearchService, CompetitorCaseStudyData


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_gemini_client():
    client = Mock()
    client.MODEL_FLASH = 'gemini-2.5-flash'
    raw_client = Mock()
    mock_response = Mock()
    mock_response.candidates = []
    mock_response.text = ""
    raw_client.models.generate_content.return_value = mock_response
    client.client = raw_client
    return client


@pytest.fixture
def sample_case_studies_json():
    return json.dumps({
        "case_studies": [
            {
                "competitor_name": "RetailCo",
                "vertical": "retail",
                "case_study_title": "AI-Driven Inventory Optimisation",
                "summary": "RetailCo deployed ML models to cut overstock by 20%.",
                "technologies_used": ["Python", "TensorFlow", "BigQuery"],
                "outcomes": ["20% reduction in overstock", "15% uplift in margin"],
                "source_url": "https://retailco.example.com/case-study",
                "relevance_score": 0.85
            },
            {
                "competitor_name": "ShopNet",
                "vertical": "retail",
                "case_study_title": "Personalisation Engine",
                "summary": "ShopNet built a real-time recommendation engine.",
                "technologies_used": ["Spark", "Databricks"],
                "outcomes": ["12% conversion uplift"],
                "source_url": "https://shopnet.example.com/ai",
                "relevance_score": 0.72
            }
        ]
    })


# ---------------------------------------------------------------------------
# Dataclass Tests
# ---------------------------------------------------------------------------

class TestCompetitorCaseStudyDataclass:

    def test_dataclass_creation_and_defaults(self):
        cs = CompetitorCaseStudyData()
        assert cs.competitor_name == ""
        assert cs.vertical == ""
        assert cs.technologies_used == []
        assert cs.outcomes == []
        assert cs.source_url == ""
        assert cs.relevance_score == 0.0

    def test_dataclass_with_values(self):
        cs = CompetitorCaseStudyData(
            competitor_name="Acme",
            vertical="technology",
            case_study_title="ML Pipeline",
            summary="Built a feature platform.",
            technologies_used=["Python"],
            outcomes=["30% faster inference"],
            source_url="https://example.com",
            relevance_score=0.9,
        )
        assert cs.competitor_name == "Acme"
        assert cs.relevance_score == 0.9
        assert "Python" in cs.technologies_used


# ---------------------------------------------------------------------------
# Unit Tests: CompetitorSearchService
# ---------------------------------------------------------------------------

class TestCompetitorSearchService:

    def _make_grounded_result(self, text, success=True, grounding_metadata=None):
        result = Mock()
        result.success = success
        result.text = text
        result.error = None if success else "error"
        result.grounding_metadata = grounding_metadata
        return result

    @patch('research.services.grounding.conduct_grounded_query')
    def test_search_returns_list_of_case_studies(self, mock_grounded, mock_gemini_client, sample_case_studies_json):
        mock_grounded.return_value = self._make_grounded_result(sample_case_studies_json)

        service = CompetitorSearchService(mock_gemini_client)
        case_studies, metadata = service.search_competitor_case_studies(
            client_name="TestCo",
            vertical="retail",
            company_overview="A retailer",
        )

        assert len(case_studies) == 2
        assert isinstance(case_studies[0], CompetitorCaseStudyData)
        assert case_studies[0].competitor_name == "RetailCo"
        assert case_studies[0].relevance_score == 0.85
        assert "Python" in case_studies[0].technologies_used

    @patch('research.services.grounding.conduct_grounded_query')
    def test_search_handles_json_parse_error_gracefully(self, mock_grounded, mock_gemini_client):
        mock_grounded.return_value = self._make_grounded_result("not valid json at all")

        service = CompetitorSearchService(mock_gemini_client)
        case_studies, metadata = service.search_competitor_case_studies(
            client_name="TestCo",
            vertical="retail",
        )

        assert case_studies == []

    @patch('research.services.grounding.conduct_grounded_query')
    def test_search_strips_markdown_code_blocks(self, mock_grounded, mock_gemini_client, sample_case_studies_json):
        wrapped = f"```json\n{sample_case_studies_json}\n```"
        mock_grounded.return_value = self._make_grounded_result(wrapped)

        service = CompetitorSearchService(mock_gemini_client)
        case_studies, metadata = service.search_competitor_case_studies(
            client_name="TestCo",
            vertical="retail",
        )

        assert len(case_studies) == 2

    @patch('research.services.grounding.conduct_grounded_query')
    def test_search_handles_empty_response(self, mock_grounded, mock_gemini_client):
        mock_grounded.return_value = self._make_grounded_result(success=False, text="")

        service = CompetitorSearchService(mock_gemini_client)
        case_studies, metadata = service.search_competitor_case_studies(
            client_name="TestCo",
            vertical="retail",
        )

        assert case_studies == []

    @patch('research.services.grounding.conduct_grounded_query')
    def test_search_handles_api_exception_gracefully(self, mock_grounded, mock_gemini_client):
        mock_grounded.side_effect = Exception("Network error")

        service = CompetitorSearchService(mock_gemini_client)
        case_studies, metadata = service.search_competitor_case_studies(
            client_name="TestCo",
            vertical="retail",
        )

        assert case_studies == []

    @patch('research.services.grounding.conduct_grounded_query')
    def test_search_truncates_long_source_urls(self, mock_grounded, mock_gemini_client):
        """Regression: source_url max_length=2000 in model; service stores raw URL."""
        long_url = "https://example.com/" + "x" * 2500
        payload = json.dumps({
            "case_studies": [{
                "competitor_name": "BigCo",
                "vertical": "retail",
                "case_study_title": "Long URL Test",
                "summary": "Testing.",
                "technologies_used": [],
                "outcomes": [],
                "source_url": long_url,
                "relevance_score": 0.5,
            }]
        })
        mock_grounded.return_value = self._make_grounded_result(payload)

        service = CompetitorSearchService(mock_gemini_client)
        case_studies, metadata = service.search_competitor_case_studies(
            client_name="TestCo",
            vertical="retail",
        )

        # Service stores the raw URL; truncation happens in finalize_result node
        assert case_studies[0].source_url == long_url

    @patch('research.services.grounding.conduct_grounded_query')
    def test_search_returns_empty_list_for_empty_case_studies(self, mock_grounded, mock_gemini_client):
        payload = json.dumps({"case_studies": []})
        mock_grounded.return_value = self._make_grounded_result(payload)

        service = CompetitorSearchService(mock_gemini_client)
        case_studies, metadata = service.search_competitor_case_studies(
            client_name="TestCo",
            vertical="retail",
        )

        assert case_studies == []


# ---------------------------------------------------------------------------
# Workflow Node Tests
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestSearchCompetitorsNode:

    @patch('research.services.grounding.conduct_grounded_query')
    def test_search_competitors_node_success(self, mock_grounded):
        from research.graph.nodes import search_competitors

        payload = json.dumps({
            "case_studies": [{
                "competitor_name": "RivalCo",
                "vertical": "technology",
                "case_study_title": "AI Platform",
                "summary": "Built an AI platform.",
                "technologies_used": ["Python"],
                "outcomes": ["50% efficiency gain"],
                "source_url": "https://rival.co/ai",
                "relevance_score": 0.8,
            }]
        })
        grounding = Mock()
        grounding.to_dict.return_value = {"web_sources": [{"uri": "https://rival.co/ai", "title": "RivalCo AI"}]}
        mock_grounded.return_value = Mock(success=True, text=payload, error=None, grounding_metadata=grounding)

        state = {
            'client_name': 'TestCo',
            'vertical': 'technology',
            'research_report': {'company_overview': 'A tech company'},
            'web_sources': [],
            'status': 'competitor_search',
        }

        result = search_competitors(state)

        assert result['status'] == 'gap_analysis'
        assert len(result['competitor_case_studies']) == 1
        assert result['competitor_case_studies'][0]['competitor_name'] == 'RivalCo'

    @patch('research.services.grounding.conduct_grounded_query')
    def test_search_competitors_node_non_fatal_on_failure(self, mock_grounded):
        from research.graph.nodes import search_competitors

        mock_grounded.side_effect = Exception("Service down")

        state = {
            'client_name': 'TestCo',
            'vertical': 'technology',
            'research_report': {},
            'web_sources': [],
            'status': 'competitor_search',
        }

        result = search_competitors(state)

        assert result['status'] == 'gap_analysis'
        assert result['competitor_case_studies'] == []

    @patch('research.services.grounding.conduct_grounded_query')
    def test_search_competitors_node_merges_web_sources(self, mock_grounded):
        from research.graph.nodes import search_competitors

        grounding = Mock()
        grounding.to_dict.return_value = {"web_sources": [
            {"uri": "https://new-source.com", "title": "New Source"}
        ]}
        mock_grounded.return_value = Mock(
            success=True,
            text=json.dumps({"case_studies": []}),
            error=None,
            grounding_metadata=grounding,
        )

        existing_sources = [{"uri": "https://existing.com", "title": "Existing"}]
        state = {
            'client_name': 'TestCo',
            'vertical': 'technology',
            'research_report': {},
            'web_sources': existing_sources,
            'status': 'competitor_search',
        }

        result = search_competitors(state)

        uris = [s['uri'] for s in result['web_sources']]
        assert 'https://existing.com' in uris
        assert 'https://new-source.com' in uris

    def test_search_competitors_node_skips_when_failed(self):
        from research.graph.nodes import search_competitors

        state = {'status': 'failed', 'error': 'Prior failure', 'client_name': 'TestCo'}
        result = search_competitors(state)

        assert result['status'] == 'failed'
