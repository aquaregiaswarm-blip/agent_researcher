"""Tests for GapAnalysisService (AGE-13)."""
import json
import pytest
from unittest.mock import Mock, patch

from research.services.gap_analysis import GapAnalysisService, GapAnalysisData


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_gemini_client():
    return Mock()


@pytest.fixture
def sample_gap_response():
    return json.dumps({
        "technology_gaps": [
            "Gap 1: Missing real-time data pipeline",
            "Gap 2: No ML model serving infrastructure",
        ],
        "capability_gaps": [
            "Gap 1: Limited data science expertise",
        ],
        "process_gaps": [
            "Gap 1: Manual reporting workflows",
            "Gap 2: No automated testing",
        ],
        "recommendations": [
            "Recommendation 1: Implement a streaming data platform",
            "Recommendation 2: Hire ML engineers",
        ],
        "priority_areas": [
            "Priority 1: Data infrastructure modernisation",
            "Priority 2: Talent acquisition in AI/ML",
        ],
        "confidence_score": 0.82,
        "analysis_notes": "Analysis based on public job postings and news coverage."
    })


# ---------------------------------------------------------------------------
# Dataclass Tests
# ---------------------------------------------------------------------------

class TestGapAnalysisData:

    def test_dataclass_creation_and_defaults(self):
        data = GapAnalysisData()
        assert data.technology_gaps == []
        assert data.capability_gaps == []
        assert data.process_gaps == []
        assert data.recommendations == []
        assert data.priority_areas == []
        assert data.confidence_score == 0.0
        assert data.analysis_notes == ""

    def test_dataclass_with_values(self):
        data = GapAnalysisData(
            technology_gaps=["T1", "T2"],
            confidence_score=0.75,
            analysis_notes="Test notes",
        )
        assert len(data.technology_gaps) == 2
        assert data.confidence_score == 0.75


# ---------------------------------------------------------------------------
# Unit Tests: GapAnalysisService
# ---------------------------------------------------------------------------

class TestGapAnalysisService:

    def test_analyze_gaps_returns_structured_data(self, mock_gemini_client, sample_gap_response):
        mock_gemini_client.generate_text.return_value = sample_gap_response

        service = GapAnalysisService(mock_gemini_client)
        result = service.analyze_gaps(client_name="TestCo", vertical="technology")

        assert isinstance(result, GapAnalysisData)
        assert len(result.technology_gaps) == 2
        assert len(result.capability_gaps) == 1
        assert len(result.process_gaps) == 2
        assert len(result.recommendations) == 2
        assert len(result.priority_areas) == 2
        assert result.confidence_score == 0.82
        assert result.analysis_notes == "Analysis based on public job postings and news coverage."

    def test_analyze_gaps_strips_markdown_code_blocks(self, mock_gemini_client, sample_gap_response):
        wrapped = f"```json\n{sample_gap_response}\n```"
        mock_gemini_client.generate_text.return_value = wrapped

        service = GapAnalysisService(mock_gemini_client)
        result = service.analyze_gaps(client_name="TestCo", vertical="technology")

        assert len(result.technology_gaps) == 2
        assert result.confidence_score == 0.82

    def test_analyze_gaps_strips_code_block_with_trailing_newline(self, mock_gemini_client, sample_gap_response):
        """Regression: trailing newline after closing ``` caused parse failure."""
        wrapped = f"```json\n{sample_gap_response}\n```\n"
        mock_gemini_client.generate_text.return_value = wrapped

        service = GapAnalysisService(mock_gemini_client)
        result = service.analyze_gaps(client_name="TestCo", vertical="technology")

        assert len(result.technology_gaps) == 2

    def test_analyze_gaps_handles_json_parse_error(self, mock_gemini_client):
        mock_gemini_client.generate_text.return_value = "not valid json"

        service = GapAnalysisService(mock_gemini_client)
        result = service.analyze_gaps(client_name="TestCo", vertical="technology")

        assert isinstance(result, GapAnalysisData)
        assert "parsing failed" in result.analysis_notes.lower()
        assert result.technology_gaps == []

    def test_analyze_gaps_handles_api_exception(self, mock_gemini_client):
        mock_gemini_client.generate_text.side_effect = Exception("API timeout")

        service = GapAnalysisService(mock_gemini_client)
        result = service.analyze_gaps(client_name="TestCo", vertical="technology")

        assert isinstance(result, GapAnalysisData)
        assert "failed" in result.analysis_notes.lower()

    def test_analyze_gaps_with_competitor_context_in_prompt(self, mock_gemini_client, sample_gap_response):
        """Competitor case studies should be included in the prompt."""
        mock_gemini_client.generate_text.return_value = sample_gap_response

        service = GapAnalysisService(mock_gemini_client)
        service.analyze_gaps(
            client_name="TestCo",
            vertical="technology",
            competitor_case_studies=[{
                "competitor_name": "RivalCo",
                "case_study_title": "AI Transformation",
                "summary": "Deployed ML to cut costs.",
            }]
        )

        call_args = mock_gemini_client.generate_text.call_args[0][0]
        assert "RivalCo" in call_args

    def test_analyze_gaps_empty_pain_points_still_works(self, mock_gemini_client, sample_gap_response):
        mock_gemini_client.generate_text.return_value = sample_gap_response

        service = GapAnalysisService(mock_gemini_client)
        result = service.analyze_gaps(
            client_name="TestCo",
            vertical="technology",
            pain_points=[],
            opportunities=[],
        )

        assert isinstance(result, GapAnalysisData)


# ---------------------------------------------------------------------------
# Workflow Node Tests
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestAnalyzeGapsNode:

    @patch('research.services.gap_analysis.GapAnalysisService.analyze_gaps')
    def test_analyze_gaps_node_success(self, mock_analyze):
        from research.graph.nodes import analyze_gaps

        mock_analyze.return_value = GapAnalysisData(
            technology_gaps=["T1"],
            capability_gaps=["C1"],
            process_gaps=[],
            recommendations=["R1"],
            priority_areas=["P1"],
            confidence_score=0.8,
            analysis_notes="Test notes",
        )

        state = {
            'client_name': 'TestCo',
            'vertical': 'technology',
            'research_report': {
                'company_overview': 'A tech co',
                'pain_points': ['P1'],
                'opportunities': ['O1'],
                'strategic_goals': [],
                'key_initiatives': [],
                'digital_maturity': 'maturing',
                'ai_footprint': 'Some AI',
                'ai_adoption_stage': 'implementing',
            },
            'competitor_case_studies': [],
            'status': 'gap_analysis',
        }

        result = analyze_gaps(state)

        assert result['status'] == 'completed'
        assert 'gap_analysis' in result
        assert result['gap_analysis']['technology_gaps'] == ['T1']
        assert result['gap_analysis']['confidence_score'] == 0.8

    @patch('research.services.gap_analysis.GapAnalysisService.analyze_gaps')
    def test_analyze_gaps_node_non_fatal_on_failure(self, mock_analyze):
        from research.graph.nodes import analyze_gaps

        mock_analyze.side_effect = Exception("Service down")

        state = {
            'client_name': 'TestCo',
            'vertical': 'technology',
            'research_report': {},
            'competitor_case_studies': [],
            'status': 'gap_analysis',
        }

        result = analyze_gaps(state)

        assert result['status'] == 'completed'
        assert result['gap_analysis'] is None

    def test_analyze_gaps_node_skips_when_status_failed(self):
        from research.graph.nodes import analyze_gaps

        state = {'status': 'failed', 'error': 'Prior failure', 'client_name': 'TestCo'}
        result = analyze_gaps(state)

        assert result['status'] == 'failed'
