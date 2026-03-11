"""Tests for workflow node functions in research/graph/nodes.py."""
import pytest
from unittest.mock import Mock, patch, MagicMock

from research.models import ResearchJob


# ---------------------------------------------------------------------------
# validate_input
# ---------------------------------------------------------------------------

class TestValidateInput:

    def test_validate_passes_with_client_name_and_api_key(self, settings):
        from research.graph.nodes import validate_input

        settings.GEMINI_API_KEY = 'test-key-123'
        state = {'client_name': 'TestCo', 'status': 'pending'}

        result = validate_input(state)

        assert result['status'] == 'researching'

    def test_validate_fails_without_client_name(self, settings):
        from research.graph.nodes import validate_input

        settings.GEMINI_API_KEY = 'test-key-123'
        state = {'client_name': '', 'status': 'pending'}

        result = validate_input(state)

        assert result['status'] == 'failed'
        assert 'client name' in result['error'].lower()

    def test_validate_fails_without_api_key(self, settings):
        from research.graph.nodes import validate_input

        settings.GEMINI_API_KEY = ''
        state = {'client_name': 'TestCo', 'status': 'pending'}

        result = validate_input(state)

        assert result['status'] == 'failed'
        assert 'api key' in result['error'].lower()

    def test_validate_preserves_other_state_keys(self, settings):
        from research.graph.nodes import validate_input

        settings.GEMINI_API_KEY = 'test-key'
        state = {
            'client_name': 'TestCo',
            'sales_history': 'Some history',
            'job_id': 'abc-123',
            'status': 'pending',
        }

        result = validate_input(state)

        assert result['sales_history'] == 'Some history'
        assert result['job_id'] == 'abc-123'


# ---------------------------------------------------------------------------
# _format_research_result
# ---------------------------------------------------------------------------

class TestFormatResearchResult:

    def test_format_includes_company_overview(self):
        from research.graph.nodes import _format_research_result

        report = {'company_overview': 'A global tech company.'}
        result = _format_research_result(report)

        assert 'Company Overview' in result
        assert 'A global tech company.' in result

    def test_format_includes_all_key_sections(self):
        from research.graph.nodes import _format_research_result

        report = {
            'company_overview': 'Overview text',
            'pain_points': ['Pain 1', 'Pain 2'],
            'opportunities': ['Opportunity 1'],
            'strategic_goals': ['Goal 1'],
            'key_initiatives': ['Initiative 1'],
            'cloud_footprint': 'AWS heavy',
            'security_posture': 'SOC2 compliant',
            'financial_signals': ['Signal 1'],
            'tech_partnerships': ['Microsoft', 'AWS'],
        }

        result = _format_research_result(report)

        assert 'Pain Points' in result
        assert 'Pain 1' in result
        assert 'Opportunities' in result
        assert 'Strategic Goals' in result
        assert 'Cloud & Infrastructure' in result
        assert 'Security Posture' in result
        assert 'Financial Signals' in result

    def test_format_handles_missing_fields_gracefully(self):
        from research.graph.nodes import _format_research_result

        result = _format_research_result({})

        assert isinstance(result, str)

    def test_format_handles_empty_report(self):
        from research.graph.nodes import _format_research_result

        result = _format_research_result({})

        assert result == ""

    def test_format_handles_decision_makers_dict(self):
        from research.graph.nodes import _format_research_result

        report = {
            'decision_makers': [
                {'name': 'Jane Doe', 'title': 'CTO', 'background': 'Ex-Google'},
            ]
        }

        result = _format_research_result(report)

        assert 'Jane Doe' in result
        assert 'CTO' in result


# ---------------------------------------------------------------------------
# conduct_research node
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestConductResearchNode:

    @patch('research.services.gemini.GeminiClient')
    def test_conduct_research_returns_correct_state_keys(self, MockGeminiClient):
        from research.graph.nodes import conduct_research

        mock_client = MockGeminiClient.return_value
        mock_report = Mock()
        mock_report.to_dict.return_value = {
            'company_overview': 'A test company',
            'pain_points': [],
            'opportunities': [],
        }
        mock_metadata = Mock()
        mock_metadata.to_dict.return_value = {'web_sources': [{'uri': 'https://example.com', 'title': 'Example'}]}
        mock_client.conduct_deep_research.return_value = (mock_report, mock_metadata)

        state = {
            'client_name': 'TestCo',
            'sales_history': '',
            'status': 'researching',
        }

        result = conduct_research(state)

        assert result['status'] == 'classifying'
        assert 'research_report' in result
        assert 'result' in result
        assert 'web_sources' in result
        assert result['research_report']['company_overview'] == 'A test company'

    @patch('research.services.gemini.GeminiClient')
    def test_conduct_research_handles_exception(self, MockGeminiClient):
        from research.graph.nodes import conduct_research

        MockGeminiClient.return_value.conduct_deep_research.side_effect = Exception("API Error")

        state = {
            'client_name': 'TestCo',
            'sales_history': '',
            'status': 'researching',
        }

        result = conduct_research(state)

        assert result['status'] == 'failed'
        assert 'error' in result

    def test_conduct_research_skips_when_already_failed(self):
        from research.graph.nodes import conduct_research

        state = {'status': 'failed', 'error': 'Validation failed', 'client_name': 'TestCo'}
        result = conduct_research(state)

        assert result['status'] == 'failed'


# ---------------------------------------------------------------------------
# finalize_result node
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestFinalizeResultNode:

    def test_finalize_skips_when_no_job_id(self):
        from research.graph.nodes import finalize_result

        state = {
            'client_name': 'TestCo',
            'status': 'completed',
            'research_report': {'company_overview': 'Test'},
            'competitor_case_studies': [],
            'gap_analysis': None,
            'internal_ops': None,
            'gap_correlations': [],
        }

        result = finalize_result(state)

        assert result['status'] == 'completed'

    def test_finalize_handles_missing_research_report(self):
        from research.graph.nodes import finalize_result

        job = ResearchJob.objects.create(client_name='TestCo', status='running')

        state = {
            'job_id': str(job.id),
            'client_name': 'TestCo',
            'status': 'completed',
            'research_report': {},
            'competitor_case_studies': [],
            'gap_analysis': None,
            'internal_ops': None,
            'gap_correlations': [],
        }

        result = finalize_result(state)

        assert result['status'] == 'completed'

    def test_finalize_skips_when_status_failed(self):
        from research.graph.nodes import finalize_result

        state = {
            'job_id': 'some-id',
            'status': 'failed',
            'error': 'Research failed',
            'client_name': 'TestCo',
        }

        result = finalize_result(state)

        assert result['status'] == 'failed'

    def test_finalize_saves_gap_analysis(self):
        from research.graph.nodes import finalize_result
        from research.models import GapAnalysis

        job = ResearchJob.objects.create(client_name='TestCo', status='running')

        state = {
            'job_id': str(job.id),
            'client_name': 'TestCo',
            'vertical': 'technology',
            'status': 'completed',
            'research_report': {'company_overview': 'Test', 'founded_year': 2010},
            'competitor_case_studies': [],
            'gap_analysis': {
                'technology_gaps': ['T1'],
                'capability_gaps': [],
                'process_gaps': [],
                'recommendations': ['R1'],
                'priority_areas': ['P1'],
                'confidence_score': 0.75,
                'analysis_notes': 'Notes',
            },
            'internal_ops': None,
            'gap_correlations': [],
        }

        finalize_result(state)

        assert GapAnalysis.objects.filter(research_job=job).exists()
        gap = GapAnalysis.objects.get(research_job=job)
        assert gap.technology_gaps == ['T1']
        assert gap.confidence_score == 0.75
