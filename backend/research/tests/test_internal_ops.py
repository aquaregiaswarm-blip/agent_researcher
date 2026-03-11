"""Tests for Internal Operations Intelligence feature (AGE-20)."""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from django.urls import reverse
from rest_framework.test import APIClient

from research.models import ResearchJob, InternalOpsIntel
from research.services.internal_ops import (
    InternalOpsService,
    InternalOpsData,
    EmployeeSentiment,
    LinkedInPresence,
    SocialMediaMention,
    JobPostings,
    NewsSentiment,
)
from research.services.gap_correlation import (
    GapCorrelationService,
    GapCorrelation,
)
from research.serializers import InternalOpsIntelSerializer, ResearchJobDetailSerializer


@pytest.fixture
def api_client():
    return APIClient()


@pytest.fixture
def mock_gemini_client():
    """Create a mock Gemini client with required attributes for the new grounded implementation."""
    client = Mock()
    client.MODEL_FLASH = 'gemini-2.5-flash'
    # client.client is the raw genai client used by conduct_grounded_query
    raw_client = Mock()
    mock_response = Mock()
    mock_response.candidates = []
    mock_response.text = ""
    raw_client.models.generate_content.return_value = mock_response
    client.client = raw_client
    return client


@pytest.fixture
def sample_internal_ops_response():
    """Sample JSON response for internal ops."""
    return json.dumps({
        "employee_sentiment": {
            "overall_rating": 3.8,
            "work_life_balance": 3.5,
            "compensation": 4.0,
            "culture": 3.7,
            "management": 3.4,
            "recommend_pct": 72,
            "positive_themes": ["Good benefits", "Collaborative team"],
            "negative_themes": ["Long hours", "Slow promotions"],
            "trend": "stable"
        },
        "linkedin_presence": {
            "follower_count": 50000,
            "engagement_level": "medium",
            "recent_posts": [
                {"title": "New Product Launch", "summary": "Exciting announcement", "date": "2024-01"}
            ],
            "employee_trend": "growing",
            "notable_changes": ["New CTO hired"]
        },
        "social_media_mentions": [
            {
                "platform": "reddit",
                "summary": "Discussion about work culture",
                "sentiment": "mixed",
                "topic": "Work-life balance"
            }
        ],
        "job_postings": {
            "total_openings": 45,
            "departments_hiring": {"Engineering": 20, "Sales": 15, "Marketing": 10},
            "skills_sought": ["Python", "Cloud", "AI/ML"],
            "seniority_distribution": {"Entry": 10, "Mid": 20, "Senior": 15},
            "urgency_signals": ["Sign-on bonus offered"],
            "insights": "Heavy focus on technical hiring"
        },
        "news_sentiment": {
            "overall_sentiment": "positive",
            "coverage_volume": "medium",
            "topics": ["Product launch", "Expansion"],
            "headlines": [
                {"title": "Company Expands to New Markets", "source": "TechCrunch", "date": "2024-01", "sentiment": "positive"}
            ]
        },
        "key_insights": [
            "Strong hiring suggests growth phase",
            "Employee sentiment indicates culture challenges"
        ],
        "confidence_score": 0.75,
        "data_freshness": "last_30_days",
        "analysis_notes": "Analysis based on public sources"
    })


@pytest.fixture
def sample_gap_correlation_response():
    """Sample JSON response for gap correlation."""
    return json.dumps({
        "gap_correlations": [
            {
                "gap_type": "technology",
                "description": "Missing cloud infrastructure",
                "evidence": "Heavy hiring for cloud engineers",
                "evidence_type": "supporting",
                "confidence": 0.85,
                "sales_implication": "Strong opportunity to propose cloud solutions"
            },
            {
                "gap_type": "capability",
                "description": "Lack of AI/ML expertise",
                "evidence": "Multiple AI/ML job postings with urgency signals",
                "evidence_type": "supporting",
                "confidence": 0.78,
                "sales_implication": "Position AI enablement services"
            }
        ],
        "overall_confidence": 0.80,
        "analysis_summary": "Multiple gaps validated by hiring patterns"
    })


# ============================================================================
# Unit Tests: InternalOpsData Dataclasses
# ============================================================================

class TestInternalOpsDataclasses:
    """Tests for Internal Ops data structures."""

    def test_employee_sentiment_defaults(self):
        """Test EmployeeSentiment has correct defaults."""
        sentiment = EmployeeSentiment()
        assert sentiment.overall_rating == 0.0
        assert sentiment.trend == "stable"
        assert sentiment.positive_themes == []
        assert sentiment.negative_themes == []

    def test_employee_sentiment_to_dict(self):
        """Test EmployeeSentiment converts to dict correctly."""
        sentiment = EmployeeSentiment(
            overall_rating=4.2,
            work_life_balance=4.0,
            positive_themes=["Good pay"],
            trend="improving"
        )
        result = sentiment.to_dict()
        assert result['overall_rating'] == 4.2
        assert result['trend'] == "improving"
        assert "Good pay" in result['positive_themes']

    def test_linkedin_presence_defaults(self):
        """Test LinkedInPresence has correct defaults."""
        presence = LinkedInPresence()
        assert presence.follower_count == 0
        assert presence.engagement_level == "medium"
        assert presence.employee_trend == "stable"

    def test_job_postings_to_dict(self):
        """Test JobPostings converts to dict correctly."""
        postings = JobPostings(
            total_openings=50,
            departments_hiring={"Engineering": 30},
            skills_sought=["Python", "AWS"]
        )
        result = postings.to_dict()
        assert result['total_openings'] == 50
        assert result['departments_hiring']['Engineering'] == 30

    def test_internal_ops_data_to_dict(self):
        """Test InternalOpsData full conversion."""
        ops_data = InternalOpsData(
            key_insights=["Insight 1", "Insight 2"],
            confidence_score=0.85,
            data_freshness="last_7_days"
        )
        result = ops_data.to_dict()
        assert len(result['key_insights']) == 2
        assert result['confidence_score'] == 0.85
        assert result['data_freshness'] == "last_7_days"


# ============================================================================
# Unit Tests: InternalOpsService
# ============================================================================

class TestInternalOpsService:
    """Tests for InternalOpsService."""

    def test_research_internal_ops_success(self, mock_gemini_client, sample_internal_ops_response):
        """Test successful internal ops research returns (InternalOpsData, metadata) tuple."""
        mock_gemini_client.generate_text.return_value = sample_internal_ops_response

        service = InternalOpsService(mock_gemini_client)
        result = service.research_internal_ops(
            client_name="Acme Corp",
            vertical="technology",
            website="https://acme.com",
            company_overview="A leading tech company"
        )

        ops_data, metadata = result
        assert isinstance(ops_data, InternalOpsData)
        assert ops_data.employee_sentiment.overall_rating == 3.8
        assert ops_data.linkedin_presence.follower_count == 50000
        assert len(ops_data.social_media_mentions) == 1
        assert ops_data.job_postings.total_openings == 45
        assert ops_data.confidence_score == 0.75

    def test_research_internal_ops_handles_json_error(self, mock_gemini_client):
        """Test service handles JSON parsing errors gracefully."""
        mock_gemini_client.generate_text.return_value = "invalid json response"

        service = InternalOpsService(mock_gemini_client)
        result = service.research_internal_ops(client_name="Test Corp")

        ops_data, metadata = result
        assert isinstance(ops_data, InternalOpsData)
        assert "parsing failed" in ops_data.analysis_notes.lower()

    def test_research_internal_ops_handles_exception(self, mock_gemini_client):
        """Test service handles exceptions gracefully."""
        mock_gemini_client.generate_text.side_effect = Exception("API Error")

        service = InternalOpsService(mock_gemini_client)
        result = service.research_internal_ops(client_name="Test Corp")

        ops_data, metadata = result
        assert isinstance(ops_data, InternalOpsData)
        assert "failed" in ops_data.analysis_notes.lower()

    def test_research_internal_ops_strips_markdown_code_blocks(self, mock_gemini_client, sample_internal_ops_response):
        """Test service handles markdown code blocks in response."""
        wrapped_response = f"```json\n{sample_internal_ops_response}\n```"
        mock_gemini_client.generate_text.return_value = wrapped_response

        service = InternalOpsService(mock_gemini_client)
        ops_data, metadata = service.research_internal_ops(client_name="Acme Corp")

        assert ops_data.employee_sentiment.overall_rating == 3.8


# ============================================================================
# Unit Tests: GapCorrelationService
# ============================================================================

class TestGapCorrelationService:
    """Tests for GapCorrelationService."""

    def test_correlate_gaps_success(self, mock_gemini_client, sample_gap_correlation_response):
        """Test successful gap correlation."""
        mock_gemini_client.generate_text.return_value = sample_gap_correlation_response

        service = GapCorrelationService(mock_gemini_client)

        gap_analysis = {
            "technology_gaps": ["Missing cloud infrastructure"],
            "capability_gaps": ["Lack of AI/ML expertise"],
            "process_gaps": []
        }
        internal_ops = {
            "employee_sentiment": {"overall_rating": 3.5},
            "job_postings": {"total_openings": 45, "skills_sought": ["Cloud", "AI/ML"]}
        }

        result = service.correlate_gaps(
            client_name="Acme Corp",
            vertical="technology",
            gap_analysis=gap_analysis,
            internal_ops=internal_ops
        )

        assert len(result) == 2
        assert isinstance(result[0], GapCorrelation)
        assert result[0].gap_type == "technology"
        assert result[0].evidence_type == "supporting"
        assert result[0].confidence == 0.85

    def test_correlate_gaps_handles_json_error(self, mock_gemini_client):
        """Test gap correlation handles JSON errors."""
        mock_gemini_client.generate_text.return_value = "not valid json"

        service = GapCorrelationService(mock_gemini_client)
        result = service.correlate_gaps(
            client_name="Test",
            vertical="other",
            gap_analysis={},
            internal_ops={}
        )

        assert result == []

    def test_correlations_to_dict(self, mock_gemini_client):
        """Test converting correlations list to dict."""
        service = GapCorrelationService(mock_gemini_client)

        correlations = [
            GapCorrelation(
                gap_type="technology",
                description="Missing cloud",
                evidence="Hiring cloud engineers",
                evidence_type="supporting",
                confidence=0.8,
                sales_implication="Opportunity for cloud services"
            )
        ]

        result = service.correlations_to_dict(correlations)

        assert len(result) == 1
        assert result[0]['gap_type'] == "technology"
        assert result[0]['confidence'] == 0.8


# ============================================================================
# Model Tests: InternalOpsIntel
# ============================================================================

@pytest.mark.django_db
class TestInternalOpsIntelModel:
    """Tests for InternalOpsIntel Django model."""

    def test_create_internal_ops_intel(self):
        """Test creating InternalOpsIntel record."""
        job = ResearchJob.objects.create(
            client_name="Test Corp",
            status="completed"
        )

        internal_ops = InternalOpsIntel.objects.create(
            research_job=job,
            employee_sentiment={"overall_rating": 4.0},
            job_postings={"total_openings": 25},
            confidence_score=0.8,
            data_freshness="last_30_days"
        )

        assert internal_ops.id is not None
        assert internal_ops.research_job == job
        assert internal_ops.employee_sentiment['overall_rating'] == 4.0
        assert internal_ops.confidence_score == 0.8

    def test_internal_ops_intel_defaults(self):
        """Test InternalOpsIntel default values."""
        job = ResearchJob.objects.create(
            client_name="Test Corp",
            status="completed"
        )

        internal_ops = InternalOpsIntel.objects.create(research_job=job)

        assert internal_ops.employee_sentiment == {}
        assert internal_ops.linkedin_presence == {}
        assert internal_ops.social_media_mentions == []
        assert internal_ops.key_insights == []
        assert internal_ops.gap_correlations == []
        assert internal_ops.confidence_score == 0.0

    def test_internal_ops_intel_str(self):
        """Test InternalOpsIntel string representation."""
        job = ResearchJob.objects.create(
            client_name="Acme Corp",
            status="completed"
        )
        internal_ops = InternalOpsIntel.objects.create(research_job=job)

        assert "Acme Corp" in str(internal_ops)

    def test_research_job_internal_ops_relation(self):
        """Test ResearchJob to InternalOpsIntel relation."""
        job = ResearchJob.objects.create(
            client_name="Test Corp",
            status="completed"
        )
        internal_ops = InternalOpsIntel.objects.create(
            research_job=job,
            confidence_score=0.9
        )

        # Test reverse relation
        assert job.internal_ops == internal_ops
        assert job.internal_ops.confidence_score == 0.9

    def test_internal_ops_cascade_delete(self):
        """Test InternalOpsIntel is deleted when ResearchJob is deleted."""
        job = ResearchJob.objects.create(
            client_name="Test Corp",
            status="completed"
        )
        internal_ops = InternalOpsIntel.objects.create(research_job=job)
        internal_ops_id = internal_ops.id

        job.delete()

        assert not InternalOpsIntel.objects.filter(id=internal_ops_id).exists()


# ============================================================================
# Serializer Tests
# ============================================================================

@pytest.mark.django_db
class TestInternalOpsIntelSerializer:
    """Tests for InternalOpsIntel serializer."""

    def test_serialize_internal_ops_intel(self):
        """Test serializing InternalOpsIntel."""
        job = ResearchJob.objects.create(
            client_name="Test Corp",
            status="completed"
        )
        internal_ops = InternalOpsIntel.objects.create(
            research_job=job,
            employee_sentiment={"overall_rating": 4.2, "trend": "improving"},
            linkedin_presence={"follower_count": 10000},
            job_postings={"total_openings": 30},
            key_insights=["Strong growth trajectory"],
            gap_correlations=[{"gap_type": "technology", "confidence": 0.8}],
            confidence_score=0.85,
            data_freshness="last_7_days"
        )

        serializer = InternalOpsIntelSerializer(internal_ops)
        data = serializer.data

        assert data['employee_sentiment']['overall_rating'] == 4.2
        assert data['linkedin_presence']['follower_count'] == 10000
        assert data['job_postings']['total_openings'] == 30
        assert len(data['key_insights']) == 1
        assert len(data['gap_correlations']) == 1
        assert data['confidence_score'] == 0.85

    def test_research_job_detail_includes_internal_ops(self):
        """Test ResearchJobDetailSerializer includes internal_ops."""
        job = ResearchJob.objects.create(
            client_name="Test Corp",
            status="completed",
            vertical="technology"
        )
        InternalOpsIntel.objects.create(
            research_job=job,
            employee_sentiment={"overall_rating": 3.5},
            confidence_score=0.7
        )

        serializer = ResearchJobDetailSerializer(job)
        data = serializer.data

        assert 'internal_ops' in data
        assert data['internal_ops'] is not None
        assert data['internal_ops']['employee_sentiment']['overall_rating'] == 3.5

    def test_research_job_detail_without_internal_ops(self):
        """Test ResearchJobDetailSerializer handles missing internal_ops."""
        job = ResearchJob.objects.create(
            client_name="Test Corp",
            status="completed"
        )

        serializer = ResearchJobDetailSerializer(job)
        data = serializer.data

        assert 'internal_ops' in data
        assert data['internal_ops'] is None


# ============================================================================
# API Integration Tests
# ============================================================================

@pytest.mark.django_db
class TestInternalOpsAPIEndpoints:
    """Tests for API endpoints with internal ops data."""

    def test_get_research_job_with_internal_ops(self, api_client):
        """Test retrieving research job includes internal ops."""
        job = ResearchJob.objects.create(
            client_name="Test Corp",
            status="completed",
            result="Research complete"
        )
        InternalOpsIntel.objects.create(
            research_job=job,
            employee_sentiment={"overall_rating": 4.0, "trend": "improving"},
            key_insights=["Growing company", "Good culture"],
            confidence_score=0.85
        )

        url = reverse('research-detail', kwargs={'pk': job.id})
        response = api_client.get(url)

        assert response.status_code == 200
        assert 'internal_ops' in response.data
        assert response.data['internal_ops']['confidence_score'] == 0.85
        assert len(response.data['internal_ops']['key_insights']) == 2


# ============================================================================
# Workflow Node Tests
# ============================================================================

@pytest.mark.django_db
class TestInternalOpsWorkflowNodes:
    """Tests for workflow nodes related to internal ops."""

    @patch('research.services.internal_ops.InternalOpsService.research_internal_ops')
    def test_research_internal_ops_node(self, mock_research):
        """Test research_internal_ops node function."""
        from research.graph.nodes import research_internal_ops

        # Setup mock — new signature returns (InternalOpsData, Optional[GroundingMetadata])
        mock_ops_data = InternalOpsData(
            key_insights=["Test insight"],
            confidence_score=0.75
        )
        mock_research.return_value = (mock_ops_data, None)

        state = {
            'client_name': 'Test Corp',
            'vertical': 'technology',
            'research_report': {'website': 'https://test.com', 'company_overview': 'Test overview'},
            'status': 'classifying'
        }

        result = research_internal_ops(state)

        assert 'internal_ops' in result
        assert result['internal_ops'] is not None
        assert result['internal_ops']['confidence_score'] == 0.75

    @patch('research.services.internal_ops.InternalOpsService.research_internal_ops')
    def test_research_internal_ops_node_handles_failure(self, mock_research):
        """Test research_internal_ops node handles failures gracefully."""
        from research.graph.nodes import research_internal_ops

        mock_research.side_effect = Exception("API Error")

        state = {
            'client_name': 'Test Corp',
            'status': 'classifying',
            'research_report': {}
        }

        result = research_internal_ops(state)

        assert result['internal_ops'] is None

    @patch('research.services.gap_correlation.GapCorrelationService.correlate_gaps')
    @patch('research.services.gap_correlation.GapCorrelationService.correlations_to_dict')
    def test_correlate_internal_ops_node(self, mock_to_dict, mock_correlate):
        """Test correlate_internal_ops node function."""
        from research.graph.nodes import correlate_internal_ops

        mock_correlate.return_value = [
            GapCorrelation(
                gap_type="technology",
                description="Test gap",
                evidence="Test evidence",
                evidence_type="supporting",
                confidence=0.8,
                sales_implication="Test implication"
            )
        ]
        mock_to_dict.return_value = [
            {"gap_type": "technology", "confidence": 0.8}
        ]

        state = {
            'client_name': 'Test Corp',
            'vertical': 'technology',
            'gap_analysis': {'technology_gaps': ['Test gap']},
            'internal_ops': {'employee_sentiment': {'overall_rating': 3.5}},
            'status': 'gap_analysis'
        }

        result = correlate_internal_ops(state)

        assert result['status'] == 'completed'
        assert 'gap_correlations' in result
        assert len(result['gap_correlations']) == 1

    def test_correlate_internal_ops_skips_when_missing_data(self):
        """Test correlate_internal_ops skips when data is missing."""
        from research.graph.nodes import correlate_internal_ops

        # Missing gap_analysis
        state = {
            'client_name': 'Test Corp',
            'gap_analysis': None,
            'internal_ops': {'employee_sentiment': {}},
            'status': 'gap_analysis'
        }

        result = correlate_internal_ops(state)

        assert result['status'] == 'completed'
        assert result['gap_correlations'] == []


# ============================================================================
# Integration Tests: Full Workflow
# ============================================================================

@pytest.mark.django_db
class TestInternalOpsFullWorkflow:
    """Integration tests for full internal ops workflow."""

    def test_finalize_result_persists_internal_ops(self):
        """Test that finalize_result saves InternalOpsIntel."""
        from research.graph.nodes import finalize_result

        job = ResearchJob.objects.create(
            client_name="Test Corp",
            status="running"
        )

        state = {
            'job_id': str(job.id),
            'client_name': 'Test Corp',
            'vertical': 'technology',
            'research_report': {
                'company_overview': 'Test company',
                'founded_year': 2010
            },
            'competitor_case_studies': [],
            'gap_analysis': {
                'technology_gaps': ['Test gap'],
                'capability_gaps': [],
                'process_gaps': [],
                'recommendations': [],
                'priority_areas': [],
                'confidence_score': 0.7,
                'analysis_notes': ''
            },
            'internal_ops': {
                'employee_sentiment': {'overall_rating': 4.0, 'trend': 'improving'},
                'linkedin_presence': {'follower_count': 5000},
                'social_media_mentions': [],
                'job_postings': {'total_openings': 20},
                'news_sentiment': {'overall_sentiment': 'positive'},
                'key_insights': ['Growth phase', 'Strong culture'],
                'confidence_score': 0.8,
                'data_freshness': 'last_30_days',
                'analysis_notes': 'Test notes'
            },
            'gap_correlations': [
                {
                    'gap_type': 'technology',
                    'description': 'Test gap',
                    'evidence': 'Test evidence',
                    'evidence_type': 'supporting',
                    'confidence': 0.85,
                    'sales_implication': 'Test implication'
                }
            ],
            'status': 'completed'
        }

        result = finalize_result(state)

        # Verify InternalOpsIntel was created
        job.refresh_from_db()
        assert hasattr(job, 'internal_ops')
        internal_ops = job.internal_ops
        assert internal_ops.employee_sentiment['overall_rating'] == 4.0
        assert internal_ops.linkedin_presence['follower_count'] == 5000
        assert len(internal_ops.key_insights) == 2
        assert len(internal_ops.gap_correlations) == 1
        assert internal_ops.confidence_score == 0.8

    def test_workflow_handles_missing_internal_ops_gracefully(self):
        """Test finalize works when internal_ops is None."""
        from research.graph.nodes import finalize_result

        job = ResearchJob.objects.create(
            client_name="Test Corp",
            status="running"
        )

        state = {
            'job_id': str(job.id),
            'client_name': 'Test Corp',
            'research_report': {'company_overview': 'Test'},
            'competitor_case_studies': [],
            'gap_analysis': None,
            'internal_ops': None,
            'gap_correlations': None,
            'status': 'completed'
        }

        # Should not raise an exception
        result = finalize_result(state)

        assert result['status'] == 'completed'
        # Verify no InternalOpsIntel was created
        job.refresh_from_db()
        assert not hasattr(job, 'internal_ops') or job.internal_ops is None
