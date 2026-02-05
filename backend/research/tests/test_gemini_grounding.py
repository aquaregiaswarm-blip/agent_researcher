"""Tests for Gemini client Google Search grounding functionality."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from research.services.gemini import (
    GeminiClient,
    WebSource,
    GroundingMetadata,
    GroundedQueryResult,
    ResearchReportData,
)


class TestWebSource:
    """Tests for WebSource dataclass."""

    def test_web_source_creation(self):
        """Test creating a WebSource with uri and title."""
        source = WebSource(uri="https://example.com", title="Example Site")
        assert source.uri == "https://example.com"
        assert source.title == "Example Site"

    def test_web_source_defaults(self):
        """Test WebSource default values."""
        source = WebSource()
        assert source.uri == ""
        assert source.title == ""

    def test_web_source_to_dict(self):
        """Test converting WebSource to dictionary."""
        source = WebSource(uri="https://example.com", title="Example")
        result = asdict(source)
        assert result == {"uri": "https://example.com", "title": "Example"}


class TestGroundingMetadata:
    """Tests for GroundingMetadata dataclass."""

    def test_grounding_metadata_creation(self):
        """Test creating GroundingMetadata with web sources."""
        sources = [
            WebSource(uri="https://example1.com", title="Source 1"),
            WebSource(uri="https://example2.com", title="Source 2"),
        ]
        metadata = GroundingMetadata(
            web_sources=sources,
            search_queries=["query 1", "query 2"]
        )
        assert len(metadata.web_sources) == 2
        assert len(metadata.search_queries) == 2

    def test_grounding_metadata_defaults(self):
        """Test GroundingMetadata default values."""
        metadata = GroundingMetadata()
        assert metadata.web_sources == []
        assert metadata.search_queries == []

    def test_grounding_metadata_to_dict(self):
        """Test converting GroundingMetadata to dictionary."""
        sources = [WebSource(uri="https://example.com", title="Example")]
        metadata = GroundingMetadata(
            web_sources=sources,
            search_queries=["test query"]
        )
        result = metadata.to_dict()
        assert result == {
            "web_sources": [{"uri": "https://example.com", "title": "Example"}],
            "search_queries": ["test query"],
        }

    def test_grounding_metadata_to_dict_with_dict_sources(self):
        """Test to_dict handles both WebSource objects and dicts."""
        metadata = GroundingMetadata(
            web_sources=[{"uri": "https://example.com", "title": "Example"}],
            search_queries=[]
        )
        result = metadata.to_dict()
        assert result["web_sources"] == [{"uri": "https://example.com", "title": "Example"}]


class TestGeminiClientGroundingExtraction:
    """Tests for grounding metadata extraction."""

    def test_extract_grounding_metadata_with_valid_response(self):
        """Test extracting grounding metadata from a valid response."""
        client = GeminiClient(api_key="test-key")

        # Mock the response structure
        mock_web = Mock()
        mock_web.uri = "https://example.com/article"
        mock_web.title = "Article Title"

        mock_chunk = Mock()
        mock_chunk.web = mock_web

        mock_grounding = Mock()
        mock_grounding.grounding_chunks = [mock_chunk]
        mock_grounding.web_search_queries = ["company name", "company news"]

        mock_candidate = Mock()
        mock_candidate.grounding_metadata = mock_grounding

        mock_response = Mock()
        mock_response.candidates = [mock_candidate]

        result = client._extract_grounding_metadata(mock_response)

        assert result is not None
        assert len(result.web_sources) == 1
        assert result.web_sources[0].uri == "https://example.com/article"
        assert result.web_sources[0].title == "Article Title"
        assert len(result.search_queries) == 2

    def test_extract_grounding_metadata_no_candidates(self):
        """Test extraction returns None when no candidates."""
        client = GeminiClient(api_key="test-key")

        mock_response = Mock()
        mock_response.candidates = []

        result = client._extract_grounding_metadata(mock_response)
        assert result is None

    def test_extract_grounding_metadata_no_grounding_metadata(self):
        """Test extraction returns None when no grounding_metadata."""
        client = GeminiClient(api_key="test-key")

        mock_candidate = Mock()
        mock_candidate.grounding_metadata = None

        mock_response = Mock()
        mock_response.candidates = [mock_candidate]

        result = client._extract_grounding_metadata(mock_response)
        assert result is None

    def test_extract_grounding_metadata_empty_chunks(self):
        """Test extraction returns None when grounding_chunks is empty."""
        client = GeminiClient(api_key="test-key")

        mock_grounding = Mock()
        mock_grounding.grounding_chunks = []
        mock_grounding.web_search_queries = []

        mock_candidate = Mock()
        mock_candidate.grounding_metadata = mock_grounding

        mock_response = Mock()
        mock_response.candidates = [mock_candidate]

        result = client._extract_grounding_metadata(mock_response)
        assert result is None

    def test_extract_grounding_metadata_handles_missing_attributes(self):
        """Test extraction handles missing attributes gracefully."""
        client = GeminiClient(api_key="test-key")

        mock_chunk = Mock(spec=[])  # No 'web' attribute

        mock_grounding = Mock()
        mock_grounding.grounding_chunks = [mock_chunk]
        mock_grounding.web_search_queries = ["query"]

        mock_candidate = Mock()
        mock_candidate.grounding_metadata = mock_grounding

        mock_response = Mock()
        mock_response.candidates = [mock_candidate]

        result = client._extract_grounding_metadata(mock_response)
        # Should still return metadata with search queries even if chunks don't have web
        assert result is not None
        assert result.search_queries == ["query"]

    def test_extract_grounding_metadata_handles_none_uri_title(self):
        """Test extraction handles None uri/title values."""
        client = GeminiClient(api_key="test-key")

        mock_web = Mock()
        mock_web.uri = None
        mock_web.title = None

        mock_chunk = Mock()
        mock_chunk.web = mock_web

        mock_grounding = Mock()
        mock_grounding.grounding_chunks = [mock_chunk]
        mock_grounding.web_search_queries = []

        mock_candidate = Mock()
        mock_candidate.grounding_metadata = mock_grounding

        mock_response = Mock()
        mock_response.candidates = [mock_candidate]

        result = client._extract_grounding_metadata(mock_response)
        assert result is not None
        assert result.web_sources[0].uri == ""
        assert result.web_sources[0].title == ""

    def test_extract_grounding_metadata_handles_exception(self):
        """Test extraction returns None on exception."""
        client = GeminiClient(api_key="test-key")

        mock_response = Mock()
        mock_response.candidates = Mock(side_effect=Exception("Test error"))

        result = client._extract_grounding_metadata(mock_response)
        assert result is None


class TestGeminiClientDeepResearch:
    """Tests for conduct_deep_research with grounding."""

    def test_conduct_deep_research_returns_tuple(self):
        """Test that conduct_deep_research returns tuple of (data, metadata)."""
        # Mock response with valid JSON
        mock_response = Mock()
        mock_response.text = '''
        {
            "company_overview": "Test company overview",
            "founded_year": 2020,
            "headquarters": "San Francisco, CA",
            "employee_count": "100-500",
            "annual_revenue": "$10M-$50M",
            "website": "https://test.com",
            "recent_news": [],
            "decision_makers": [],
            "pain_points": ["Pain 1"],
            "opportunities": ["Opportunity 1"],
            "digital_maturity": "developing",
            "ai_footprint": "Early stage AI adoption",
            "ai_adoption_stage": "experimenting",
            "strategic_goals": ["Goal 1"],
            "key_initiatives": ["Initiative 1"],
            "talking_points": ["Point 1"]
        }
        '''
        mock_response.candidates = []  # No grounding metadata

        mock_genai_client = Mock()
        mock_genai_client.models.generate_content.return_value = mock_response

        with patch('google.genai.Client', return_value=mock_genai_client):
            client = GeminiClient(api_key="test-key")
            result = client.conduct_deep_research("Test Company")

        assert isinstance(result, tuple)
        assert len(result) == 2
        report_data, grounding_metadata = result
        assert isinstance(report_data, ResearchReportData)
        assert report_data.company_overview == "Test company overview"
        assert grounding_metadata is None  # No grounding in mock

    def test_conduct_deep_research_with_grounding_metadata(self):
        """Test that grounding metadata is extracted and returned."""
        # Mock web source
        mock_web = Mock()
        mock_web.uri = "https://source.com/article"
        mock_web.title = "Source Article"

        mock_chunk = Mock()
        mock_chunk.web = mock_web

        mock_grounding = Mock()
        mock_grounding.grounding_chunks = [mock_chunk]
        mock_grounding.web_search_queries = ["test company"]

        mock_candidate = Mock()
        mock_candidate.grounding_metadata = mock_grounding

        mock_response = Mock()
        mock_response.text = '''
        {
            "company_overview": "Test overview",
            "founded_year": 2020,
            "headquarters": "NYC",
            "employee_count": "50",
            "annual_revenue": "$5M",
            "website": "https://test.com",
            "recent_news": [],
            "decision_makers": [],
            "pain_points": [],
            "opportunities": [],
            "digital_maturity": "developing",
            "ai_footprint": "",
            "ai_adoption_stage": "exploring",
            "strategic_goals": [],
            "key_initiatives": [],
            "talking_points": []
        }
        '''
        mock_response.candidates = [mock_candidate]

        mock_genai_client = Mock()
        mock_genai_client.models.generate_content.return_value = mock_response

        with patch('google.genai.Client', return_value=mock_genai_client):
            client = GeminiClient(api_key="test-key")
            result = client.conduct_deep_research("Test Company")

        report_data, grounding_metadata = result
        assert grounding_metadata is not None
        assert len(grounding_metadata.web_sources) == 1
        assert grounding_metadata.web_sources[0].uri == "https://source.com/article"

    def test_conduct_deep_research_handles_json_decode_error(self):
        """Test handling of invalid JSON response."""
        mock_response = Mock()
        mock_response.text = "This is not valid JSON"
        mock_response.candidates = []

        mock_genai_client = Mock()
        mock_genai_client.models.generate_content.return_value = mock_response

        with patch('google.genai.Client', return_value=mock_genai_client):
            client = GeminiClient(api_key="test-key")
            result = client.conduct_deep_research("Test Company")

        report_data, grounding_metadata = result
        assert "structured parsing failed" in report_data.company_overview

    def test_conduct_deep_research_handles_markdown_code_blocks(self):
        """Test handling of JSON wrapped in markdown code blocks."""
        mock_response = Mock()
        mock_response.text = '''```json
        {
            "company_overview": "Markdown wrapped",
            "founded_year": 2021,
            "headquarters": "Boston",
            "employee_count": "200",
            "annual_revenue": "$20M",
            "website": "https://test.com",
            "recent_news": [],
            "decision_makers": [],
            "pain_points": [],
            "opportunities": [],
            "digital_maturity": "maturing",
            "ai_footprint": "",
            "ai_adoption_stage": "implementing",
            "strategic_goals": [],
            "key_initiatives": [],
            "talking_points": []
        }
```'''
        mock_response.candidates = []

        mock_genai_client = Mock()
        mock_genai_client.models.generate_content.return_value = mock_response

        with patch('google.genai.Client', return_value=mock_genai_client):
            client = GeminiClient(api_key="test-key")
            result = client.conduct_deep_research("Test Company")

        report_data, _ = result
        assert report_data.company_overview == "Markdown wrapped"


@pytest.mark.django_db
class TestWebSourcesIntegration:
    """Integration tests for web sources in the research workflow."""

    def test_research_report_with_web_sources(self):
        """Test that ResearchReport model stores web_sources correctly."""
        from research.models import ResearchJob, ResearchReport

        job = ResearchJob.objects.create(
            client_name="Test Corp",
            status="completed",
        )

        web_sources = [
            {"uri": "https://example1.com", "title": "Source 1"},
            {"uri": "https://example2.com", "title": "Source 2"},
        ]

        report = ResearchReport.objects.create(
            research_job=job,
            company_overview="Test overview",
            web_sources=web_sources,
        )

        # Refresh from database
        report.refresh_from_db()

        assert len(report.web_sources) == 2
        assert report.web_sources[0]["uri"] == "https://example1.com"
        assert report.web_sources[1]["title"] == "Source 2"

    def test_research_report_empty_web_sources(self):
        """Test ResearchReport with no web sources."""
        from research.models import ResearchJob, ResearchReport

        job = ResearchJob.objects.create(
            client_name="Test Corp",
            status="completed",
        )

        report = ResearchReport.objects.create(
            research_job=job,
            company_overview="Test overview",
        )

        report.refresh_from_db()
        assert report.web_sources == []

    def test_serializer_includes_web_sources(self):
        """Test that ResearchReportSerializer includes web_sources."""
        from research.models import ResearchJob, ResearchReport
        from research.serializers import ResearchReportSerializer

        job = ResearchJob.objects.create(
            client_name="Test Corp",
            status="completed",
        )

        web_sources = [{"uri": "https://example.com", "title": "Example"}]

        report = ResearchReport.objects.create(
            research_job=job,
            company_overview="Test",
            web_sources=web_sources,
        )

        serializer = ResearchReportSerializer(report)
        data = serializer.data

        assert "web_sources" in data
        assert len(data["web_sources"]) == 1
        assert data["web_sources"][0]["uri"] == "https://example.com"

    def test_research_job_detail_includes_web_sources(self, api_client):
        """Test that research job detail endpoint includes web_sources."""
        from django.urls import reverse
        from rest_framework.test import APIClient
        from research.models import ResearchJob, ResearchReport

        client = APIClient()

        job = ResearchJob.objects.create(
            client_name="Test Corp",
            status="completed",
        )

        web_sources = [
            {"uri": "https://source1.com", "title": "Source 1"},
            {"uri": "https://source2.com", "title": "Source 2"},
        ]

        ResearchReport.objects.create(
            research_job=job,
            company_overview="Test overview",
            web_sources=web_sources,
        )

        url = reverse('research-detail', kwargs={'pk': job.id})
        response = client.get(url)

        assert response.status_code == 200
        assert "report" in response.data
        assert "web_sources" in response.data["report"]
        assert len(response.data["report"]["web_sources"]) == 2


@pytest.fixture
def api_client():
    """Provide API client for tests."""
    from rest_framework.test import APIClient
    return APIClient()


class TestGroundedQueryResult:
    """Tests for GroundedQueryResult dataclass."""

    def test_grounded_query_result_creation(self):
        """Test creating a GroundedQueryResult."""
        result = GroundedQueryResult(
            query_type="profile",
            text="Company info here",
            success=True,
        )
        assert result.query_type == "profile"
        assert result.text == "Company info here"
        assert result.success is True
        assert result.grounding_metadata is None
        assert result.error is None

    def test_grounded_query_result_with_metadata(self):
        """Test GroundedQueryResult with grounding metadata."""
        metadata = GroundingMetadata(
            web_sources=[WebSource(uri="https://example.com", title="Example")],
            search_queries=["test query"],
        )
        result = GroundedQueryResult(
            query_type="news",
            text="Recent news",
            grounding_metadata=metadata,
            success=True,
        )
        assert result.grounding_metadata is not None
        assert len(result.grounding_metadata.web_sources) == 1

    def test_grounded_query_result_failure(self):
        """Test GroundedQueryResult for failed query."""
        result = GroundedQueryResult(
            query_type="leadership",
            text="",
            success=False,
            error="API timeout",
        )
        assert result.success is False
        assert result.error == "API timeout"


class TestConductGroundedQuery:
    """Tests for _conduct_grounded_query method."""

    def test_conduct_grounded_query_success(self):
        """Test successful grounded query."""
        mock_web = Mock()
        mock_web.uri = "https://source.com/article"
        mock_web.title = "Article"

        mock_chunk = Mock()
        mock_chunk.web = mock_web

        mock_grounding = Mock()
        mock_grounding.grounding_chunks = [mock_chunk]
        mock_grounding.web_search_queries = ["company name"]

        mock_candidate = Mock()
        mock_candidate.grounding_metadata = mock_grounding

        mock_response = Mock()
        mock_response.text = "Company information found"
        mock_response.candidates = [mock_candidate]

        mock_genai_client = Mock()
        mock_genai_client.models.generate_content.return_value = mock_response

        with patch('google.genai.Client', return_value=mock_genai_client):
            client = GeminiClient(api_key="test-key")
            result = client._conduct_grounded_query("Find info about TestCo", "profile")

        assert result.success is True
        assert result.query_type == "profile"
        assert result.text == "Company information found"
        assert result.grounding_metadata is not None
        assert len(result.grounding_metadata.web_sources) == 1

    def test_conduct_grounded_query_failure(self):
        """Test grounded query that fails."""
        mock_genai_client = Mock()
        mock_genai_client.models.generate_content.side_effect = Exception("API Error")

        with patch('google.genai.Client', return_value=mock_genai_client):
            client = GeminiClient(api_key="test-key")
            result = client._conduct_grounded_query("Find info about TestCo", "news")

        assert result.success is False
        assert result.query_type == "news"
        assert result.error == "API Error"


class TestRunParallelGroundedQueries:
    """Tests for _run_parallel_grounded_queries method."""

    def test_run_parallel_grounded_queries_all_success(self):
        """Test parallel queries with all succeeding."""
        def mock_generate_content(*args, **kwargs):
            mock_response = Mock()
            mock_response.text = "Research results"
            mock_response.candidates = []
            return mock_response

        mock_genai_client = Mock()
        mock_genai_client.models.generate_content.side_effect = mock_generate_content

        with patch('google.genai.Client', return_value=mock_genai_client):
            client = GeminiClient(api_key="test-key")
            results = client._run_parallel_grounded_queries("Test Company")

        assert len(results) == 4
        assert 'profile' in results
        assert 'news' in results
        assert 'leadership' in results
        assert 'technology' in results
        assert all(r.success for r in results.values())

    def test_run_parallel_grounded_queries_partial_failure(self):
        """Test parallel queries with some failing."""
        call_count = [0]

        def mock_generate_content(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Network error")
            mock_response = Mock()
            mock_response.text = "Research results"
            mock_response.candidates = []
            return mock_response

        mock_genai_client = Mock()
        mock_genai_client.models.generate_content.side_effect = mock_generate_content

        with patch('google.genai.Client', return_value=mock_genai_client):
            client = GeminiClient(api_key="test-key")
            results = client._run_parallel_grounded_queries("Test Company")

        assert len(results) == 4
        # At least 3 should succeed
        successful = sum(1 for r in results.values() if r.success)
        assert successful >= 3


class TestMergeGroundingMetadata:
    """Tests for _merge_grounding_metadata method."""

    def test_merge_grounding_metadata_deduplication(self):
        """Test that duplicate URIs are removed."""
        client = GeminiClient(api_key="test-key")

        results = {
            'profile': GroundedQueryResult(
                query_type='profile',
                text='Profile data',
                grounding_metadata=GroundingMetadata(
                    web_sources=[
                        WebSource(uri="https://example.com/1", title="Source 1"),
                        WebSource(uri="https://example.com/2", title="Source 2"),
                    ],
                    search_queries=["profile query"],
                ),
            ),
            'news': GroundedQueryResult(
                query_type='news',
                text='News data',
                grounding_metadata=GroundingMetadata(
                    web_sources=[
                        WebSource(uri="https://example.com/1", title="Source 1 Dup"),  # Duplicate URI
                        WebSource(uri="https://example.com/3", title="Source 3"),
                    ],
                    search_queries=["news query"],
                ),
            ),
        }

        merged = client._merge_grounding_metadata(results)

        assert merged is not None
        # Should have 3 unique sources (deduped by URI)
        assert len(merged.web_sources) == 3
        uris = [s.uri for s in merged.web_sources]
        assert len(set(uris)) == 3  # All unique

    def test_merge_grounding_metadata_empty_results(self):
        """Test merging when no grounding metadata."""
        client = GeminiClient(api_key="test-key")

        results = {
            'profile': GroundedQueryResult(
                query_type='profile',
                text='Profile data',
                grounding_metadata=None,
            ),
            'news': GroundedQueryResult(
                query_type='news',
                text='News data',
                grounding_metadata=None,
            ),
        }

        merged = client._merge_grounding_metadata(results)
        assert merged is None

    def test_merge_grounding_metadata_combines_queries(self):
        """Test that search queries are combined and deduplicated."""
        client = GeminiClient(api_key="test-key")

        results = {
            'profile': GroundedQueryResult(
                query_type='profile',
                text='Profile data',
                grounding_metadata=GroundingMetadata(
                    web_sources=[WebSource(uri="https://a.com", title="A")],
                    search_queries=["company profile", "company info"],
                ),
            ),
            'news': GroundedQueryResult(
                query_type='news',
                text='News data',
                grounding_metadata=GroundingMetadata(
                    web_sources=[WebSource(uri="https://b.com", title="B")],
                    search_queries=["company news", "company info"],  # "company info" is duplicate
                ),
            ),
        }

        merged = client._merge_grounding_metadata(results)

        assert merged is not None
        # Should have 3 unique queries (deduped)
        assert len(merged.search_queries) == 3


class TestApplyFallbackDefaults:
    """Tests for _apply_fallback_defaults method."""

    def test_fallback_for_failed_leadership_query(self):
        """Test fallback when leadership query fails."""
        client = GeminiClient(api_key="test-key")

        report_data = ResearchReportData(
            company_overview="Test company",
            decision_makers=None,  # Will become [] due to default
        )

        query_results = {
            'profile': GroundedQueryResult(query_type='profile', success=True),
            'leadership': GroundedQueryResult(query_type='leadership', success=False, error="Failed"),
        }

        result = client._apply_fallback_defaults(report_data, query_results)
        assert result.decision_makers == []

    def test_fallback_for_failed_technology_query(self):
        """Test fallback when technology query fails."""
        client = GeminiClient(api_key="test-key")

        report_data = ResearchReportData(
            company_overview="Test company",
            digital_maturity="",  # Empty
        )

        query_results = {
            'profile': GroundedQueryResult(query_type='profile', success=True),
            'technology': GroundedQueryResult(query_type='technology', success=False, error="Failed"),
        }

        result = client._apply_fallback_defaults(report_data, query_results)
        assert result.digital_maturity == "unknown"


class TestParallelDeepResearch:
    """Integration tests for the 3-phase parallel research approach."""

    def test_conduct_deep_research_three_phase(self):
        """Test that conduct_deep_research uses the 3-phase approach."""
        # Track call order
        call_prompts = []

        def mock_generate_content(*args, **kwargs):
            contents = kwargs.get('contents') or (args[1] if len(args) > 1 else '')
            call_prompts.append(contents[:50])  # Store first 50 chars

            mock_response = Mock()
            mock_response.candidates = []

            # Return JSON for the formatting phase
            if 'JSON' in contents or 'format' in contents.lower():
                mock_response.text = '''{
                    "company_overview": "Test overview",
                    "founded_year": 2020,
                    "headquarters": "NYC",
                    "employee_count": "100",
                    "annual_revenue": "$10M",
                    "website": "https://test.com",
                    "recent_news": [],
                    "decision_makers": [],
                    "pain_points": ["Pain 1"],
                    "opportunities": ["Opp 1"],
                    "digital_maturity": "developing",
                    "ai_footprint": "Early",
                    "ai_adoption_stage": "exploring",
                    "strategic_goals": ["Goal 1"],
                    "key_initiatives": ["Init 1"],
                    "talking_points": ["Point 1"]
                }'''
            else:
                mock_response.text = "Research findings for the company."

            return mock_response

        mock_genai_client = Mock()
        mock_genai_client.models.generate_content.side_effect = mock_generate_content

        with patch('google.genai.Client', return_value=mock_genai_client):
            client = GeminiClient(api_key="test-key")
            report_data, grounding_metadata = client.conduct_deep_research("Test Company")

        # Should have at least 6 calls: 4 parallel + synthesis + formatting
        assert mock_genai_client.models.generate_content.call_count >= 6
        assert report_data.company_overview == "Test overview"

    def test_conduct_deep_research_aggregates_sources(self):
        """Test that sources from all queries are aggregated."""
        call_count = [0]

        def mock_generate_content(*args, **kwargs):
            call_count[0] += 1
            mock_response = Mock()

            # Create unique grounding for each of the first 4 calls (parallel queries)
            if call_count[0] <= 4:
                mock_web = Mock()
                mock_web.uri = f"https://source{call_count[0]}.com/article"
                mock_web.title = f"Source {call_count[0]}"

                mock_chunk = Mock()
                mock_chunk.web = mock_web

                mock_grounding = Mock()
                mock_grounding.grounding_chunks = [mock_chunk]
                mock_grounding.web_search_queries = [f"query {call_count[0]}"]

                mock_candidate = Mock()
                mock_candidate.grounding_metadata = mock_grounding

                mock_response.candidates = [mock_candidate]
                mock_response.text = f"Research from query {call_count[0]}"
            else:
                mock_response.candidates = []
                if call_count[0] == 6:  # Formatting phase
                    mock_response.text = '''{
                        "company_overview": "Test",
                        "founded_year": 2020,
                        "headquarters": "NYC",
                        "employee_count": "100",
                        "annual_revenue": "$10M",
                        "website": "https://test.com",
                        "recent_news": [],
                        "decision_makers": [],
                        "pain_points": [],
                        "opportunities": [],
                        "digital_maturity": "developing",
                        "ai_footprint": "",
                        "ai_adoption_stage": "exploring",
                        "strategic_goals": [],
                        "key_initiatives": [],
                        "talking_points": []
                    }'''
                else:
                    mock_response.text = "Synthesis results"

            return mock_response

        mock_genai_client = Mock()
        mock_genai_client.models.generate_content.side_effect = mock_generate_content

        with patch('google.genai.Client', return_value=mock_genai_client):
            client = GeminiClient(api_key="test-key")
            report_data, grounding_metadata = client.conduct_deep_research("Test Company")

        # Should have 4 unique sources from the 4 parallel queries
        assert grounding_metadata is not None
        assert len(grounding_metadata.web_sources) == 4
