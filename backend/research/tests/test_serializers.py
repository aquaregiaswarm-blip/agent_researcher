"""Tests for research serializers — field assertions and edge cases."""
import pytest
from research.models import ResearchJob, ResearchReport, CompetitorCaseStudy, GapAnalysis, InternalOpsIntel
from research.serializers import (
    ResearchReportSerializer,
    CompetitorCaseStudySerializer,
    GapAnalysisSerializer,
    ResearchJobDetailSerializer,
)


@pytest.fixture
def completed_job():
    return ResearchJob.objects.create(
        client_name="Test Corp",
        status="completed",
        vertical="technology",
    )


# ---------------------------------------------------------------------------
# ResearchReportSerializer
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestResearchReportSerializer:

    def test_all_enrichment_fields_present(self, completed_job):
        report = ResearchReport.objects.create(
            research_job=completed_job,
            company_overview="A tech company",
            cloud_footprint="AWS and GCP",
            security_posture="SOC2 certified",
            data_maturity="advanced",
            financial_signals=["Recent Series B", "Profitable"],
            tech_partnerships=["Microsoft", "Snowflake"],
            web_sources=[{"uri": "https://example.com", "title": "Example"}],
        )

        data = ResearchReportSerializer(report).data

        assert data['cloud_footprint'] == "AWS and GCP"
        assert data['security_posture'] == "SOC2 certified"
        assert data['data_maturity'] == "advanced"
        assert data['financial_signals'] == ["Recent Series B", "Profitable"]
        assert data['tech_partnerships'] == ["Microsoft", "Snowflake"]

    def test_web_sources_serialized_as_list(self, completed_job):
        report = ResearchReport.objects.create(
            research_job=completed_job,
            web_sources=[
                {"uri": "https://source1.com", "title": "Source 1"},
                {"uri": "https://source2.com", "title": "Source 2"},
            ],
        )

        data = ResearchReportSerializer(report).data

        assert isinstance(data['web_sources'], list)
        assert len(data['web_sources']) == 2

    def test_nullable_fields_return_none(self, completed_job):
        report = ResearchReport.objects.create(research_job=completed_job)

        data = ResearchReportSerializer(report).data

        assert data['founded_year'] is None
        assert data['company_overview'] == ""


# ---------------------------------------------------------------------------
# CompetitorCaseStudySerializer
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestCompetitorCaseStudySerializer:

    def test_all_fields_present(self, completed_job):
        cs = CompetitorCaseStudy.objects.create(
            research_job=completed_job,
            competitor_name="RivalCo",
            vertical="technology",
            case_study_title="AI Transformation",
            summary="Did great things.",
            technologies_used=["Python", "TensorFlow"],
            outcomes=["30% cost reduction"],
            source_url="https://rival.co/case-study",
            relevance_score=0.87,
        )

        data = CompetitorCaseStudySerializer(cs).data

        assert data['competitor_name'] == "RivalCo"
        assert data['vertical'] == "technology"
        assert data['case_study_title'] == "AI Transformation"
        assert data['summary'] == "Did great things."
        assert data['technologies_used'] == ["Python", "TensorFlow"]
        assert data['outcomes'] == ["30% cost reduction"]
        assert data['source_url'] == "https://rival.co/case-study"
        assert data['relevance_score'] == 0.87

    def test_source_url_accepts_long_url(self, completed_job):
        """Regression: source_url max_length was 200, now 2000."""
        long_url = "https://example.com/" + "x" * 1900  # ~1920 chars
        cs = CompetitorCaseStudy.objects.create(
            research_job=completed_job,
            competitor_name="LongURLCo",
            source_url=long_url,
        )

        data = CompetitorCaseStudySerializer(cs).data

        assert len(data['source_url']) > 200


# ---------------------------------------------------------------------------
# GapAnalysisSerializer
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestGapAnalysisSerializer:

    def test_all_fields_present(self, completed_job):
        gap = GapAnalysis.objects.create(
            research_job=completed_job,
            technology_gaps=["T1", "T2"],
            capability_gaps=["C1"],
            process_gaps=["P1"],
            recommendations=["R1", "R2"],
            priority_areas=["PA1"],
            confidence_score=0.78,
            analysis_notes="Based on job postings and public data.",
        )

        data = GapAnalysisSerializer(gap).data

        assert data['technology_gaps'] == ["T1", "T2"]
        assert data['capability_gaps'] == ["C1"]
        assert data['process_gaps'] == ["P1"]
        assert data['recommendations'] == ["R1", "R2"]
        assert data['priority_areas'] == ["PA1"]
        assert data['confidence_score'] == 0.78
        assert data['analysis_notes'] == "Based on job postings and public data."

    def test_confidence_score_range(self, completed_job):
        gap_low = GapAnalysis.objects.create(
            research_job=completed_job,
            confidence_score=0.0,
        )
        data = GapAnalysisSerializer(gap_low).data
        assert data['confidence_score'] == 0.0

        # Create a separate job for the high score test
        job2 = ResearchJob.objects.create(client_name="Test Corp 2", status="completed")
        gap_high = GapAnalysis.objects.create(
            research_job=job2,
            confidence_score=1.0,
        )
        data2 = GapAnalysisSerializer(gap_high).data
        assert data2['confidence_score'] == 1.0


# ---------------------------------------------------------------------------
# ResearchJobDetailSerializer
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestResearchJobDetailSerializer:

    def test_includes_report_competitor_gap_intel_fields(self, completed_job):
        ResearchReport.objects.create(
            research_job=completed_job,
            company_overview="Test overview",
        )
        CompetitorCaseStudy.objects.create(
            research_job=completed_job,
            competitor_name="RivalCo",
        )
        GapAnalysis.objects.create(
            research_job=completed_job,
            confidence_score=0.7,
        )
        InternalOpsIntel.objects.create(
            research_job=completed_job,
            confidence_score=0.8,
        )

        data = ResearchJobDetailSerializer(completed_job).data

        assert 'report' in data
        assert data['report']['company_overview'] == "Test overview"
        assert len(data['competitor_case_studies']) == 1
        assert data['gap_analysis']['confidence_score'] == 0.7
        assert data['internal_ops']['confidence_score'] == 0.8

    def test_nested_serializers_null_when_missing(self):
        job = ResearchJob.objects.create(client_name="Empty Corp", status="completed")

        data = ResearchJobDetailSerializer(job).data

        assert data['report'] is None
        assert data['competitor_case_studies'] == []
        assert data['gap_analysis'] is None
        assert data['internal_ops'] is None

    def test_includes_vertical_and_status(self, completed_job):
        data = ResearchJobDetailSerializer(completed_job).data

        assert data['status'] == 'completed'
        assert data['vertical'] == 'technology'
        assert data['client_name'] == 'Test Corp'
