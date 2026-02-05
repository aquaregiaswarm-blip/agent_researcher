from rest_framework import serializers
from .models import ResearchJob, ResearchReport, CompetitorCaseStudy, GapAnalysis, InternalOpsIntel


class ResearchJobCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating a new research job."""

    class Meta:
        model = ResearchJob
        fields = ['id', 'client_name', 'sales_history', 'prompt']
        read_only_fields = ['id']


class ResearchReportSerializer(serializers.ModelSerializer):
    """Serializer for structured research report (AGE-10)."""

    class Meta:
        model = ResearchReport
        fields = [
            'id',
            'company_overview',
            'founded_year',
            'headquarters',
            'employee_count',
            'annual_revenue',
            'website',
            'recent_news',
            'decision_makers',
            'pain_points',
            'opportunities',
            'digital_maturity',
            'ai_footprint',
            'ai_adoption_stage',
            'strategic_goals',
            'key_initiatives',
            'talking_points',
            'web_sources',
            'created_at',
            'updated_at',
        ]
        read_only_fields = fields


class CompetitorCaseStudySerializer(serializers.ModelSerializer):
    """Serializer for competitor case studies (AGE-12)."""

    class Meta:
        model = CompetitorCaseStudy
        fields = [
            'id',
            'competitor_name',
            'vertical',
            'case_study_title',
            'summary',
            'technologies_used',
            'outcomes',
            'source_url',
            'relevance_score',
            'created_at',
        ]
        read_only_fields = fields


class GapAnalysisSerializer(serializers.ModelSerializer):
    """Serializer for gap analysis (AGE-13)."""

    class Meta:
        model = GapAnalysis
        fields = [
            'id',
            'technology_gaps',
            'capability_gaps',
            'process_gaps',
            'recommendations',
            'priority_areas',
            'confidence_score',
            'analysis_notes',
            'created_at',
            'updated_at',
        ]
        read_only_fields = fields


class InternalOpsIntelSerializer(serializers.ModelSerializer):
    """Serializer for internal operations intelligence (AGE-20)."""

    class Meta:
        model = InternalOpsIntel
        fields = [
            'id',
            'employee_sentiment',
            'linkedin_presence',
            'social_media_mentions',
            'job_postings',
            'news_sentiment',
            'key_insights',
            'gap_correlations',
            'confidence_score',
            'data_freshness',
            'analysis_notes',
            'created_at',
            'updated_at',
        ]
        read_only_fields = fields


class ResearchJobDetailSerializer(serializers.ModelSerializer):
    """Serializer for retrieving research job details."""

    report = ResearchReportSerializer(read_only=True)
    competitor_case_studies = CompetitorCaseStudySerializer(many=True, read_only=True)
    gap_analysis = GapAnalysisSerializer(read_only=True)
    internal_ops = InternalOpsIntelSerializer(read_only=True)

    class Meta:
        model = ResearchJob
        fields = [
            'id',
            'client_name',
            'sales_history',
            'prompt',
            'status',
            'result',
            'error',
            'vertical',
            'report',
            'competitor_case_studies',
            'gap_analysis',
            'internal_ops',
            'created_at',
            'updated_at',
        ]
        read_only_fields = [
            'id',
            'status',
            'result',
            'error',
            'vertical',
            'report',
            'competitor_case_studies',
            'gap_analysis',
            'internal_ops',
            'created_at',
            'updated_at',
        ]
