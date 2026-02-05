import uuid
from django.db import models
from .constants import Vertical, DigitalMaturityLevel, AIAdoptionStage


class ResearchJob(models.Model):
    """Model to track research job status and results."""

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    client_name = models.CharField(max_length=255)
    sales_history = models.TextField(blank=True)
    prompt = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    result = models.TextField(blank=True)
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Vertical classification (AGE-11)
    vertical = models.CharField(
        max_length=50,
        choices=Vertical.choices(),
        blank=True,
        null=True,
    )

    # Link to project iteration (nullable for backward compatibility)
    iteration = models.OneToOneField(
        'projects.Iteration',
        on_delete=models.CASCADE,
        related_name='research_job',
        null=True,
        blank=True,
    )

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Research: {self.client_name} ({self.status})"


class ResearchReport(models.Model):
    """Structured deep research report for a client (AGE-10)."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    research_job = models.OneToOneField(
        ResearchJob,
        on_delete=models.CASCADE,
        related_name='report',
    )

    # Company overview
    company_overview = models.TextField(blank=True)
    founded_year = models.IntegerField(null=True, blank=True)
    headquarters = models.CharField(max_length=255, blank=True)
    employee_count = models.CharField(max_length=100, blank=True)
    annual_revenue = models.CharField(max_length=100, blank=True)
    website = models.URLField(blank=True)

    # Recent news and updates
    recent_news = models.JSONField(default=list, blank=True)

    # Decision makers
    decision_makers = models.JSONField(default=list, blank=True)

    # Pain points and opportunities
    pain_points = models.JSONField(default=list, blank=True)
    opportunities = models.JSONField(default=list, blank=True)

    # Digital and AI assessment
    digital_maturity = models.CharField(
        max_length=20,
        choices=DigitalMaturityLevel.choices(),
        blank=True,
    )
    ai_footprint = models.TextField(blank=True)
    ai_adoption_stage = models.CharField(
        max_length=20,
        choices=AIAdoptionStage.choices(),
        blank=True,
    )

    # Strategic information
    strategic_goals = models.JSONField(default=list, blank=True)
    key_initiatives = models.JSONField(default=list, blank=True)

    # Talking points for sales
    talking_points = models.JSONField(default=list, blank=True)

    # Web sources from Google Search grounding
    web_sources = models.JSONField(default=list, blank=True)

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Report: {self.research_job.client_name}"


class CompetitorCaseStudy(models.Model):
    """Competitor AI case studies found during research (AGE-12)."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    research_job = models.ForeignKey(
        ResearchJob,
        on_delete=models.CASCADE,
        related_name='competitor_case_studies',
    )

    competitor_name = models.CharField(max_length=255)
    vertical = models.CharField(
        max_length=50,
        choices=Vertical.choices(),
        blank=True,
    )
    case_study_title = models.CharField(max_length=500)
    summary = models.TextField()
    technologies_used = models.JSONField(default=list, blank=True)
    outcomes = models.JSONField(default=list, blank=True)
    source_url = models.URLField(blank=True)
    relevance_score = models.FloatField(default=0.0)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-relevance_score', '-created_at']
        verbose_name_plural = 'Competitor case studies'

    def __str__(self):
        return f"{self.competitor_name}: {self.case_study_title[:50]}"


class GapAnalysis(models.Model):
    """Technology/capability gap analysis from sales history (AGE-13)."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    research_job = models.OneToOneField(
        ResearchJob,
        on_delete=models.CASCADE,
        related_name='gap_analysis',
    )

    # Identified gaps
    technology_gaps = models.JSONField(default=list, blank=True)
    capability_gaps = models.JSONField(default=list, blank=True)
    process_gaps = models.JSONField(default=list, blank=True)

    # Recommendations
    recommendations = models.JSONField(default=list, blank=True)
    priority_areas = models.JSONField(default=list, blank=True)

    # Analysis metadata
    confidence_score = models.FloatField(default=0.0)
    analysis_notes = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'Gap analyses'

    def __str__(self):
        return f"Gap Analysis: {self.research_job.client_name}"


class InternalOpsIntel(models.Model):
    """Internal Operations Intelligence gathered from public sources (AGE-20)."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    research_job = models.OneToOneField(
        ResearchJob,
        on_delete=models.CASCADE,
        related_name='internal_ops',
    )

    # Employee Sentiment (from Glassdoor-like sources)
    # Structure: {overall_rating, work_life_balance, compensation, culture, management,
    #             recommend_pct, positive_themes, negative_themes, trend}
    employee_sentiment = models.JSONField(default=dict, blank=True)

    # LinkedIn Presence
    # Structure: {follower_count, engagement_level, recent_posts, employee_trend, notable_changes}
    linkedin_presence = models.JSONField(default=dict, blank=True)

    # Social Media Mentions
    # Structure: [{platform, summary, sentiment, topic}]
    social_media_mentions = models.JSONField(default=list, blank=True)

    # Job Postings Analysis
    # Structure: {total_openings, departments_hiring, skills_sought, seniority_dist,
    #             urgency_signals, insights}
    job_postings = models.JSONField(default=dict, blank=True)

    # News Sentiment
    # Structure: {overall_sentiment, coverage_volume, topics, headlines}
    news_sentiment = models.JSONField(default=dict, blank=True)

    # Synthesized Analysis
    key_insights = models.JSONField(default=list, blank=True)

    # Gap Correlations - cross-referenced with Gap Analysis
    # Structure: [{gap_type, description, evidence, evidence_type, confidence, sales_implication}]
    gap_correlations = models.JSONField(default=list, blank=True)

    # Metadata
    confidence_score = models.FloatField(default=0.0)
    data_freshness = models.CharField(max_length=50, blank=True)  # e.g., "last_30_days", "last_90_days"
    analysis_notes = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Internal Operations Intelligence'
        verbose_name_plural = 'Internal Operations Intelligence'

    def __str__(self):
        return f"Internal Ops Intel: {self.research_job.client_name}"
