"""Internal Operations Intelligence service (AGE-20).

Gathers and analyzes public signals about an organization's internal state
to provide sales teams with insight into internal challenges, culture,
and current priorities.
"""
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EmployeeSentiment:
    """Employee sentiment data structure."""
    overall_rating: float = 0.0
    work_life_balance: float = 0.0
    compensation: float = 0.0
    culture: float = 0.0
    management: float = 0.0
    recommend_pct: int = 0
    positive_themes: List[str] = field(default_factory=list)
    negative_themes: List[str] = field(default_factory=list)
    trend: str = "stable"  # improving, declining, stable

    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall_rating': self.overall_rating,
            'work_life_balance': self.work_life_balance,
            'compensation': self.compensation,
            'culture': self.culture,
            'management': self.management,
            'recommend_pct': self.recommend_pct,
            'positive_themes': self.positive_themes,
            'negative_themes': self.negative_themes,
            'trend': self.trend,
        }


@dataclass
class LinkedInPresence:
    """LinkedIn presence data structure."""
    follower_count: int = 0
    engagement_level: str = "medium"  # low, medium, high
    recent_posts: List[Dict[str, str]] = field(default_factory=list)  # [{title, summary, date}]
    employee_trend: str = "stable"  # growing, shrinking, stable
    notable_changes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'follower_count': self.follower_count,
            'engagement_level': self.engagement_level,
            'recent_posts': self.recent_posts,
            'employee_trend': self.employee_trend,
            'notable_changes': self.notable_changes,
        }


@dataclass
class SocialMediaMention:
    """Social media mention data structure."""
    platform: str = ""  # reddit, twitter, facebook
    summary: str = ""
    sentiment: str = "neutral"  # positive, negative, neutral, mixed
    topic: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'platform': self.platform,
            'summary': self.summary,
            'sentiment': self.sentiment,
            'topic': self.topic,
        }


@dataclass
class JobPostings:
    """Job postings analysis data structure."""
    total_openings: int = 0
    departments_hiring: Dict[str, int] = field(default_factory=dict)  # {dept_name: count}
    skills_sought: List[str] = field(default_factory=list)
    seniority_distribution: Dict[str, int] = field(default_factory=dict)  # {level: count}
    urgency_signals: List[str] = field(default_factory=list)  # e.g., "signing bonus", "immediate start"
    insights: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_openings': self.total_openings,
            'departments_hiring': self.departments_hiring,
            'skills_sought': self.skills_sought,
            'seniority_distribution': self.seniority_distribution,
            'urgency_signals': self.urgency_signals,
            'insights': self.insights,
        }


@dataclass
class NewsSentiment:
    """News sentiment data structure."""
    overall_sentiment: str = "neutral"  # positive, negative, neutral, mixed
    coverage_volume: str = "low"  # low, medium, high
    topics: List[str] = field(default_factory=list)
    headlines: List[Dict[str, str]] = field(default_factory=list)  # [{title, source, date, sentiment}]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall_sentiment': self.overall_sentiment,
            'coverage_volume': self.coverage_volume,
            'topics': self.topics,
            'headlines': self.headlines,
        }


@dataclass
class InternalOpsData:
    """Complete internal operations intelligence data."""
    employee_sentiment: EmployeeSentiment = field(default_factory=EmployeeSentiment)
    linkedin_presence: LinkedInPresence = field(default_factory=LinkedInPresence)
    social_media_mentions: List[SocialMediaMention] = field(default_factory=list)
    job_postings: JobPostings = field(default_factory=JobPostings)
    news_sentiment: NewsSentiment = field(default_factory=NewsSentiment)
    key_insights: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    data_freshness: str = "unknown"
    analysis_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'employee_sentiment': self.employee_sentiment.to_dict(),
            'linkedin_presence': self.linkedin_presence.to_dict(),
            'social_media_mentions': [m.to_dict() for m in self.social_media_mentions],
            'job_postings': self.job_postings.to_dict(),
            'news_sentiment': self.news_sentiment.to_dict(),
            'key_insights': self.key_insights,
            'confidence_score': self.confidence_score,
            'data_freshness': self.data_freshness,
            'analysis_notes': self.analysis_notes,
        }


class InternalOpsService:
    """Service to gather internal operations intelligence."""

    INTERNAL_OPS_JOB_POSTINGS_PROMPT = '''Research current job postings and hiring activity for {client_name} ({vertical}).

Search LinkedIn Jobs, Indeed, Glassdoor Jobs, and the company's own careers page at {website}.

Find:
- Number of open positions
- Which departments are hiring (Engineering, Sales, Marketing, Operations, etc.)
- Key technical skills and qualifications being sought
- Seniority levels being hired (Entry, Mid, Senior, Executive)
- Any urgency signals (sign-on bonuses, "immediate start", "hiring for growth")
- What the hiring patterns tell us about company priorities

Provide a detailed, factual summary based on publicly available job postings.'''

    INTERNAL_OPS_SENTIMENT_PROMPT = '''Research employee sentiment and workplace culture for {client_name}.

Search Glassdoor, Indeed company reviews, Blind, and other public employer review platforms.

Find:
- Overall employee rating (if available)
- Ratings for: work-life balance, compensation & benefits, culture & values, management
- Percentage of employees who would recommend the company
- Common positive themes in reviews (what employees appreciate)
- Common negative themes (challenges or frustrations)
- Whether sentiment is improving, declining, or stable over time
- Notable leadership or management feedback

Provide a factual summary based on publicly available employee reviews.'''

    INTERNAL_OPS_NEWS_SOCIAL_PROMPT = '''Research recent news, press releases, and social media activity for {client_name}.

Search news aggregators, company press releases, Reddit, Twitter/X, and LinkedIn posts.

Find:
- Company LinkedIn follower count and recent company posts
- Employee headcount trend (growing/shrinking/stable)
- Notable leadership changes or key hires announced
- Recent news coverage topics and overall sentiment
- Relevant Reddit discussions about working at or partnering with {client_name}
- Twitter/X mentions about the company's culture, products, or strategy

Provide a factual summary based on publicly available information.'''

    INTERNAL_OPS_SYNTHESIS_PROMPT = '''You are a business intelligence analyst. Based on the following research about {client_name}, create a comprehensive internal operations intelligence report for a sales team.

## Job Postings & Hiring Research:
{job_postings_research}

## Employee Sentiment Research:
{sentiment_research}

## News & Social Media Research:
{news_social_research}

Synthesize this information into a structured JSON report. Respond ONLY with valid JSON:
{{
    "employee_sentiment": {{
        "overall_rating": 3.8,
        "work_life_balance": 3.5,
        "compensation": 3.7,
        "culture": 3.6,
        "management": 3.4,
        "recommend_pct": 68,
        "positive_themes": ["Good benefits", "Smart colleagues", "Interesting work"],
        "negative_themes": ["Long hours", "Bureaucracy", "Slow promotions"],
        "trend": "stable"
    }},
    "linkedin_presence": {{
        "follower_count": 50000,
        "engagement_level": "medium",
        "recent_posts": [
            {{"title": "Post title", "summary": "Brief summary", "date": "2024-01"}}
        ],
        "employee_trend": "growing",
        "notable_changes": ["New CTO hired", "Expanded engineering team"]
    }},
    "social_media_mentions": [
        {{
            "platform": "reddit",
            "summary": "Discussion summary on r/tech",
            "sentiment": "mixed",
            "topic": "Work culture"
        }},
        {{
            "platform": "twitter",
            "summary": "Summary of Twitter discussions",
            "sentiment": "neutral",
            "topic": "Product announcements"
        }}
    ],
    "job_postings": {{
        "total_openings": 45,
        "departments_hiring": {{
            "Engineering": 20,
            "Sales": 10,
            "Marketing": 8,
            "Operations": 7
        }},
        "skills_sought": ["Python", "Cloud", "AI/ML", "Leadership"],
        "seniority_distribution": {{
            "Entry": 10,
            "Mid": 20,
            "Senior": 12,
            "Executive": 3
        }},
        "urgency_signals": ["Competitive salary", "Sign-on bonus for engineering roles"],
        "insights": "Heavy focus on technical hiring suggests major development initiative"
    }},
    "news_sentiment": {{
        "overall_sentiment": "positive",
        "coverage_volume": "medium",
        "topics": ["Product launch", "Funding round", "Industry recognition"],
        "headlines": [
            {{"title": "Company Announces New Product", "source": "TechCrunch", "date": "2024-01", "sentiment": "positive"}}
        ]
    }},
    "key_insights": [
        "Heavy engineering hiring suggests major product development underway",
        "Employee reviews indicate culture challenges that may create opportunity for solutions",
        "Recent leadership changes may mean new strategic priorities"
    ],
    "confidence_score": 0.75,
    "data_freshness": "last_30_days",
    "analysis_notes": "Analysis based on publicly available information from LinkedIn, job boards, and news sources."
}}

IMPORTANT:
- Base analysis on information from the research provided above
- Rate scales: 1.0-5.0 for sentiment ratings, 0-100 for percentages
- confidence_score: 0.0-1.0 based on information availability
- data_freshness options: "last_7_days", "last_30_days", "last_90_days", "older"
- trend options: "improving", "declining", "stable"
- engagement_level: "low", "medium", "high"
- employee_trend: "growing", "shrinking", "stable"
- overall_sentiment: "positive", "negative", "neutral", "mixed"
- Respond ONLY with valid JSON
'''

    def __init__(self, gemini_client):
        """Initialize with a Gemini client."""
        self.gemini_client = gemini_client

    def research_internal_ops(
        self,
        client_name: str,
        vertical: str = "",
        website: str = "",
        company_overview: str = "",
    ) -> tuple:
        """Research internal operations intelligence using parallel grounded queries.

        Runs 3 parallel grounded queries (job postings, sentiment, news/social),
        then synthesises into structured InternalOpsData.

        Args:
            client_name: Name of the target company
            vertical: Industry vertical
            website: Company website URL
            company_overview: Brief company description

        Returns:
            tuple: (InternalOpsData, Optional[GroundingMetadata])
        """
        from .grounding import conduct_grounded_query, merge_grounding_metadata
        from .gemini import GroundedQueryResult

        genai_client = self.gemini_client.client
        model = self.gemini_client.MODEL_FLASH

        queries = {
            'job_postings': self.INTERNAL_OPS_JOB_POSTINGS_PROMPT.format(
                client_name=client_name,
                vertical=vertical or "Not specified",
                website=website or "Not available",
            ),
            'sentiment': self.INTERNAL_OPS_SENTIMENT_PROMPT.format(
                client_name=client_name,
            ),
            'news_social': self.INTERNAL_OPS_NEWS_SOCIAL_PROMPT.format(
                client_name=client_name,
            ),
        }

        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_type = {
                executor.submit(conduct_grounded_query, genai_client, prompt, query_type, model): query_type
                for query_type, prompt in queries.items()
            }
            for future in as_completed(future_to_type):
                query_type = future_to_type[future]
                try:
                    results[query_type] = future.result()
                except Exception as e:
                    logger.error(f"Internal ops query '{query_type}' raised exception: {e}")
                    results[query_type] = GroundedQueryResult(
                        query_type=query_type,
                        text="",
                        success=False,
                        error=str(e),
                    )

        merged_metadata = merge_grounding_metadata(results)

        _empty = GroundedQueryResult(query_type='')
        job_postings_text = results.get('job_postings', _empty).text or "No data available."
        sentiment_text = results.get('sentiment', _empty).text or "No data available."
        news_social_text = results.get('news_social', _empty).text or "No data available."

        synthesis_prompt = self.INTERNAL_OPS_SYNTHESIS_PROMPT.format(
            client_name=client_name,
            job_postings_research=job_postings_text,
            sentiment_research=sentiment_text,
            news_social_research=news_social_text,
        )

        try:
            response = self.gemini_client.generate_text(synthesis_prompt)
            response_text = response.strip()

            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1])

            data = json.loads(response_text)
            ops_data = self._parse_ops_data(data)
            return ops_data, merged_metadata

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse internal ops synthesis: {e}")
            return InternalOpsData(analysis_notes=f"Analysis parsing failed: {str(e)}"), merged_metadata
        except Exception as e:
            logger.exception("Error during internal ops synthesis")
            return InternalOpsData(analysis_notes=f"Research failed: {str(e)}"), merged_metadata

    def _parse_ops_data(self, data: dict) -> InternalOpsData:
        """Parse raw JSON dict into InternalOpsData."""
        es_data = data.get('employee_sentiment', {})
        employee_sentiment = EmployeeSentiment(
            overall_rating=float(es_data.get('overall_rating', 0.0)),
            work_life_balance=float(es_data.get('work_life_balance', 0.0)),
            compensation=float(es_data.get('compensation', 0.0)),
            culture=float(es_data.get('culture', 0.0)),
            management=float(es_data.get('management', 0.0)),
            recommend_pct=int(es_data.get('recommend_pct', 0)),
            positive_themes=es_data.get('positive_themes', []),
            negative_themes=es_data.get('negative_themes', []),
            trend=es_data.get('trend', 'stable'),
        )

        li_data = data.get('linkedin_presence', {})
        linkedin_presence = LinkedInPresence(
            follower_count=int(li_data.get('follower_count', 0)),
            engagement_level=li_data.get('engagement_level', 'medium'),
            recent_posts=li_data.get('recent_posts', []),
            employee_trend=li_data.get('employee_trend', 'stable'),
            notable_changes=li_data.get('notable_changes', []),
        )

        social_media_mentions = [
            SocialMediaMention(
                platform=sm.get('platform', ''),
                summary=sm.get('summary', ''),
                sentiment=sm.get('sentiment', 'neutral'),
                topic=sm.get('topic', ''),
            )
            for sm in data.get('social_media_mentions', [])
        ]

        jp_data = data.get('job_postings', {})
        job_postings = JobPostings(
            total_openings=int(jp_data.get('total_openings', 0)),
            departments_hiring=jp_data.get('departments_hiring', {}),
            skills_sought=jp_data.get('skills_sought', []),
            seniority_distribution=jp_data.get('seniority_distribution', {}),
            urgency_signals=jp_data.get('urgency_signals', []),
            insights=jp_data.get('insights', ''),
        )

        ns_data = data.get('news_sentiment', {})
        news_sentiment = NewsSentiment(
            overall_sentiment=ns_data.get('overall_sentiment', 'neutral'),
            coverage_volume=ns_data.get('coverage_volume', 'low'),
            topics=ns_data.get('topics', []),
            headlines=ns_data.get('headlines', []),
        )

        return InternalOpsData(
            employee_sentiment=employee_sentiment,
            linkedin_presence=linkedin_presence,
            social_media_mentions=social_media_mentions,
            job_postings=job_postings,
            news_sentiment=news_sentiment,
            key_insights=data.get('key_insights', []),
            confidence_score=float(data.get('confidence_score', 0.0)),
            data_freshness=data.get('data_freshness', 'unknown'),
            analysis_notes=data.get('analysis_notes', ''),
        )

    def create_internal_ops_model(
        self,
        research_job,
        ops_data: InternalOpsData,
        gap_correlations: Optional[List[Dict[str, Any]]] = None,
    ):
        """Create InternalOpsIntel model instance.

        Args:
            research_job: The ResearchJob instance
            ops_data: InternalOpsData object
            gap_correlations: Optional list of gap correlation data

        Returns:
            Created InternalOpsIntel instance
        """
        from ..models import InternalOpsIntel

        internal_ops, created = InternalOpsIntel.objects.update_or_create(
            research_job=research_job,
            defaults={
                'employee_sentiment': ops_data.employee_sentiment.to_dict(),
                'linkedin_presence': ops_data.linkedin_presence.to_dict(),
                'social_media_mentions': [m.to_dict() for m in ops_data.social_media_mentions],
                'job_postings': ops_data.job_postings.to_dict(),
                'news_sentiment': ops_data.news_sentiment.to_dict(),
                'key_insights': ops_data.key_insights,
                'gap_correlations': gap_correlations or [],
                'confidence_score': ops_data.confidence_score,
                'data_freshness': ops_data.data_freshness,
                'analysis_notes': ops_data.analysis_notes,
            }
        )

        return internal_ops
