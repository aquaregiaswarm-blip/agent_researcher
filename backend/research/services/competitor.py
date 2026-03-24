"""Competitor search and case study extraction service (AGE-12)."""
import json
import logging
from typing import List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CompetitorCaseStudyData:
    """Data structure for a competitor case study."""
    competitor_name: str = ""
    vertical: str = ""
    case_study_title: str = ""
    summary: str = ""
    technologies_used: List[str] = field(default_factory=list)
    outcomes: List[str] = field(default_factory=list)
    source_url: str = ""
    relevance_score: float = 0.0


class CompetitorSearchService:
    """Service to find competitor AI case studies."""

    COMPETITOR_SEARCH_PROMPT = '''You are a competitive intelligence researcher. Find real, published case studies of AI and technology implementations at companies in the same industry as {client_name}. These will be used to create competitive urgency in sales conversations.

Target Company: {client_name}
Industry Vertical: {vertical}
Company Overview: {company_overview}

Answer these questions:

1. What companies compete directly with {client_name} or operate in the same industry segment? Identify 3-5 named competitors by searching for "{vertical} industry competitors" and "{client_name} competitors."
2. Have any of these competitors published AI, ML, automation, or digital transformation case studies? Search vendor case study libraries (AWS, Microsoft, Google Cloud, Snowflake, Databricks, CrowdStrike, ServiceNow, etc.) for case studies featuring companies in the {vertical} industry.
3. For each case study found: what was the business problem, the technology solution, the named vendors/platforms used, and the measurable outcomes (cost savings %, efficiency gains, revenue impact)?
4. Are there industry analyst reports (Gartner, Forrester, McKinsey, Deloitte) highlighting AI adoption trends in {vertical} that reference specific company examples?

Respond with valid JSON matching this structure:
{{
    "case_studies": [
        {{
            "competitor_name": "Company Name",
            "vertical": "industry vertical",
            "case_study_title": "Title of the case study or project",
            "summary": "2-3 sentence summary of what they did, why, and the outcome",
            "technologies_used": ["Named Technology 1", "Named Technology 2"],
            "outcomes": ["Specific measurable outcome 1", "Specific measurable outcome 2"],
            "source_url": "https://example.com/case-study",
            "relevance_score": 0.85
        }}
    ]
}}

RULES:
- Include 3-5 case studies. If fewer exist, return what you find rather than fabricating.
- relevance_score: 0.0-1.0 based on industry, company size, and problem similarity to {client_name}
- Every case study must have a real source_url. If no URL is available, use "" and note "Source URL not found" in the summary.
- Be specific about technologies and measurable outcomes — no vague claims.
- Respond ONLY with valid JSON
'''

    def __init__(self, gemini_client):
        """Initialize with a Gemini client."""
        self.gemini_client = gemini_client

    def search_competitor_case_studies(
        self,
        client_name: str,
        vertical: str,
        company_overview: str = "",
    ) -> tuple:
        """Search for competitor AI case studies using a grounded query.

        Args:
            client_name: Name of the target company
            vertical: Industry vertical
            company_overview: Description of the target company

        Returns:
            tuple: (List[CompetitorCaseStudyData], Optional[GroundingMetadata])
        """
        from .grounding import conduct_grounded_query

        prompt = self.COMPETITOR_SEARCH_PROMPT.format(
            client_name=client_name,
            vertical=vertical,
            company_overview=company_overview or "Not available",
        )

        try:
            result = conduct_grounded_query(
                self.gemini_client.client,
                prompt,
                'competitor_case_studies',
                self.gemini_client.MODEL_FLASH,
            )
        except Exception as e:
            logger.exception("Error calling grounded query for competitor search")
            return [], None

        grounding_metadata = result.grounding_metadata

        if not result.success or not result.text:
            logger.error(f"Competitor search failed: {result.error}")
            return [], grounding_metadata

        try:
            response_text = result.text.strip()

            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1])

            data = json.loads(response_text)
            case_studies = []

            for cs in data.get('case_studies', []):
                case_studies.append(CompetitorCaseStudyData(
                    competitor_name=cs.get('competitor_name', ''),
                    vertical=cs.get('vertical', ''),
                    case_study_title=cs.get('case_study_title', ''),
                    summary=cs.get('summary', ''),
                    technologies_used=cs.get('technologies_used', []),
                    outcomes=cs.get('outcomes', []),
                    source_url=cs.get('source_url', ''),
                    relevance_score=float(cs.get('relevance_score', 0.0)),
                ))

            return case_studies, grounding_metadata

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse competitor search response: {e}")
            return [], grounding_metadata
        except Exception as e:
            logger.exception("Error during competitor search")
            return [], grounding_metadata

    def create_case_study_models(
        self,
        research_job,
        case_studies: List[CompetitorCaseStudyData],
    ):
        """Create CompetitorCaseStudy model instances.

        Args:
            research_job: The ResearchJob instance
            case_studies: List of case study data objects

        Returns:
            List of created CompetitorCaseStudy instances
        """
        from ..models import CompetitorCaseStudy

        created = []
        for cs_data in case_studies:
            cs = CompetitorCaseStudy.objects.create(
                research_job=research_job,
                competitor_name=cs_data.competitor_name,
                vertical=cs_data.vertical,
                case_study_title=cs_data.case_study_title,
                summary=cs_data.summary,
                technologies_used=cs_data.technologies_used,
                outcomes=cs_data.outcomes,
                source_url=cs_data.source_url,
                relevance_score=cs_data.relevance_score,
            )
            created.append(cs)

        return created
