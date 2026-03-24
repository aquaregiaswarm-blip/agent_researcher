"""Gap analysis service for sales history (AGE-13)."""
import json
import re
import logging
from typing import List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GapAnalysisData:
    """Data structure for gap analysis results."""
    technology_gaps: List[str] = field(default_factory=list)
    capability_gaps: List[str] = field(default_factory=list)
    process_gaps: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    priority_areas: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    analysis_notes: str = ""


class GapAnalysisService:
    """Service to analyze gaps from sales history."""

    GAP_ANALYSIS_PROMPT = '''You are a sales strategy analyst. Analyze the company information and sales history to identify technology, capability, and process gaps.

Target Company: {client_name}
Industry Vertical: {vertical}
Company Overview: {company_overview}

## Research Directive (from submitter):
{prompt}

Weight your gap analysis toward themes flagged above. If the submitter identified specific technology areas or competitive concerns, ensure those are reflected in your gap identification.

Sales History:
{sales_history}

Existing Pain Points:
{pain_points}

Identified Opportunities:
{opportunities}

Strategic Goals:
{strategic_goals}

Key Initiatives:
{key_initiatives}

Digital Maturity: {digital_maturity}
AI Footprint: {ai_footprint}
AI Adoption Stage: {ai_adoption_stage}

Competitor Case Studies (what similar companies have done):
{competitor_case_studies}

Analyze this information to identify:
1. Technology gaps - missing or outdated technologies
2. Capability gaps - skills or competencies they lack
3. Process gaps - inefficient or missing business processes
4. Recommendations - specific solutions or approaches
5. Priority areas - where to focus first

Respond with valid JSON matching this structure:
{{
    "technology_gaps": [
        "Gap 1: Description of technology gap and its business impact",
        "Gap 2: Another technology gap"
    ],
    "capability_gaps": [
        "Gap 1: Description of capability/skill gap",
        "Gap 2: Another capability gap"
    ],
    "process_gaps": [
        "Gap 1: Description of process or workflow gap",
        "Gap 2: Another process gap"
    ],
    "recommendations": [
        "Recommendation 1: Specific, actionable recommendation",
        "Recommendation 2: Another recommendation"
    ],
    "priority_areas": [
        "Priority 1: Highest priority area with rationale",
        "Priority 2: Second priority area"
    ],
    "confidence_score": 0.75,
    "analysis_notes": "Summary of analysis methodology and key findings"
}}

IMPORTANT:
- Include 3-5 items for each gap category
- Be specific about business impact
- Prioritize based on potential value and feasibility
- confidence_score should be 0.0-1.0 based on how much information was available
- If sales history is minimal, focus on industry-typical gaps
- Respond ONLY with valid JSON
'''

    def __init__(self, gemini_client):
        """Initialize with a Gemini client."""
        self.gemini_client = gemini_client

    def analyze_gaps(
        self,
        client_name: str,
        vertical: str,
        company_overview: str = "",
        sales_history: str = "",
        prompt: str = "",
        pain_points: list = None,
        opportunities: list = None,
        strategic_goals: list = None,
        key_initiatives: list = None,
        digital_maturity: str = "",
        ai_footprint: str = "",
        ai_adoption_stage: str = "",
        competitor_case_studies: list = None,
    ) -> GapAnalysisData:
        """Analyze gaps from sales history and enriched company context.

        Args:
            client_name: Name of the target company
            vertical: Industry vertical
            company_overview: Description of the company
            sales_history: Historical sales interaction data
            pain_points: Known pain points from research report
            opportunities: Known opportunities from research report
            strategic_goals: Strategic goals from research report
            key_initiatives: Key initiatives from research report
            digital_maturity: Digital maturity level
            ai_footprint: AI usage description
            ai_adoption_stage: AI adoption stage
            competitor_case_studies: Competitor case studies for context

        Returns:
            GapAnalysisData object with identified gaps
        """
        def fmt_list(items):
            return "\n".join(f"- {i}" for i in (items or [])) or "None identified"

        def fmt_case_studies(studies):
            if not studies:
                return "None available"
            summaries = []
            for cs in (studies or []):
                if isinstance(cs, dict):
                    name = cs.get('competitor_name', 'Unknown')
                    title = cs.get('case_study_title', '')
                    summary = cs.get('summary', '')
                    summaries.append(f"- {name}: {title} — {summary}")
            return "\n".join(summaries) if summaries else "None available"

        prompt_value = prompt.strip() if prompt and prompt.strip() else "No specific directive provided."
        formatted_prompt = self.GAP_ANALYSIS_PROMPT.format(
            client_name=client_name,
            vertical=vertical,
            company_overview=company_overview or "Not available",
            prompt=prompt_value,
            sales_history=sales_history or "No sales history provided",
            pain_points=fmt_list(pain_points),
            opportunities=fmt_list(opportunities),
            strategic_goals=fmt_list(strategic_goals),
            key_initiatives=fmt_list(key_initiatives),
            digital_maturity=digital_maturity or "Unknown",
            ai_footprint=ai_footprint or "Unknown",
            ai_adoption_stage=ai_adoption_stage or "Unknown",
            competitor_case_studies=fmt_case_studies(competitor_case_studies),
        )

        try:
            response = self.gemini_client.generate_text(formatted_prompt)

            # Parse JSON response
            response_text = response.strip()

            # Strip markdown code fences robustly (handles ```json, trailing newlines, etc.)
            if '```' in response_text:
                match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response_text, re.DOTALL)
                if match:
                    response_text = match.group(1).strip()

            data = json.loads(response_text)

            return GapAnalysisData(
                technology_gaps=data.get('technology_gaps', []),
                capability_gaps=data.get('capability_gaps', []),
                process_gaps=data.get('process_gaps', []),
                recommendations=data.get('recommendations', []),
                priority_areas=data.get('priority_areas', []),
                confidence_score=float(data.get('confidence_score', 0.0)),
                analysis_notes=data.get('analysis_notes', ''),
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse gap analysis response: {e}")
            return GapAnalysisData(
                analysis_notes=f"Analysis parsing failed. Raw output: {response}"
            )
        except Exception as e:
            logger.exception("Error during gap analysis")
            return GapAnalysisData(
                analysis_notes=f"Analysis failed: {str(e)}"
            )

    def create_gap_analysis_model(
        self,
        research_job,
        gap_data: GapAnalysisData,
    ):
        """Create GapAnalysis model instance.

        Args:
            research_job: The ResearchJob instance
            gap_data: GapAnalysisData object

        Returns:
            Created GapAnalysis instance
        """
        from ..models import GapAnalysis

        gap_analysis, created = GapAnalysis.objects.update_or_create(
            research_job=research_job,
            defaults={
                'technology_gaps': gap_data.technology_gaps,
                'capability_gaps': gap_data.capability_gaps,
                'process_gaps': gap_data.process_gaps,
                'recommendations': gap_data.recommendations,
                'priority_areas': gap_data.priority_areas,
                'confidence_score': gap_data.confidence_score,
                'analysis_notes': gap_data.analysis_notes,
            }
        )

        return gap_analysis
