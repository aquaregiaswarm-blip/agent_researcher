"""Account plan generation service (AGE-23)."""
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict

logger = logging.getLogger(__name__)


@dataclass
class AccountPlanData:
    """Data structure for an account plan."""
    title: str = ""
    executive_summary: str = ""
    account_overview: str = ""
    strategic_objectives: List[str] = field(default_factory=list)
    key_stakeholders: List[Dict] = field(default_factory=list)
    opportunities: List[Dict] = field(default_factory=list)
    competitive_landscape: str = ""
    swot_analysis: Dict = field(default_factory=dict)
    engagement_strategy: str = ""
    value_propositions: List[str] = field(default_factory=list)
    action_plan: List[Dict] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)
    milestones: List[Dict] = field(default_factory=list)
    timeline: str = ""


class AccountPlanGenerator:
    """Service to generate strategic account plans."""

    ACCOUNT_PLAN_PROMPT = '''You are a strategic account planning expert.

Based on the following research:
- Company: {client_name}
- Industry: {vertical}
- Company Overview: {company_overview}
- Decision Makers: {decision_makers}
- Pain Points: {pain_points}
- Opportunities: {opportunities}
- Strategic Goals: {strategic_goals}
- Digital Maturity: {digital_maturity}
- Competitor Case Studies: {competitor_info}
- Gap Analysis: {gap_info}

Create a comprehensive strategic account plan.

Respond with valid JSON:
{{
    "title": "Account Plan: Company Name",
    "executive_summary": "Executive summary of the account strategy",
    "account_overview": "Overview of the account and relationship",
    "strategic_objectives": ["Objective 1", "Objective 2", "Objective 3"],
    "key_stakeholders": [
        {{"name": "Name", "title": "Title", "role_in_decision": "Champion/Influencer/Blocker", "engagement_approach": "How to engage"}}
    ],
    "opportunities": [
        {{"name": "Opportunity name", "value": "$100K-500K", "timeline": "Q3 2025", "probability": "60%"}}
    ],
    "competitive_landscape": "Analysis of competitive positioning",
    "swot_analysis": {{
        "strengths": ["Strength 1", "Strength 2"],
        "weaknesses": ["Weakness 1", "Weakness 2"],
        "opportunities": ["Opportunity 1", "Opportunity 2"],
        "threats": ["Threat 1", "Threat 2"]
    }},
    "engagement_strategy": "Overall engagement strategy",
    "value_propositions": ["Value prop 1", "Value prop 2"],
    "action_plan": [
        {{"action": "Action item", "owner": "Sales rep", "due_date": "Within 30 days", "status": "Not started"}}
    ],
    "success_metrics": ["Metric 1", "Metric 2"],
    "milestones": [
        {{"milestone": "Milestone name", "target_date": "Within 60 days", "criteria": "Success criteria"}}
    ],
    "timeline": "Overall timeline description"
}}

IMPORTANT:
- Be strategic and actionable
- Include specific stakeholder engagement tactics
- Provide realistic timelines
- Use realistic future dates relative to the current date, not hardcoded past dates
- Respond ONLY with valid JSON
'''

    def __init__(self, gemini_client=None):
        """Initialize with optional Gemini client."""
        self._gemini_client = gemini_client

    @property
    def gemini_client(self):
        """Lazy initialization of Gemini client."""
        if self._gemini_client is None:
            from research.services.gemini import GeminiClient
            self._gemini_client = GeminiClient()
        return self._gemini_client

    def generate_account_plan(self, research_job) -> AccountPlanData:
        """Generate an account plan from research.

        Args:
            research_job: ResearchJob model instance

        Returns:
            AccountPlanData object
        """
        report = getattr(research_job, 'report', None)
        gap_analysis = getattr(research_job, 'gap_analysis', None)

        # Format decision makers
        decision_makers = "Not identified"
        if report and report.decision_makers:
            dms = []
            for dm in report.decision_makers[:5]:
                if isinstance(dm, dict):
                    dms.append(f"{dm.get('name', '')} ({dm.get('title', '')})")
                else:
                    dms.append(str(dm))
            decision_makers = ', '.join(dms)

        # Format competitor info
        competitor_info = "Not analyzed"
        competitor_studies = research_job.competitor_case_studies.all()[:3]
        if competitor_studies:
            competitor_info = ', '.join([cs.competitor_name for cs in competitor_studies])

        # Format gap info
        gap_info = "Not analyzed"
        if gap_analysis:
            gaps = gap_analysis.priority_areas[:3] if gap_analysis.priority_areas else []
            gap_info = ', '.join(gaps) or "Not analyzed"

        context = {
            'client_name': research_job.client_name,
            'vertical': research_job.vertical or 'unknown',
            'company_overview': report.company_overview[:500] if report and report.company_overview else 'Not available',
            'decision_makers': decision_makers,
            'pain_points': ', '.join(report.pain_points[:5]) if report and report.pain_points else 'Not identified',
            'opportunities': ', '.join(report.opportunities[:5]) if report and report.opportunities else 'Not identified',
            'strategic_goals': ', '.join(report.strategic_goals[:5]) if report and report.strategic_goals else 'Not identified',
            'digital_maturity': report.digital_maturity if report else 'unknown',
            'competitor_info': competitor_info,
            'gap_info': gap_info,
        }

        prompt = self.ACCOUNT_PLAN_PROMPT.format(**context)

        try:
            response = self.gemini_client.generate_text(prompt)

            # Parse JSON response
            response_text = response.strip()
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1])

            data = json.loads(response_text)

            return AccountPlanData(
                title=data.get('title', ''),
                executive_summary=data.get('executive_summary', ''),
                account_overview=data.get('account_overview', ''),
                strategic_objectives=data.get('strategic_objectives', []),
                key_stakeholders=data.get('key_stakeholders', []),
                opportunities=data.get('opportunities', []),
                competitive_landscape=data.get('competitive_landscape', ''),
                swot_analysis=data.get('swot_analysis', {}),
                engagement_strategy=data.get('engagement_strategy', ''),
                value_propositions=data.get('value_propositions', []),
                action_plan=data.get('action_plan', []),
                success_metrics=data.get('success_metrics', []),
                milestones=data.get('milestones', []),
                timeline=data.get('timeline', ''),
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse account plan response: {e}")
            return AccountPlanData(title=f"Account Plan: {research_job.client_name}")
        except Exception as e:
            logger.exception("Error during account plan generation")
            return AccountPlanData(title=f"Account Plan: {research_job.client_name}")

    def create_account_plan_model(self, research_job, plan_data: AccountPlanData):
        """Create AccountPlan model instance.

        Args:
            research_job: The ResearchJob instance
            plan_data: AccountPlanData object

        Returns:
            Created AccountPlan instance
        """
        from ..models import AccountPlan

        plan, created = AccountPlan.objects.update_or_create(
            research_job=research_job,
            defaults={
                'title': plan_data.title,
                'executive_summary': plan_data.executive_summary,
                'account_overview': plan_data.account_overview,
                'strategic_objectives': plan_data.strategic_objectives,
                'key_stakeholders': plan_data.key_stakeholders,
                'opportunities': plan_data.opportunities,
                'competitive_landscape': plan_data.competitive_landscape,
                'swot_analysis': plan_data.swot_analysis,
                'engagement_strategy': plan_data.engagement_strategy,
                'value_propositions': plan_data.value_propositions,
                'action_plan': plan_data.action_plan,
                'success_metrics': plan_data.success_metrics,
                'milestones': plan_data.milestones,
                'timeline': plan_data.timeline,
            }
        )

        return plan
