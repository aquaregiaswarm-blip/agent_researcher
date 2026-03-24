"""Use case generation service (AGE-18)."""
import json
import logging
from typing import List, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class UseCaseData:
    """Data structure for a generated use case."""
    title: str = ""
    description: str = ""
    business_problem: str = ""
    proposed_solution: str = ""
    expected_benefits: List[str] = field(default_factory=list)
    estimated_roi: str = ""
    time_to_value: str = ""
    technologies: List[str] = field(default_factory=list)
    data_requirements: List[str] = field(default_factory=list)
    integration_points: List[str] = field(default_factory=list)
    priority: str = "medium"
    impact_score: float = 0.0
    feasibility_score: float = 0.0


class UseCaseGenerator:
    """Service to generate AI use cases from research insights."""

    USE_CASE_PROMPT = '''You are an AI solutions architect generating use cases for a prospect on behalf of a value-added reseller and systems integrator.

Our capabilities include: cloud infrastructure and migration, cybersecurity and compliance, data and analytics platforms, AI/ML solutions, managed services and outsourced IT operations, networking and SD-WAN, and application modernization.

Only propose use cases that align with these capability areas. Do not propose use cases requiring proprietary SaaS product development, custom hardware, or capabilities outside our scope.

Based on the following research:
- Company: {client_name}
- Industry: {vertical}
- Overview: {company_overview}
- Pain Points: {pain_points}
- Opportunities: {opportunities}
- Digital Maturity: {digital_maturity}
- AI Adoption Stage: {ai_adoption_stage}
- Gaps: {gaps}

Generate 3-5 high-value AI/technology use cases that address their specific needs.

Respond with valid JSON:
{{
    "use_cases": [
        {{
            "title": "Use case title",
            "description": "Brief description",
            "business_problem": "Specific business problem this solves",
            "proposed_solution": "Overview of the AI/tech solution",
            "expected_benefits": ["Benefit 1", "Benefit 2"],
            "estimated_roi": "2-3x ROI within 12 months",
            "time_to_value": "3-6 months",
            "technologies": ["Technology 1", "Technology 2"],
            "data_requirements": ["Data need 1", "Data need 2"],
            "integration_points": ["System 1", "System 2"],
            "priority": "high|medium|low",
            "impact_score": 0.85,
            "feasibility_score": 0.75
        }}
    ]
}}

IMPORTANT:
- Generate 3-5 use cases prioritized by impact and feasibility
- Be specific to the company's industry and maturity level
- impact_score and feasibility_score should be 0.0-1.0
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

    def generate_use_cases(self, research_job) -> List[UseCaseData]:
        """Generate use cases from a completed research job.

        Args:
            research_job: ResearchJob model instance

        Returns:
            List of UseCaseData objects
        """
        report = getattr(research_job, 'report', None)
        gap_analysis = getattr(research_job, 'gap_analysis', None)

        # Build context from research
        context = {
            'client_name': research_job.client_name,
            'vertical': research_job.vertical or 'unknown',
            'company_overview': report.company_overview if report else '',
            'pain_points': ', '.join(report.pain_points[:5]) if report and report.pain_points else 'Not identified',
            'opportunities': ', '.join(report.opportunities[:5]) if report and report.opportunities else 'Not identified',
            'digital_maturity': report.digital_maturity if report else 'unknown',
            'ai_adoption_stage': report.ai_adoption_stage if report else 'unknown',
            'gaps': self._format_gaps(gap_analysis) if gap_analysis else 'Not analyzed',
        }

        prompt = self.USE_CASE_PROMPT.format(**context)

        try:
            response = self.gemini_client.generate_text(prompt)

            # Parse JSON response
            response_text = response.strip()
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1])

            data = json.loads(response_text)
            use_cases = []

            for uc in data.get('use_cases', []):
                use_cases.append(UseCaseData(
                    title=uc.get('title', ''),
                    description=uc.get('description', ''),
                    business_problem=uc.get('business_problem', ''),
                    proposed_solution=uc.get('proposed_solution', ''),
                    expected_benefits=uc.get('expected_benefits', []),
                    estimated_roi=uc.get('estimated_roi', ''),
                    time_to_value=uc.get('time_to_value', ''),
                    technologies=uc.get('technologies', []),
                    data_requirements=uc.get('data_requirements', []),
                    integration_points=uc.get('integration_points', []),
                    priority=uc.get('priority', 'medium'),
                    impact_score=float(uc.get('impact_score', 0.0)),
                    feasibility_score=float(uc.get('feasibility_score', 0.0)),
                ))

            return use_cases

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse use case generation response: {e}")
            return []
        except Exception as e:
            logger.exception("Error during use case generation")
            return []

    def _format_gaps(self, gap_analysis) -> str:
        """Format gap analysis for prompt."""
        parts = []
        if gap_analysis.technology_gaps:
            parts.append(f"Technology: {', '.join(gap_analysis.technology_gaps[:3])}")
        if gap_analysis.capability_gaps:
            parts.append(f"Capability: {', '.join(gap_analysis.capability_gaps[:3])}")
        if gap_analysis.process_gaps:
            parts.append(f"Process: {', '.join(gap_analysis.process_gaps[:3])}")
        return '; '.join(parts) or 'Not analyzed'

    def create_use_case_models(self, research_job, use_cases: List[UseCaseData]):
        """Create UseCase model instances.

        Args:
            research_job: The ResearchJob instance
            use_cases: List of UseCaseData objects

        Returns:
            List of created UseCase instances
        """
        from ..models import UseCase

        created = []
        for uc_data in use_cases:
            uc = UseCase.objects.create(
                research_job=research_job,
                title=uc_data.title,
                description=uc_data.description,
                business_problem=uc_data.business_problem,
                proposed_solution=uc_data.proposed_solution,
                expected_benefits=uc_data.expected_benefits,
                estimated_roi=uc_data.estimated_roi,
                time_to_value=uc_data.time_to_value,
                technologies=uc_data.technologies,
                data_requirements=uc_data.data_requirements,
                integration_points=uc_data.integration_points,
                priority=uc_data.priority,
                impact_score=uc_data.impact_score,
                feasibility_score=uc_data.feasibility_score,
            )
            created.append(uc)

        return created
