"""Node functions for the research workflow."""
import logging
from django.conf import settings
from .state import ResearchState

logger = logging.getLogger(__name__)


def validate_input(state: ResearchState) -> ResearchState:
    """Validate the input data before starting research."""
    if not state.get('client_name'):
        return {
            **state,
            'status': 'failed',
            'error': 'Client name is required',
        }

    if not settings.GEMINI_API_KEY:
        return {
            **state,
            'status': 'failed',
            'error': 'Gemini API key is not configured',
        }

    return {
        **state,
        'status': 'researching',
    }


def conduct_research(state: ResearchState) -> ResearchState:
    """Conduct deep research using Gemini API with Google Search grounding (AGE-10)."""
    if state.get('status') == 'failed':
        return state

    try:
        from ..services.gemini import GeminiClient

        client = GeminiClient()

        # Conduct structured deep research with grounding
        report_data, grounding_metadata = client.conduct_deep_research(
            client_name=state.get('client_name', ''),
            sales_history=state.get('sales_history', ''),
        )

        # Convert to dict for state storage
        report_dict = report_data.to_dict()

        # Extract web sources from grounding metadata
        web_sources = []
        if grounding_metadata:
            web_sources = grounding_metadata.to_dict().get('web_sources', [])

        # Also generate plain text result for backward compatibility
        result_text = _format_research_result(report_dict)

        return {
            **state,
            'status': 'classifying',
            'result': result_text,
            'research_report': report_dict,
            'web_sources': web_sources,
        }

    except Exception as e:
        logger.exception("Error during research")
        return {
            **state,
            'status': 'failed',
            'error': str(e),
        }


def classify_vertical(state: ResearchState) -> ResearchState:
    """Classify the company into an industry vertical (AGE-11)."""
    if state.get('status') == 'failed':
        return state

    try:
        from ..services.gemini import GeminiClient
        from ..services.classifier import VerticalClassifier

        gemini_client = GeminiClient()
        classifier = VerticalClassifier(gemini_client=gemini_client)

        report = state.get('research_report', {})
        vertical = classifier.classify(
            client_name=state.get('client_name', ''),
            company_overview=report.get('company_overview', ''),
            use_llm=True,
        )

        return {
            **state,
            'status': 'competitor_search',
            'vertical': vertical,
        }

    except Exception as e:
        logger.exception("Error during vertical classification")
        # Non-fatal, continue with 'other' vertical
        return {
            **state,
            'status': 'competitor_search',
            'vertical': 'other',
        }


def search_competitors(state: ResearchState) -> ResearchState:
    """Search for competitor AI case studies (AGE-12)."""
    if state.get('status') == 'failed':
        return state

    try:
        from ..services.gemini import GeminiClient
        from ..services.competitor import CompetitorSearchService

        gemini_client = GeminiClient()
        competitor_service = CompetitorSearchService(gemini_client)

        report = state.get('research_report', {})
        case_studies = competitor_service.search_competitor_case_studies(
            client_name=state.get('client_name', ''),
            vertical=state.get('vertical', 'other'),
            company_overview=report.get('company_overview', ''),
        )

        # Convert to list of dicts for state storage
        case_studies_list = [
            {
                'competitor_name': cs.competitor_name,
                'vertical': cs.vertical,
                'case_study_title': cs.case_study_title,
                'summary': cs.summary,
                'technologies_used': cs.technologies_used,
                'outcomes': cs.outcomes,
                'source_url': cs.source_url,
                'relevance_score': cs.relevance_score,
            }
            for cs in case_studies
        ]

        return {
            **state,
            'status': 'gap_analysis',
            'competitor_case_studies': case_studies_list,
        }

    except Exception as e:
        logger.exception("Error during competitor search")
        # Non-fatal, continue without competitor data
        return {
            **state,
            'status': 'gap_analysis',
            'competitor_case_studies': [],
        }


def analyze_gaps(state: ResearchState) -> ResearchState:
    """Analyze technology and capability gaps (AGE-13)."""
    if state.get('status') == 'failed':
        return state

    try:
        from ..services.gemini import GeminiClient
        from ..services.gap_analysis import GapAnalysisService

        gemini_client = GeminiClient()
        gap_service = GapAnalysisService(gemini_client)

        report = state.get('research_report', {})
        gap_data = gap_service.analyze_gaps(
            client_name=state.get('client_name', ''),
            vertical=state.get('vertical', 'other'),
            company_overview=report.get('company_overview', ''),
            sales_history=state.get('sales_history', ''),
        )

        # Convert to dict for state storage
        gap_dict = {
            'technology_gaps': gap_data.technology_gaps,
            'capability_gaps': gap_data.capability_gaps,
            'process_gaps': gap_data.process_gaps,
            'recommendations': gap_data.recommendations,
            'priority_areas': gap_data.priority_areas,
            'confidence_score': gap_data.confidence_score,
            'analysis_notes': gap_data.analysis_notes,
        }

        return {
            **state,
            'status': 'completed',
            'gap_analysis': gap_dict,
        }

    except Exception as e:
        logger.exception("Error during gap analysis")
        # Non-fatal, continue without gap analysis
        return {
            **state,
            'status': 'completed',
            'gap_analysis': None,
        }


def research_internal_ops(state: ResearchState) -> ResearchState:
    """Research internal operations intelligence (AGE-20).

    This runs in parallel with the main research pipeline.
    Gathers employee sentiment, LinkedIn presence, job postings, etc.
    """
    if state.get('status') == 'failed':
        return state

    try:
        from ..services.gemini import GeminiClient
        from ..services.internal_ops import InternalOpsService

        gemini_client = GeminiClient()
        internal_ops_service = InternalOpsService(gemini_client)

        report = state.get('research_report', {})
        ops_data = internal_ops_service.research_internal_ops(
            client_name=state.get('client_name', ''),
            vertical=state.get('vertical', ''),
            website=report.get('website', ''),
            company_overview=report.get('company_overview', ''),
        )

        # Convert to dict for state storage
        ops_dict = ops_data.to_dict()

        return {
            **state,
            'internal_ops': ops_dict,
        }

    except Exception as e:
        logger.exception("Error during internal ops research")
        # Non-fatal, continue without internal ops data
        return {
            **state,
            'internal_ops': None,
        }


def correlate_internal_ops(state: ResearchState) -> ResearchState:
    """Correlate gap analysis findings with internal ops evidence (AGE-20)."""
    if state.get('status') == 'failed':
        return state

    gap_analysis = state.get('gap_analysis')
    internal_ops = state.get('internal_ops')

    # Skip correlation if either is missing
    if not gap_analysis or not internal_ops:
        logger.info("Skipping gap correlation - missing gap_analysis or internal_ops")
        return {
            **state,
            'status': 'completed',
            'gap_correlations': [],
        }

    try:
        from ..services.gemini import GeminiClient
        from ..services.gap_correlation import GapCorrelationService

        gemini_client = GeminiClient()
        correlation_service = GapCorrelationService(gemini_client)

        correlations = correlation_service.correlate_gaps(
            client_name=state.get('client_name', ''),
            vertical=state.get('vertical', 'other'),
            gap_analysis=gap_analysis,
            internal_ops=internal_ops,
        )

        # Convert to list of dicts
        correlations_list = correlation_service.correlations_to_dict(correlations)

        return {
            **state,
            'status': 'completed',
            'gap_correlations': correlations_list,
        }

    except Exception as e:
        logger.exception("Error during gap correlation")
        # Non-fatal, continue without correlations
        return {
            **state,
            'status': 'completed',
            'gap_correlations': [],
        }


def finalize_result(state: ResearchState) -> ResearchState:
    """Finalize the research result and persist to database."""
    if state.get('status') == 'failed':
        return state

    try:
        from ..models import ResearchJob, ResearchReport, CompetitorCaseStudy, GapAnalysis, InternalOpsIntel

        job_id = state.get('job_id')
        if not job_id:
            logger.warning("No job_id in state, skipping database persistence")
            return {**state, 'status': 'completed'}

        job = ResearchJob.objects.get(id=job_id)

        # Save vertical to job
        if state.get('vertical'):
            job.vertical = state['vertical']
            job.save(update_fields=['vertical'])

        # Create ResearchReport
        report_data = state.get('research_report', {})
        if report_data:
            # Clean founded_year - must be int or None
            founded_year = report_data.get('founded_year')
            if founded_year is not None:
                try:
                    founded_year = int(founded_year)
                except (ValueError, TypeError):
                    founded_year = None

            ResearchReport.objects.update_or_create(
                research_job=job,
                defaults={
                    'company_overview': report_data.get('company_overview', ''),
                    'founded_year': founded_year,
                    'headquarters': report_data.get('headquarters', ''),
                    'employee_count': report_data.get('employee_count', ''),
                    'annual_revenue': report_data.get('annual_revenue', ''),
                    'website': report_data.get('website', ''),
                    'recent_news': report_data.get('recent_news', []),
                    'decision_makers': report_data.get('decision_makers', []),
                    'pain_points': report_data.get('pain_points', []),
                    'opportunities': report_data.get('opportunities', []),
                    'digital_maturity': report_data.get('digital_maturity', ''),
                    'ai_footprint': report_data.get('ai_footprint', ''),
                    'ai_adoption_stage': report_data.get('ai_adoption_stage', ''),
                    'strategic_goals': report_data.get('strategic_goals', []),
                    'key_initiatives': report_data.get('key_initiatives', []),
                    'talking_points': report_data.get('talking_points', []),
                    'web_sources': state.get('web_sources', []),
                }
            )

        # Create CompetitorCaseStudy records
        case_studies = state.get('competitor_case_studies', [])
        for cs in case_studies:
            # Truncate fields to fit database column limits
            CompetitorCaseStudy.objects.create(
                research_job=job,
                competitor_name=cs.get('competitor_name', '')[:255],
                vertical=cs.get('vertical', '')[:50],
                case_study_title=cs.get('case_study_title', '')[:500],
                summary=cs.get('summary', ''),
                technologies_used=cs.get('technologies_used', []),
                outcomes=cs.get('outcomes', []),
                source_url=cs.get('source_url', '')[:500],
                relevance_score=cs.get('relevance_score', 0.0),
            )

        # Create GapAnalysis record
        gap_data = state.get('gap_analysis')
        if gap_data:
            GapAnalysis.objects.update_or_create(
                research_job=job,
                defaults={
                    'technology_gaps': gap_data.get('technology_gaps', []),
                    'capability_gaps': gap_data.get('capability_gaps', []),
                    'process_gaps': gap_data.get('process_gaps', []),
                    'recommendations': gap_data.get('recommendations', []),
                    'priority_areas': gap_data.get('priority_areas', []),
                    'confidence_score': gap_data.get('confidence_score', 0.0),
                    'analysis_notes': gap_data.get('analysis_notes', ''),
                }
            )

        # Create InternalOpsIntel record (AGE-20)
        internal_ops_data = state.get('internal_ops')
        if internal_ops_data:
            InternalOpsIntel.objects.update_or_create(
                research_job=job,
                defaults={
                    'employee_sentiment': internal_ops_data.get('employee_sentiment', {}),
                    'linkedin_presence': internal_ops_data.get('linkedin_presence', {}),
                    'social_media_mentions': internal_ops_data.get('social_media_mentions', []),
                    'job_postings': internal_ops_data.get('job_postings', {}),
                    'news_sentiment': internal_ops_data.get('news_sentiment', {}),
                    'key_insights': internal_ops_data.get('key_insights', []),
                    'gap_correlations': state.get('gap_correlations', []),
                    'confidence_score': internal_ops_data.get('confidence_score', 0.0),
                    'data_freshness': internal_ops_data.get('data_freshness', ''),
                    'analysis_notes': internal_ops_data.get('analysis_notes', ''),
                }
            )

        logger.info(f"Successfully finalized research job {job_id}")

        # Auto-capture to memory (AGE-17)
        try:
            from memory.services import MemoryCapture
            capture = MemoryCapture()
            capture_result = capture.capture_from_research(job)
            logger.info(f"Memory capture result: {capture_result}")
        except Exception as mem_error:
            logger.warning(f"Memory capture failed (non-fatal): {mem_error}")

    except Exception as e:
        logger.exception("Error during finalization")
        # Non-fatal for finalization, research is still complete

    return {
        **state,
        'status': 'completed',
    }


def _format_research_result(report: dict) -> str:
    """Format the research report as plain text for backward compatibility."""
    sections = []

    if report.get('company_overview'):
        sections.append(f"## Company Overview\n{report['company_overview']}")

    # Company details
    details = []
    if report.get('founded_year'):
        details.append(f"- Founded: {report['founded_year']}")
    if report.get('headquarters'):
        details.append(f"- Headquarters: {report['headquarters']}")
    if report.get('employee_count'):
        details.append(f"- Employees: {report['employee_count']}")
    if report.get('annual_revenue'):
        details.append(f"- Revenue: {report['annual_revenue']}")
    if report.get('website'):
        details.append(f"- Website: {report['website']}")
    if details:
        sections.append(f"## Company Details\n" + "\n".join(details))

    # Recent news
    if report.get('recent_news'):
        news_items = []
        for item in report['recent_news']:
            if isinstance(item, dict):
                news_items.append(f"- **{item.get('title', 'News')}**: {item.get('summary', '')}")
            else:
                news_items.append(f"- {item}")
        sections.append(f"## Recent News\n" + "\n".join(news_items))

    # Decision makers
    if report.get('decision_makers'):
        dm_items = []
        for dm in report['decision_makers']:
            if isinstance(dm, dict):
                dm_items.append(f"- **{dm.get('name', 'Unknown')}** - {dm.get('title', '')}: {dm.get('background', '')}")
            else:
                dm_items.append(f"- {dm}")
        sections.append(f"## Key Decision Makers\n" + "\n".join(dm_items))

    # Pain points
    if report.get('pain_points'):
        sections.append(f"## Pain Points\n" + "\n".join(f"- {p}" for p in report['pain_points']))

    # Opportunities
    if report.get('opportunities'):
        sections.append(f"## Opportunities\n" + "\n".join(f"- {o}" for o in report['opportunities']))

    # Digital maturity
    if report.get('digital_maturity') or report.get('ai_footprint'):
        dm_section = "## Digital & AI Assessment\n"
        if report.get('digital_maturity'):
            dm_section += f"- Digital Maturity: {report['digital_maturity'].title()}\n"
        if report.get('ai_adoption_stage'):
            dm_section += f"- AI Adoption Stage: {report['ai_adoption_stage'].title()}\n"
        if report.get('ai_footprint'):
            dm_section += f"- AI Footprint: {report['ai_footprint']}"
        sections.append(dm_section)

    # Strategic goals
    if report.get('strategic_goals'):
        sections.append(f"## Strategic Goals\n" + "\n".join(f"- {g}" for g in report['strategic_goals']))

    # Key initiatives
    if report.get('key_initiatives'):
        sections.append(f"## Key Initiatives\n" + "\n".join(f"- {i}" for i in report['key_initiatives']))

    # Talking points
    if report.get('talking_points'):
        sections.append(f"## Recommended Talking Points\n" + "\n".join(f"- {t}" for t in report['talking_points']))

    return "\n\n".join(sections)
