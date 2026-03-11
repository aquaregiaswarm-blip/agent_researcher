"""Gemini API client for deep research."""
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from dataclasses import dataclass, field, asdict
from django.conf import settings

logger = logging.getLogger(__name__)


@dataclass
class WebSource:
    """A web source from grounding metadata."""
    uri: str = ""
    title: str = ""


@dataclass
class GroundingMetadata:
    """Grounding metadata from Gemini response."""
    web_sources: list = field(default_factory=list)
    search_queries: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'web_sources': [asdict(ws) if isinstance(ws, WebSource) else ws for ws in self.web_sources],
            'search_queries': self.search_queries,
        }


@dataclass
class GroundedQueryResult:
    """Result from a single grounded query."""
    query_type: str
    text: str = ""
    grounding_metadata: Optional['GroundingMetadata'] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class DecisionMaker:
    """A key decision maker at the company."""
    name: str = ""
    title: str = ""
    background: str = ""
    linkedin_url: str = ""


@dataclass
class NewsItem:
    """A recent news item about the company."""
    title: str = ""
    summary: str = ""
    date: str = ""
    source: str = ""
    url: str = ""


@dataclass
class ResearchReportData:
    """Structured data from deep research."""
    # Company overview
    company_overview: str = ""
    founded_year: Optional[int] = None
    headquarters: str = ""
    employee_count: str = ""
    annual_revenue: str = ""
    website: str = ""

    # Recent news
    recent_news: list = field(default_factory=list)

    # Decision makers
    decision_makers: list = field(default_factory=list)

    # Pain points and opportunities
    pain_points: list = field(default_factory=list)
    opportunities: list = field(default_factory=list)

    # Digital and AI assessment
    digital_maturity: str = ""
    ai_footprint: str = ""
    ai_adoption_stage: str = ""

    # Strategic information
    strategic_goals: list = field(default_factory=list)
    key_initiatives: list = field(default_factory=list)

    # Talking points
    talking_points: list = field(default_factory=list)

    # Cloud, security and data intelligence
    cloud_footprint: str = ""
    security_posture: str = ""
    data_maturity: str = ""
    financial_signals: list = field(default_factory=list)
    tech_partnerships: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for model storage."""
        return asdict(self)


class GeminiClient:
    """Client for Gemini API with structured output parsing."""

    MODEL_PRO = 'gemini-2.5-pro-preview-06-05'
    MODEL_FLASH = 'gemini-2.5-flash'

    # Phase 1 — grounded query prompts

    PROFILE_QUERY_PROMPT = '''Research basic profile for {client_name}: company overview, headquarters, employee count, revenue, founding year, website. Provide comprehensive factual information.'''

    NEWS_QUERY_PROMPT = '''Find recent news and developments about {client_name}. Include headlines, dates, sources, and brief summaries of each news item. Focus on the most recent and significant news.'''

    LEADERSHIP_QUERY_PROMPT = '''Research key executives and decision makers at {client_name}. Include names, titles, professional backgrounds, and any notable achievements or public statements.'''

    TECHNOLOGY_QUERY_PROMPT = '''Research {client_name}'s technology stack, digital maturity, AI adoption, and key technical initiatives. Include information about their digital transformation efforts and any AI/ML implementations.'''

    CLOUD_INFRASTRUCTURE_QUERY_PROMPT = '''Research {client_name}'s cloud and infrastructure strategy. Find: cloud provider partnerships (AWS, Azure, GCP), migration announcements, data centre consolidation or expansion plans, hybrid/multi-cloud strategy, cloud-native vs legacy infrastructure signals, and any infrastructure vendor partnerships.'''

    CYBERSECURITY_COMPLIANCE_QUERY_PROMPT = '''Research {client_name}'s cybersecurity posture and compliance profile. Find: any publicly disclosed data breaches or security incidents, CISO hiring activity or leadership changes, compliance certifications (SOC2, ISO27001, HIPAA, FedRAMP, PCI-DSS), cybersecurity vendor partnerships, security investment announcements, and regulatory compliance pressures in their industry.'''

    DATA_ANALYTICS_QUERY_PROMPT = '''Research {client_name}'s data and analytics maturity. Find: data warehouse or data lake platforms in use (Snowflake, Databricks, BigQuery, Redshift), BI tooling (Tableau, Power BI, Looker), data governance initiatives, chief data officer or data leadership hiring, analytics vendor partnerships, and any public mentions of data strategy or data-driven transformation.'''

    FINANCIAL_FILINGS_QUERY_PROMPT = '''Research {client_name}'s financial filings and investor communications for technology signals. Find: annual report or 10-K technology investment themes, risk factors related to technology or cybersecurity, earnings call mentions of AI, digital transformation or data initiatives, capital expenditure allocation signals, and analyst commentary on their technology strategy.'''

    SYNTHESIS_PROMPT = '''You are a deep research assistant. Synthesize the following research into a comprehensive company profile for a sales team preparing to engage with {client_name}.

## Sales Context (prioritise these themes in your analysis):
{sales_history}

## Company Profile:
{profile}

## Financial & Investor Signals:
{financial_filings}

## Recent News:
{news}

## Leadership Information:
{leadership}

## Technology Assessment:
{technology}

## Cloud & Infrastructure:
{cloud_infrastructure}

## Data & Analytics:
{data_analytics}

## Cybersecurity & Compliance:
{cybersecurity_compliance}

Create a cohesive analysis that covers:
1. Company overview and key facts
2. Leadership and decision-makers
3. Recent developments and news
4. Digital maturity and AI adoption status
5. Cloud infrastructure footprint
6. Security posture and compliance
7. Data and analytics maturity
8. Financial signals and technology investment themes
9. Pain points they may face
10. Opportunities for AI/technology solutions
11. Strategic goals and initiatives
12. Technology partnerships and vendor relationships
13. Recommended talking points for sales conversations (informed by the sales context above)

Be comprehensive and actionable. Include specific details and cite facts from the research.'''

    JSON_FORMAT_PROMPT = '''Convert the following research into a structured JSON format. Your response MUST be valid JSON matching this exact structure:

## Research to Format:
{research_text}

Required JSON Structure:
{{
    "company_overview": "Comprehensive overview of the company",
    "founded_year": 2000,
    "headquarters": "City, State/Country",
    "employee_count": "1,000-5,000",
    "annual_revenue": "$500M - $1B",
    "website": "https://example.com",
    "recent_news": [
        {{
            "title": "News headline",
            "summary": "Brief summary",
            "date": "2024-01-15",
            "source": "Source name",
            "url": "https://source.com/article"
        }}
    ],
    "decision_makers": [
        {{
            "name": "Full Name",
            "title": "Job Title",
            "background": "Brief background",
            "linkedin_url": ""
        }}
    ],
    "pain_points": ["Pain point 1", "Pain point 2"],
    "opportunities": ["Opportunity 1", "Opportunity 2"],
    "digital_maturity": "nascent|developing|maturing|advanced|leading",
    "ai_footprint": "Description of AI usage",
    "ai_adoption_stage": "exploring|experimenting|implementing|scaling|optimizing",
    "strategic_goals": ["Goal 1", "Goal 2"],
    "key_initiatives": ["Initiative 1", "Initiative 2"],
    "talking_points": ["Talking point 1", "Talking point 2"],
    "cloud_footprint": "Description of cloud provider footprint (AWS/Azure/GCP partnerships, data centre strategy, hybrid or cloud-native signals)",
    "security_posture": "Description of security posture, compliance certifications held, recent incidents or investments, CISO signals",
    "data_maturity": "Description of data maturity level, BI tooling in use, data governance initiatives, analytics capabilities",
    "financial_signals": ["Technology capex signal from annual report", "AI mention from earnings call"],
    "tech_partnerships": ["AWS Premier Partner", "Snowflake Select Partner"]
}}

IMPORTANT: Respond ONLY with valid JSON, no additional text or markdown formatting.'''

    VERTICAL_CLASSIFICATION_PROMPT = '''Based on the following company information, classify the company into one of these industry verticals:

Company: {client_name}
Overview: {company_overview}

Available verticals:
- healthcare: Healthcare, pharmaceuticals, medical devices, health services
- finance: Banking, insurance, investment, fintech
- retail: Retail, e-commerce, consumer goods
- manufacturing: Manufacturing, industrial, production
- technology: Software, IT services, tech products
- energy: Oil & gas, utilities, renewable energy
- telecommunications: Telecom, network services
- media_entertainment: Media, entertainment, gaming, publishing
- transportation: Logistics, shipping, airlines, automotive
- real_estate: Real estate, property management
- professional_services: Consulting, legal, accounting
- education: Education, EdTech, training
- government: Government, public sector
- hospitality: Hotels, restaurants, travel
- agriculture: Agriculture, food production
- construction: Construction, engineering
- nonprofit: Non-profit organizations
- other: Other industries

Respond with ONLY the vertical name (e.g., "healthcare" or "finance"), nothing else.
'''

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gemini client."""
        self.api_key = api_key or settings.GEMINI_API_KEY
        self._client = None

    @property
    def client(self):
        """Lazy initialization of the Gemini client."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _extract_grounding_metadata(self, response) -> Optional[GroundingMetadata]:
        """Extract grounding metadata (web sources) from Gemini response."""
        try:
            if not response.candidates:
                return None

            candidate = response.candidates[0]
            grounding_metadata = getattr(candidate, 'grounding_metadata', None)
            if not grounding_metadata:
                return None

            grounding_chunks = getattr(grounding_metadata, 'grounding_chunks', [])
            if not grounding_chunks:
                # Still return metadata if we have search queries
                search_queries = getattr(grounding_metadata, 'web_search_queries', []) or []
                if search_queries:
                    return GroundingMetadata(web_sources=[], search_queries=list(search_queries))
                return None

            web_sources = []
            for chunk in grounding_chunks:
                web = getattr(chunk, 'web', None)
                if web:
                    uri = getattr(web, 'uri', None) or ""
                    title = getattr(web, 'title', None) or ""
                    web_sources.append(WebSource(uri=uri, title=title))

            search_queries = getattr(grounding_metadata, 'web_search_queries', []) or []

            # Return metadata if we have either sources or queries
            if web_sources or search_queries:
                return GroundingMetadata(
                    web_sources=web_sources,
                    search_queries=list(search_queries),
                )
            return None

        except Exception as e:
            logger.warning(f"Failed to extract grounding metadata: {e}")
            return None

    def _conduct_grounded_query(self, prompt: str, query_type: str) -> GroundedQueryResult:
        """Make a single grounded query with Google Search enabled."""
        from .grounding import conduct_grounded_query
        return conduct_grounded_query(self.client, prompt, query_type, self.MODEL_FLASH)

    def _run_parallel_grounded_queries(self, client_name: str) -> dict:
        """Run 8 grounded queries in parallel using ThreadPoolExecutor."""
        queries = {
            'profile': self.PROFILE_QUERY_PROMPT.format(client_name=client_name),
            'news': self.NEWS_QUERY_PROMPT.format(client_name=client_name),
            'leadership': self.LEADERSHIP_QUERY_PROMPT.format(client_name=client_name),
            'technology': self.TECHNOLOGY_QUERY_PROMPT.format(client_name=client_name),
            'cloud_infrastructure': self.CLOUD_INFRASTRUCTURE_QUERY_PROMPT.format(client_name=client_name),
            'cybersecurity_compliance': self.CYBERSECURITY_COMPLIANCE_QUERY_PROMPT.format(client_name=client_name),
            'data_analytics': self.DATA_ANALYTICS_QUERY_PROMPT.format(client_name=client_name),
            'financial_filings': self.FINANCIAL_FILINGS_QUERY_PROMPT.format(client_name=client_name),
        }

        results = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_type = {
                executor.submit(self._conduct_grounded_query, prompt, query_type): query_type
                for query_type, prompt in queries.items()
            }

            for future in as_completed(future_to_type):
                query_type = future_to_type[future]
                try:
                    results[query_type] = future.result()
                except Exception as e:
                    logger.error(f"Query '{query_type}' raised exception: {e}")
                    results[query_type] = GroundedQueryResult(
                        query_type=query_type,
                        text="",
                        success=False,
                        error=str(e),
                    )

        return results

    def _merge_grounding_metadata(self, results: dict) -> Optional[GroundingMetadata]:
        """Merge grounding metadata from multiple queries, deduplicating by URI."""
        from .grounding import merge_grounding_metadata
        return merge_grounding_metadata(results)

    def _apply_fallback_defaults(self, report_data: ResearchReportData, query_results: dict) -> ResearchReportData:
        """Apply fallback defaults for failed queries."""
        # Handle leadership query failure
        if query_results.get('leadership') and not query_results['leadership'].success:
            if not report_data.decision_makers:
                report_data.decision_makers = []

        # Handle technology query failure
        if query_results.get('technology') and not query_results['technology'].success:
            if not report_data.digital_maturity:
                report_data.digital_maturity = "unknown"
            if not report_data.ai_adoption_stage:
                report_data.ai_adoption_stage = "unknown"
            if not report_data.ai_footprint:
                report_data.ai_footprint = "Unable to determine AI footprint."

        # Handle news query failure
        if query_results.get('news') and not query_results['news'].success:
            if not report_data.recent_news:
                report_data.recent_news = []

        return report_data

    def conduct_deep_research(
        self,
        client_name: str,
        sales_history: str = "",
    ) -> tuple[ResearchReportData, Optional[GroundingMetadata]]:
        """Conduct deep research using 3-phase approach with Google Search grounding.

        Phase 1: Run 8 parallel grounded queries
        Phase 2: Synthesize results (MODEL_PRO with flash fallback)
        Phase 3: Format into structured JSON (MODEL_FLASH)

        Returns:
            tuple: (ResearchReportData, Optional[GroundingMetadata])
        """
        try:
            # Phase 1: Run parallel grounded queries
            logger.info(f"Phase 1: Running 8 parallel grounded queries for '{client_name}'")
            query_results = self._run_parallel_grounded_queries(client_name)

            # Merge grounding metadata from all queries
            merged_metadata = self._merge_grounding_metadata(query_results)

            # Build synthesis input from results
            def _text(key: str, default: str) -> str:
                r = query_results.get(key, GroundedQueryResult(query_type=key))
                return r.text or default

            profile_text = _text('profile', 'No profile data available.')
            news_text = _text('news', 'No news data available.')
            leadership_text = _text('leadership', 'No leadership data available.')
            technology_text = _text('technology', 'No technology data available.')
            cloud_infra_text = _text('cloud_infrastructure', 'No cloud infrastructure data available.')
            cybersec_text = _text('cybersecurity_compliance', 'No cybersecurity data available.')
            data_analytics_text = _text('data_analytics', 'No data analytics data available.')
            financial_text = _text('financial_filings', 'No financial filings data available.')

            # Phase 2: Synthesis with MODEL_PRO, flash fallback
            logger.info(f"Phase 2: Synthesizing research for '{client_name}'")
            synthesis_prompt = self.SYNTHESIS_PROMPT.format(
                client_name=client_name,
                sales_history=sales_history or "No prior sales history provided.",
                profile=profile_text,
                news=news_text,
                leadership=leadership_text,
                technology=technology_text,
                cloud_infrastructure=cloud_infra_text,
                cybersecurity_compliance=cybersec_text,
                data_analytics=data_analytics_text,
                financial_filings=financial_text,
            )

            try:
                synthesis_response = self.client.models.generate_content(
                    model=self.MODEL_PRO,
                    contents=synthesis_prompt,
                )
            except Exception:
                logger.warning("Pro model unavailable, falling back to flash for synthesis")
                synthesis_response = self.client.models.generate_content(
                    model=self.MODEL_FLASH,
                    contents=synthesis_prompt,
                )

            synthesis_text = synthesis_response.text

            # Phase 3: JSON formatting — stays on MODEL_FLASH
            logger.info(f"Phase 3: Formatting research for '{client_name}'")
            format_prompt = self.JSON_FORMAT_PROMPT.format(research_text=synthesis_text)

            format_response = self.client.models.generate_content(
                model=self.MODEL_FLASH,
                contents=format_prompt,
            )

            # Parse JSON response
            response_text = format_response.text.strip()

            # Handle potential markdown code blocks
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                # Remove first and last lines (```json and ```)
                response_text = '\n'.join(lines[1:-1])

            data = json.loads(response_text)
            # Filter to known fields to avoid unexpected keyword arguments
            known_fields = {f.name for f in ResearchReportData.__dataclass_fields__.values()}
            filtered_data = {k: v for k, v in data.items() if k in known_fields}
            report_data = ResearchReportData(**filtered_data)

            # Apply fallback defaults for any failed queries
            report_data = self._apply_fallback_defaults(report_data, query_results)

            return report_data, merged_metadata

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            # Return partial data with raw text in overview
            report_data = ResearchReportData(
                company_overview=f"Research completed but structured parsing failed. Raw synthesis output available."
            )
            return report_data, merged_metadata if 'merged_metadata' in locals() else None

        except Exception as e:
            logger.exception("Error during deep research")
            raise

    def classify_vertical(
        self,
        client_name: str,
        company_overview: str,
    ) -> str:
        """Classify a company into an industry vertical."""
        prompt = self.VERTICAL_CLASSIFICATION_PROMPT.format(
            client_name=client_name,
            company_overview=company_overview,
        )

        try:
            response = self.client.models.generate_content(
                model=self.MODEL_FLASH,
                contents=prompt,
            )

            vertical = response.text.strip().lower()

            # Validate against known verticals
            valid_verticals = [
                'healthcare', 'finance', 'retail', 'manufacturing', 'technology',
                'energy', 'telecommunications', 'media_entertainment', 'transportation',
                'real_estate', 'professional_services', 'education', 'government',
                'hospitality', 'agriculture', 'construction', 'nonprofit', 'other'
            ]

            if vertical in valid_verticals:
                return vertical
            else:
                logger.warning(f"Unknown vertical returned: {vertical}, defaulting to 'other'")
                return 'other'

        except Exception as e:
            logger.exception("Error during vertical classification")
            return 'other'

    def generate_text(self, prompt: str, model: str = None) -> str:
        """Generate text using Gemini API."""
        model = model or self.MODEL_FLASH
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            logger.exception("Error generating text")
            raise
