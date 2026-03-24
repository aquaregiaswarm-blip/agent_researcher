"""Gemini API client for deep research."""
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from dataclasses import dataclass, field, asdict
from django.conf import settings

logger = logging.getLogger(__name__)


def _strip_invalid_citations(data: dict, max_n: int) -> dict:
    """Strip [N] markers where N > max_n (out-of-range citations from LLM hallucination)."""
    import re

    def clean(value):
        if isinstance(value, str):
            return re.sub(
                r'\[(\d+)\]',
                lambda m: m.group(0) if int(m.group(1)) <= max_n else '',
                value
            ).strip()
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    return {k: clean(v) for k, v in data.items()}


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

    PROFILE_QUERY_PROMPT = '''Build a factual company profile for {client_name}.

Answer each of these questions with specific, sourced facts:

1. What is {client_name}'s official company description, founding year, and headquarters location (city, state, country)?
2. What is the most recent reported annual revenue (exact figure or estimated range) and total employee count?
3. What industry and market segments does {client_name} operate in? What are the primary business lines or product categories?
4. Is {client_name} publicly traded (ticker symbol, exchange) or privately held? If private, identify ownership: PE-backed (which firm), founder-owned, or subsidiary (of which parent).
5. What are key subsidiaries, acquired companies, or major brand names under the {client_name} umbrella?
6. What is {client_name}'s geographic footprint — number of offices, countries of operation, or regional concentration?

OUTPUT FORMAT — use these section headers:
COMPANY SUMMARY: 3-4 sentence overview
KEY FACTS: Founded | HQ | Industry | Revenue | Employees | Ownership | Ticker (if public)
BUSINESS SEGMENTS: Each segment with a one-line description
GEOGRAPHIC PRESENCE: Operating regions and notable locations

For every factual claim, cite the source inline. If a data point is not found, state "Not publicly available."{prompt_focus}'''

    NEWS_QUERY_PROMPT = '''Produce a current-events intelligence briefing for {client_name}. Focus on strategic and business developments from the last 90 days.

Answer each of these questions:

1. Has {client_name} made any major business announcements in the last 90 days? Search for: acquisitions, divestitures, partnerships, product launches, market expansions, restructurings, or layoffs/RIFs.
2. Have there been earnings releases, revenue guidance changes, or notable stock price movements? Include specific numbers and dates.
3. Has {client_name} announced executive leadership changes — new hires, departures, or reorganizations — in the last 90 days?
4. Has {client_name} been featured in industry analyst reports (Gartner, Forrester, IDC) or received awards/recognition recently?
5. Are there any pending or recent regulatory actions, lawsuits, investigations, or compliance events involving {client_name}?

SCOPE: Focus ONLY on business strategy, financial, and organizational news. Do NOT cover employee culture, social media sentiment, or job postings — those are handled separately.

OUTPUT FORMAT — for each news item:
DATE | HEADLINE | 2-3 sentence summary | SOURCE with URL

Group by category: Financial & Business | Leadership & Personnel | Products & Partnerships | Regulatory & Legal

If fewer than 3 items found in the last 90 days, extend the window to 180 days and note this.{prompt_focus}'''

    LEADERSHIP_QUERY_PROMPT = '''Identify the executive leadership team and key technology decision makers at {client_name}.

Answer each of these questions:

1. Who is the CEO of {client_name}? When did they assume the role? What is their professional background (prior companies, board seats)?
2. Who holds these titles (or equivalents) at {client_name}? For each, provide name, exact title, and approximate tenure:
   - CTO or Chief Technology Officer
   - CIO or Chief Information Officer
   - CISO or Chief Information Security Officer
   - CDO or Chief Data Officer
   - CFO or Chief Financial Officer
   - VP/SVP of Engineering, Infrastructure, or IT
3. Has {client_name} made any technology or security leadership hires in the last 12 months? New hires in these roles signal shifting priorities.
4. Have any of these leaders made public statements about technology strategy, AI adoption, cloud migration, cybersecurity, or digital transformation? Search LinkedIn posts, conference talks, podcast appearances, and earnings call transcripts.
5. Who leads IT procurement, vendor management, or strategic sourcing at {client_name}?

OUTPUT FORMAT — for each person found:
NAME | EXACT TITLE | TENURE (start year or "Since YYYY") | BACKGROUND (2 sentences) | NOTABLE STATEMENTS (paraphrase with source, or "None found")

Group into:
- C-SUITE / SVP: CEO, CTO, CIO, CISO, CDO, CFO
- TECHNOLOGY LEADERSHIP: VP/Director level in IT, Engineering, Security, Data
- PROCUREMENT / VENDOR MANAGEMENT: If identifiable

Flag any role that appears unfilled or where the previous holder departed within 6 months.{prompt_focus}'''

    TECHNOLOGY_QUERY_PROMPT = '''Map {client_name}'s technology landscape, digital transformation programs, and AI adoption posture.

Answer each of these questions:

1. What enterprise software platforms does {client_name} use? Search for evidence of:
   - ERP: SAP, Oracle, Workday, NetSuite, Infor, Epicor
   - CRM: Salesforce, HubSpot, Microsoft Dynamics 365
   - Collaboration: Microsoft 365, Google Workspace, Slack, Zoom
   - ITSM: ServiceNow, BMC, Jira Service Management, Freshservice
   - HR/HCM: Workday, ADP, UKG, BambooHR
2. What technology signals appear in {client_name}'s job postings? Search for programming languages, frameworks, DevOps tools (Kubernetes, Terraform, Jenkins), and cloud-native patterns referenced in engineering job descriptions.
3. Has {client_name} publicly announced AI or ML initiatives? Search for:
   - AI vendor partnerships (OpenAI, Google, Microsoft Copilot, AWS Bedrock, etc.)
   - Internal AI/ML teams or hiring for AI roles
   - AI-powered product features or customer-facing AI
   - Generative AI pilots or governance frameworks
   - Published AI ethics or responsible AI policies
4. Has {client_name} announced named digital transformation programs? Include: program name, stated budget or investment range, timeline, and business objectives.
5. Does {client_name} operate an innovation lab, R&D center, technology incubator, or dedicated digital team?
6. What technology vendor partnerships has {client_name} announced in the last 24 months? Include: vendor name, partnership type (customer, reseller, co-development), and date.

OUTPUT FORMAT:
KNOWN TECHNOLOGY STACK: Table — Category | Vendor/Product | Evidence Source | Confidence (Confirmed = direct announcement; Inferred = job posting or indirect reference)
AI & ML POSTURE: Stage (None / Exploring / Piloting / Deployed at Scale), named initiatives, vendors
DIGITAL TRANSFORMATION PROGRAMS: Named programs with objectives, budget signals, timeline
TECHNOLOGY PARTNERSHIPS: Vendor | Partnership type | Date | Source
HIRING-DERIVED SIGNALS: Key technologies from job postings not in official announcements{prompt_focus}'''

    CLOUD_INFRASTRUCTURE_QUERY_PROMPT = '''Assess {client_name}'s cloud infrastructure strategy, data center posture, and infrastructure vendor ecosystem.

Answer each of these questions:

1. Does {client_name} have a relationship with AWS, Microsoft Azure, or Google Cloud? Search for: partnership tier announcements, case studies published by the cloud provider, mentions at re:Invent / Ignite / Next conferences, and cloud-specific job postings.
2. Has {client_name} announced cloud migration initiatives? Include: source workloads (mainframe, on-prem VMware, legacy DC), target platform, timeline, named migration partner or SI, and stated business driver.
3. Is there evidence of hybrid cloud or multi-cloud strategy? Search for mentions of: VMware (vSphere, VCF, Tanzu), Nutanix, Red Hat OpenShift, Azure Arc, AWS Outposts, Google Anthos, or any "cloud-adjacent" infrastructure.
4. Does {client_name} operate its own data centers? Search for: data center locations, colocation partnerships (Equinix, Digital Realty, QTS, CyrusOne), consolidation announcements, or new facility investments.
5. What networking and infrastructure vendors does {client_name} use? Search for: Cisco (Meraki, Catalyst, ACI), Arista, Juniper, Palo Alto Networks, Fortinet, SD-WAN providers (Cisco Viptela, VMware VeloCloud, Cato Networks), and load balancers (F5, NGINX).
6. What do {client_name}'s infrastructure-related job postings reveal about platform preferences? Search for cloud architect, SRE, platform engineer, network engineer postings and note specific technologies required.

OUTPUT FORMAT:
CLOUD STRATEGY SUMMARY: 2-3 sentences on overall posture (cloud-first, hybrid, legacy-heavy)
CLOUD PROVIDER RELATIONSHIPS: Provider | Relationship type (customer/partner/case study) | Evidence | Source
MIGRATION INITIATIVES: Scope | Timeline | Partner | Business driver
ON-PREMISES / HYBRID SIGNALS: Named platforms and evidence
INFRASTRUCTURE VENDORS: Vendor | Category (networking/compute/storage/security) | Evidence
HIRING SIGNALS: Relevant job titles and technology requirements{prompt_focus}'''

    CYBERSECURITY_COMPLIANCE_QUERY_PROMPT = '''Assess {client_name}'s cybersecurity posture, compliance profile, and security investment signals.

Answer each of these questions:

1. Has {client_name} experienced any publicly disclosed data breaches, security incidents, or regulatory enforcement actions? For each: date, nature of incident, records affected, regulatory response, and remediation actions. Search breach notification databases and news archives.
2. What compliance certifications does {client_name} hold or is required to maintain? Search for: SOC 2 Type II, ISO 27001, HIPAA/HITRUST, FedRAMP, PCI-DSS, CMMC (and target level), StateRAMP, GDPR, NIST 800-171, or industry-specific frameworks (NERC CIP for energy, FFIEC for financial services).
3. Who is the CISO or most senior security leader at {client_name}? Has there been turnover in this role in the last 12 months?
4. Has {client_name} announced cybersecurity vendor partnerships or tool deployments? Search for:
   - SIEM/SOAR: Splunk, Microsoft Sentinel, Elastic, IBM QRadar, Chronicle
   - EDR/XDR: CrowdStrike, SentinelOne, Microsoft Defender, Carbon Black
   - Identity/IAM: Okta, CyberArk, SailPoint, Microsoft Entra, Ping Identity
   - Network Security: Palo Alto (Prisma, Cortex), Zscaler, Fortinet, Netskope
   - Email Security: Proofpoint, Mimecast, Abnormal Security
   - Vulnerability Mgmt: Tenable, Qualys, Rapid7
5. What regulatory environment creates compliance pressure for {client_name}? Consider: financial services (SEC, OCC, FFIEC, NYDFS), healthcare (HHS OCR), government contracting (CMMC, NIST 800-171, ITAR), privacy laws (CCPA/CPRA, Quebec Bill 25, EU AI Act), and critical infrastructure (CISA directives).
6. What security-related job postings reveal about maturity and tool preferences? Search for SOC analyst, security engineer, GRC analyst, penetration tester postings.

OUTPUT FORMAT:
SECURITY INCIDENTS: Date | Incident | Impact | Outcome | Source (or "No public incidents found")
COMPLIANCE PROFILE: Certification | Status (Confirmed / Required by regulation / Inferred) | Evidence
SECURITY LEADERSHIP: Name | Title | Tenure | Recent changes
SECURITY VENDOR ECOSYSTEM: Vendor | Category | Evidence source
REGULATORY PRESSURE POINTS: Regulation | Why it applies | Compliance implication
HIRING SIGNALS: Security job postings with notable tool/framework requirements{prompt_focus}'''

    DATA_ANALYTICS_QUERY_PROMPT = '''Assess {client_name}'s data infrastructure, analytics capabilities, and data governance posture.

Answer each of these questions:

1. What data platform does {client_name} use? Search for evidence of:
   - Data Warehouse: Snowflake, Databricks, Google BigQuery, Amazon Redshift, Azure Synapse, Teradata
   - Data Lake: Databricks (Delta Lake), AWS Lake Formation, Azure Data Lake
   - Data Integration/ETL: Informatica, Fivetran, Matillion, Talend, dbt, Airflow
   - Streaming: Kafka, Confluent, Amazon Kinesis, Azure Event Hubs
2. What BI and analytics tools does {client_name} use? Search for: Tableau, Power BI, Looker, Qlik, ThoughtSpot, Sigma Computing, Sisense, Domo.
3. Does {client_name} have a Chief Data Officer, VP of Data, VP of Analytics, or Head of Data Engineering? Include name, title, and background.
4. Has {client_name} announced data governance, data quality, data catalog, or master data management initiatives? Search for partnerships with: Collibra, Alation, Informatica (MDM/CDQ), Ataccama, Monte Carlo (data observability), or Great Expectations.
5. Has {client_name} made public statements about data strategy, becoming "data-driven," or investing in advanced analytics, ML pipelines, real-time analytics, or data mesh architecture?
6. What do data engineering and analytics job postings reveal? Search for: specific tools (dbt, Spark, Airflow, Kafka), cloud data services (Redshift, BigQuery, Synapse), and data governance or DataOps roles.

OUTPUT FORMAT:
DATA PLATFORM STACK: Category (Warehouse/Lake/BI/ETL/Governance/Streaming) | Vendor | Evidence | Confidence
DATA LEADERSHIP: Name | Title | Background
DATA STRATEGY SIGNALS: Named initiatives, investment themes, public commitments
DATA MATURITY ASSESSMENT: Level (Basic Reporting / Self-Service BI / Predictive Analytics / AI-Driven Decision Making) with evidence
HIRING SIGNALS: Data/analytics job postings and stack references{prompt_focus}'''

    FINANCIAL_FILINGS_QUERY_PROMPT = '''Extract technology-relevant intelligence from {client_name}'s financial disclosures, investor communications, and analyst coverage.

Answer each of these questions:

1. In {client_name}'s most recent 10-K or annual report, what technology investment themes are discussed? Search for mentions of: digital transformation, AI/ML, automation, cloud migration, cybersecurity investment, data analytics, IT modernization, or technical debt remediation.
2. What technology-related risk factors does {client_name} disclose? Search for: cybersecurity risk, IT system failure risk, technology obsolescence, data privacy litigation exposure, and regulatory technology mandates.
3. In the last 2 quarterly earnings calls, did executives discuss: AI initiatives, technology spending plans, vendor consolidation, IT headcount changes, cloud costs, or digital customer experience investments? Provide specific paraphrases with the quarter referenced.
4. What is the capital expenditure trend? Is there evidence of increased or decreased technology spending? Include dollar amounts or percentage changes if disclosed.
5. Has {client_name} made technology-related acquisitions, divestitures, or strategic investments in the last 24 months? Include: target name, deal value, date, and the technology capability acquired.
6. What do sell-side analysts say about {client_name}'s technology strategy, digital competitiveness, or IT spending trajectory?

If {client_name} is privately held: Search instead for private funding rounds, PE portfolio pages, debt issuances, press releases referencing investment, and any available revenue estimates from Crunchbase, PitchBook, or press coverage.

OUTPUT FORMAT:
TECHNOLOGY INVESTMENT THEMES: Theme | Description | Source (filing, earnings call, press release)
RISK FACTORS (TECH-RELATED): Risk | Description | Filing reference
EARNINGS CALL INTELLIGENCE: Quarter | Speaker | Key statement (paraphrased) | Technology relevance
CAPEX / SPENDING SIGNALS: Trend direction, dollar amounts, % of revenue if available
M&A / STRATEGIC INVESTMENTS: Target | Date | Deal value | Technology capability acquired
ANALYST COMMENTARY: Analyst/Firm | Key insight | Date{prompt_focus}'''

    SYNTHESIS_PROMPT = '''You are a senior presales research analyst. Synthesize the following research into a comprehensive intelligence brief for a sales team preparing to engage {client_name}.

Your audience is account executives and presales engineers at a value-added reseller and systems integrator. They need actionable intelligence for deal strategy, not a Wikipedia summary.

## Research Directive (from submitter — prioritize these themes):
{prompt}

## Sales Context (existing relationship and history):
{sales_history}

## RESEARCH INPUTS

### Company Profile:
{profile}

### Financial & Investor Signals:
{financial_filings}

### Recent News & Developments:
{news}

### Leadership & Decision Makers:
{leadership}

### Technology Stack & Digital Maturity:
{technology}

### Cloud & Infrastructure:
{cloud_infrastructure}

### Data & Analytics:
{data_analytics}

### Cybersecurity & Compliance:
{cybersecurity_compliance}

## SYNTHESIS INSTRUCTIONS

Produce a cohesive analysis covering these sections. For each section, lead with the most actionable insight — not the most generic fact.

1. **Company Overview**: Key facts and what makes this account strategically interesting
2. **Leadership & Decision Makers**: Who to engage, their stated priorities, and professional context that informs how they buy
3. **Recent Developments**: Last 90 days — what has changed that creates urgency or timing
4. **Technology Landscape**: Known stack, digital maturity level, and where they stand versus industry peers
5. **Cloud & Infrastructure Posture**: Current state, migration trajectory, vendor loyalties
6. **Cybersecurity & Compliance**: Certifications, incidents, regulatory pressure, security vendor ecosystem
7. **Data & Analytics Maturity**: Platform stack, leadership, governance posture
8. **Financial Signals**: Technology investment themes from filings, earnings calls, and analyst commentary
9. **Pain Points**: Specific, evidence-backed problems. Each pain point must reference a source or signal from the research — not generic industry challenges.
10. **Opportunities for Our Solutions**: Where our capabilities (cloud, security, data, AI, infrastructure, managed services, application modernization) map to their gaps. Be specific about which solution area fits and why.
11. **Strategic Goals & Initiatives**: Their stated objectives and named programs
12. **Technology Partnerships**: Existing vendor relationships — both potential allies and competitive incumbents
13. **Recommended Talking Points**: 5-7 conversation starters tailored to the sales context and research directive. Each must reference a specific finding from the research.

## AVAILABLE SOURCES (numbered for citation):
{source_list}

## CITATION RULES:
- After each specific fact, add [N] using the source number above.
- Only cite when clearly attributable to a specific source.
- Do not cite general industry knowledge or your own analysis.
- Preserve citation numbers exactly — do not renumber or modify.'''

    JSON_FORMAT_PROMPT = '''Convert the following research synthesis into structured JSON. Your response MUST be valid JSON matching the exact structure below.

## Research to Format:
{research_text}

## Required JSON Structure:
{{
    "company_overview": "Comprehensive overview emphasizing what makes this account strategic",
    "founded_year": 2000,
    "headquarters": "City, State/Country",
    "employee_count": "Specific number or range, e.g. '12,500' or '10,000-15,000'",
    "annual_revenue": "Specific figure or range, e.g. '$2.3B' or '$1B-$2B'",
    "website": "https://example.com",
    "recent_news": [
        {{
            "title": "Headline",
            "summary": "2-3 sentence summary with specific details",
            "date": "YYYY-MM-DD",
            "source": "Publication name",
            "url": "https://source.com/article"
        }}
    ],
    "decision_makers": [
        {{
            "name": "Full Name",
            "title": "Exact Title",
            "background": "2-sentence background including prior companies and stated priorities",
            "linkedin_url": ""
        }}
    ],
    "pain_points": ["Each pain point should be specific and evidence-backed, with [N] citation where applicable"],
    "opportunities": ["Each opportunity should name the solution area and reference supporting evidence [N]"],
    "digital_maturity": "nascent|developing|maturing|advanced|leading",
    "ai_footprint": "Detailed description: named tools, vendors, specific initiatives, governance frameworks",
    "ai_adoption_stage": "none|exploring|experimenting|implementing|scaling|optimizing",
    "strategic_goals": ["Specific goals with details — not generic aspirations"],
    "key_initiatives": ["Named initiative with timeline if known"],
    "talking_points": ["Each talking point should reference a specific finding and suggest an opening question"],
    "cloud_footprint": "Named providers, migration status, hybrid signals, infrastructure vendors with specifics",
    "security_posture": "Certifications held, incidents, CISO name/status, named security vendors, regulatory pressures",
    "data_maturity": "Named platforms (warehouse, BI, ETL), data leadership, governance posture with specifics",
    "financial_signals": ["Each signal should cite the source: 10-K, earnings call quarter, analyst report [N]"],
    "tech_partnerships": ["Each entry should name the vendor, relationship type, and whether they are a competitive threat"]
}}

## RULES:
- CITATION PRESERVATION: The research text contains [N] citation markers. Preserve these EXACTLY within string values. Do not remove, renumber, or modify them.
- If a field has no data, use empty string "" for strings, empty array [] for arrays, or null for numbers. Never fabricate data to fill a field.
- Respond ONLY with valid JSON. No markdown fences, no preamble, no commentary.'''

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

    def _run_parallel_grounded_queries(self, client_name: str, prompt: str = "") -> dict:
        """Run 8 grounded queries in parallel using ThreadPoolExecutor."""
        prompt_focus = ""
        if prompt and prompt.strip():
            prompt_focus = (
                f"\n\nResearch focus (from submitter): {prompt}\n"
                "If focus areas are specified above, ensure those topics receive deeper coverage."
            )

        queries = {
            'profile': self.PROFILE_QUERY_PROMPT.format(client_name=client_name, prompt_focus=prompt_focus),
            'news': self.NEWS_QUERY_PROMPT.format(client_name=client_name, prompt_focus=prompt_focus),
            'leadership': self.LEADERSHIP_QUERY_PROMPT.format(client_name=client_name, prompt_focus=prompt_focus),
            'technology': self.TECHNOLOGY_QUERY_PROMPT.format(client_name=client_name, prompt_focus=prompt_focus),
            'cloud_infrastructure': self.CLOUD_INFRASTRUCTURE_QUERY_PROMPT.format(client_name=client_name, prompt_focus=prompt_focus),
            'cybersecurity_compliance': self.CYBERSECURITY_COMPLIANCE_QUERY_PROMPT.format(client_name=client_name, prompt_focus=prompt_focus),
            'data_analytics': self.DATA_ANALYTICS_QUERY_PROMPT.format(client_name=client_name, prompt_focus=prompt_focus),
            'financial_filings': self.FINANCIAL_FILINGS_QUERY_PROMPT.format(client_name=client_name, prompt_focus=prompt_focus),
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
        prompt: str = "",
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
            query_results = self._run_parallel_grounded_queries(client_name, prompt=prompt)

            # Merge grounding metadata from all queries
            merged_metadata = self._merge_grounding_metadata(query_results)

            # Build numbered source list for citation instructions
            source_list_text = ""
            if merged_metadata and merged_metadata.web_sources:
                source_list_text = "\n".join(
                    f"{i+1}. {s.title or 'Source'} — {s.uri}"
                    for i, s in enumerate(merged_metadata.web_sources)
                )

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
            prompt_value = prompt.strip() if prompt and prompt.strip() else "No specific research directive provided."
            synthesis_prompt = self.SYNTHESIS_PROMPT.format(
                client_name=client_name,
                prompt=prompt_value,
                sales_history=sales_history or "No prior sales history provided.",
                profile=profile_text,
                news=news_text,
                leadership=leadership_text,
                technology=technology_text,
                cloud_infrastructure=cloud_infra_text,
                cybersecurity_compliance=cybersec_text,
                data_analytics=data_analytics_text,
                financial_filings=financial_text,
                source_list=source_list_text,
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

            # Strip out-of-range citation markers that the LLM may have hallucinated
            max_citations = len(merged_metadata.web_sources) if merged_metadata else 0
            if max_citations:
                data = _strip_invalid_citations(data, max_citations)

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
