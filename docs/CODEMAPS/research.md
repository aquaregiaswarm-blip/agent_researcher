# Codemap: Research App

**App:** `backend/research/`
**Purpose:** Core AI research pipeline using LangGraph and Gemini for deep prospect intelligence gathering

**Last Updated:** 2026-03-10

---

## Overview

The Research app orchestrates the complete AI research workflow. It runs 8 sequential pipeline stages (LangGraph nodes) that transform a company name into a comprehensive research report with competitor analysis, gap analysis, and internal operations intelligence. All stages use Google Gemini (`gemini-2.0-flash`), with the first phase using Google Search grounding for web data.

---

## LangGraph Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Research Workflow                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. validate() → Check inputs, API key                       │
│       ↓                                                       │
│  2. research() → Phase 1-3: Grounded queries + synthesis     │
│       ↓          (Returns ResearchReport)                     │
│  3. classify() → Vertical classification (18 industries)     │
│       ↓                                                       │
│  4. internal_ops() → Employee sentiment, talent, LinkedIn    │
│       ↓                                                       │
│  5. competitors() → AI case studies from competitors         │
│       ↓                                                       │
│  6. gap_analysis() → Tech/capability/process gaps            │
│       ↓                                                       │
│  7. correlate() → Cross-reference gaps with internal ops     │
│       ↓                                                       │
│  ✓─── finalize() → Persist all results, auto-capture memory  │
│                                                              │
│  [should_continue] → Exit early on status == 'failed'       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Files & Modules

### Models (`models.py`)

| Model | Purpose | Key Fields |
|-------|---------|-----------|
| `ResearchJob` | Tracks research job status and lifecycle | `client_name`, `status` (pending/running/completed/failed), `vertical`, `result` (raw JSON), `error`, `iteration` (FK) |
| `ResearchReport` | Structured deep research output (1:1 with ResearchJob) | `company_overview`, `founded_year`, `headquarters`, `employee_count`, `annual_revenue`, `website`, `recent_news`, `decision_makers`, `pain_points`, `opportunities`, `digital_maturity`, `ai_adoption_stage`, `ai_footprint`, `strategic_goals`, `key_initiatives`, `talking_points`, `web_sources` |
| `CompetitorCaseStudy` | Competitor AI implementations found during research | `competitor_name`, `vertical`, `case_study_title`, `summary`, `technologies_used`, `outcomes`, `source_url`, `relevance_score` |
| `GapAnalysis` | Technology, capability, and process gaps identified | `technology_gaps`, `capability_gaps`, `process_gaps`, `recommendations`, `priority_areas`, `confidence_score`, `analysis_notes` |
| `InternalOpsIntelligence` | Employee sentiment, hiring, social presence (internal Intel tab) | `employee_sentiment` (JSON), `job_postings`, `linkedin_presence`, `social_media_mentions`, `news_sentiment`, `key_insights`, `data_freshness` |
| `GapCorrelation` | Evidence linking gaps to internal operations signals | `gap_type`, `description`, `evidence`, `evidence_type` (supporting/contradicting/neutral), `confidence`, `sales_implication` |

---

### Services (`services/`)

| Service | File | Key Functions | Purpose |
|---------|------|----------------|---------|
| **GeminiClient** | `gemini.py` | `generate_text(prompt)`, `generate_with_grounding(prompt)`, `call_parallel_grounded_queries()` | Wraps Google Gemini API; handles 2 call types: Type A (grounded + Google Search) for Phase 1, Type B (plain completion) for all other services |
| **ClassifierService** | `classifier.py` | `classify_vertical(research_report)` | Maps company overview to 1 of 18 industry verticals |
| **CompetitorSearchService** | `competitor.py` | `search_competitors(research_report)` | Finds 3–5 AI case studies from competitors with relevance scores |
| **GapAnalysisService** | `gap_analysis.py` | `analyze_gaps(research_report)` | Identifies technology, capability, and process gaps with recommendations |
| **InternalOpsService** | `internal_ops.py` | `research_internal_ops(company_name, vertical)` | Fetches employee sentiment, job postings, LinkedIn data, social media mentions, news sentiment |
| **GapCorrelationService** | `gap_correlation.py` | `correlate_gaps(gaps, internal_ops, company_name)` | Cross-references gaps with internal ops evidence; flags supporting/contradicting signals |

---

### Graph (LangGraph Workflow)

| File | Contents | Purpose |
|------|----------|---------|
| `graph/state.py` | `ResearchState` dataclass | Unified state dict passed between pipeline nodes; includes `client_name`, `research_report`, `competitor_case_studies`, `gap_analysis`, `internal_ops`, `gap_correlations`, `status`, `error` |
| `graph/nodes.py` | Node functions: `validate_inputs()`, `execute_research()`, `classify_vertical()`, `research_internal_ops()`, `search_competitors()`, `analyze_gaps()`, `correlate_gaps()`, `finalize_result()` | Each node transforms ResearchState; nodes are orchestrated by workflow.py |
| `graph/workflow.py` | `build_research_workflow()` | Compiles LangGraph graph with 8 nodes, conditional `should_continue` edge, and Gemini service integration |

---

### Views & API (`views.py`, `urls.py`)

**ResearchJobViewSet** (DRF ModelViewSet)

| Endpoint | Method | Action | Returns |
|----------|--------|--------|---------|
| `/api/research/` | GET | List all research jobs | `ResearchJob[]` |
| `/api/research/` | POST | Create new research job | `ResearchJob` |
| `/api/research/{id}/` | GET | Retrieve job status & results | `ResearchJob` (includes `report`, `competitor_case_studies`, `gap_analysis`, `internal_ops`, `gap_correlations`) |
| `/api/research/{id}/execute/` | POST | Start the LangGraph pipeline | `ResearchJob` (async execution) |
| `/api/research/{id}/export/pdf/` | GET | Export research as PDF | Binary PDF file |

---

### Serializers (`serializers.py`)

| Serializer | Model | Purpose |
|------------|-------|---------|
| `ResearchJobSerializer` | ResearchJob | Serializes job status, client name, vertical classification |
| `ResearchReportSerializer` | ResearchReport | Nested in job response; includes all structured report fields |
| `CompetitorCaseStudySerializer` | CompetitorCaseStudy | List of competitors with case studies, technologies, outcomes, relevance scores |
| `GapAnalysisSerializer` | GapAnalysis | Tech/capability/process gaps with recommendations and confidence |
| `InternalOpsSerializer` | InternalOpsIntelligence | Employee sentiment, job postings, LinkedIn data, social media, news |
| `GapCorrelationSerializer` | GapCorrelation | Gap type, evidence, evidence type, confidence, sales implication |

---

## Gemini Call Types

### Type A: Grounded (Google Search enabled)
**Used only in Phase 1** by `GeminiClient.generate_with_grounding()`

```python
# 4 parallel grounded queries (ThreadPoolExecutor, max_workers=4):
1. PROFILE_QUERY_PROMPT    → Company overview, HQ, revenue, employee count, founding year
2. NEWS_QUERY_PROMPT       → Recent news headlines with dates and sources
3. LEADERSHIP_QUERY_PROMPT → Executive names, titles, backgrounds
4. TECHNOLOGY_QUERY_PROMPT → Tech stack, digital maturity, AI adoption
```

Returns grounding metadata: `web_sources[]` with `uri` and `title` of every source used.

### Type B: Plain Completion (no Google Search)
**Used by all other services** via `GeminiClient.generate_text()`

```python
# Synthesis:          Combines 4 Phase 1 results into narrative
# JSON Formatting:    Converts narrative to structured JSON
# Vertical Classifier: Maps company to 18 industry verticals
# Competitor Search:  Finds AI case studies from competitors
# Gap Analysis:       Identifies tech/capability/process gaps
# Internal Ops:       Employee sentiment, hiring, social, news
# Gap Correlation:    Cross-references gaps with internal ops evidence
```

---

## Constants & Enums (`constants.py`)

| Enum | Values | Purpose |
|------|--------|---------|
| `Vertical` | 18 industry types (e.g., "Technology", "Healthcare", "Finance", "Retail") | Industry vertical classification |
| `DigitalMaturityLevel` | "Beginner", "Intermediate", "Advanced", "Mature" | Company's digital transformation maturity |
| `AIAdoptionStage` | "Exploring", "Piloting", "Scaling", "Optimizing" | Company's AI adoption lifecycle stage |

---

## Data Flow Example

```
User Input: "Acme Corp" (sales history optional)
     ↓
POST /api/research/
     ↓
ResearchJob created (status: pending)
     ↓
POST /api/research/{id}/execute/
     ↓
1. validate_inputs() → Check GEMINI_API_KEY, client_name
2. execute_research() → Phase 1-3 (4 grounded + 2 plain Gemini calls)
   - Returns ResearchReport with: company_overview, decision_makers,
     pain_points, opportunities, digital_maturity, ai_adoption_stage,
     strategic_goals, web_sources (grounding URLs)
3. classify_vertical() → ResearchJob.vertical = "Technology"
4. research_internal_ops() → InternalOpsIntelligence created
   (employee_sentiment, job_postings, linkedin_presence, etc.)
5. search_competitors() → 3–5 CompetitorCaseStudy records created
6. analyze_gaps() → GapAnalysis created
   (technology_gaps, capability_gaps, process_gaps, recommendations)
7. correlate_gaps() → GapCorrelation records created
   (links gaps to employee sentiment, job postings, news, etc.)
8. finalize_result() → Persist all to DB, trigger memory auto-capture
     ↓
ResearchJob.status = "completed"
     ↓
Frontend polls /api/research/{id}/ → Gets full research + all nested objects
     ↓
UI renders: Overview tab (quick stats), Deep Research tab (full details),
Competitors tab, Gap Analysis tab, Inside Intel tab (internal ops + correlations),
Sources tab (web_sources), Raw Output tab
```

---

## Frontend Integration (Frontend Status: Incomplete)

**What's wired up:**
- Research job creation form (home page)
- Research job polling and status display
- Results tabs: Overview, Deep Research, Competitors, Gap Analysis, Inside Intel, Sources, Raw Output
- PDF export

**What's missing:**
- No Ideation section (use cases, plays, feasibility) — Backend complete, UI zero
- No Asset generation buttons (personas, one-pagers, account plans) — Backend complete, UI zero
- No Memory browser — Backend auto-captures, UI invisible
- No integration of Work Products sidebar with research results

See `TODO.md` for full build-out roadmap.

---

## Related Areas

- **Ideation App** (`docs/CODEMAPS/ideation.md`) — Builds on ResearchReport to generate use cases and sales plays
- **Assets App** (`docs/CODEMAPS/assets.md`) — Generates personas, one-pagers, account plans from research
- **Projects App** (`docs/CODEMAPS/projects.md`) — Wraps research jobs in iterative project workflow
- **Memory App** (`docs/CODEMAPS/memory.md`) — Auto-captures research insights for reuse
- **Prompts App** — Configurable default research prompt template

---

## Setup & Testing

```bash
# Backend setup
cd backend
source venv/bin/activate
export GEMINI_API_KEY="your-key"
python manage.py migrate

# Run dev server
python manage.py runserver

# Run tests
pytest research/tests/
pytest -k "test_research_job"  # Single test by name
pytest research/tests/test_views.py::test_create_research_job  # Single test

# Django shell
python manage.py shell
>>> from research.models import ResearchJob
>>> jobs = ResearchJob.objects.all()
```

---

## Known Issues & Notes

- **Fragile JSON parsing:** Services strip markdown code fences before `json.loads()` — Gemini occasionally wraps JSON in ` ```json ` blocks despite instructions
- **No streaming:** All Gemini calls use synchronous `generate_content()`, no streaming to frontend
- **Hard-coded model:** `gemini-2.0-flash` is not configurable; no model routing or fallback
- **API key auth:** Uses raw API key, not Vertex AI / ADC — consider migration for GCP deployment
- **Grounding URL deduplication:** Web sources are deduplicated by URI before storage in `ResearchReport.web_sources`
- **Memory auto-capture:** Runs at end of `finalize_result()` node silently; no UI confirmation or visibility

---

## Version History

| Date | Changes |
|------|---------|
| 2026-03-10 | Initial codemap — 8 pipeline stages, 6 models, 3 Gemini call types, grounding metadata |
| 2026-02-03 | Added internal operations intelligence and gap correlation |
| 2026-01-15 | Core research pipeline with vertical classification |
