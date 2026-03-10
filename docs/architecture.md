# Deep Prospecting Engine — Architecture

> Last updated: 2026-03-10

---

## 1. System Overview

The Deep Prospecting Engine is an AI-powered sales research tool designed to help sales teams gather comprehensive intelligence about prospects and generate tailored sales assets. It automates the research process, identifies opportunities, and creates personalized sales materials.

### Target Users
- Sales representatives conducting pre-call research
- Account executives preparing for enterprise deals
- Sales managers planning account strategies
- Pre-sales engineers identifying technical opportunities

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Frontend (Next.js 14)                      │
│                    App Router — :3000                        │
└──────────────────────────┬──────────────────────────────────┘
                           │ REST API (JSON)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               Backend (Django 5 + DRF — :8000)              │
├──────────┬───────────┬──────────┬───────────┬───────────────┤
│ research │ ideation  │  assets  │ projects  │    memory     │
├──────────┴───────────┴──────────┴───────────┴───────────────┤
│                  LangGraph Orchestration                     │
├─────────────────────────────────────────────────────────────┤
│         Google Gemini API          │   ChromaDB Vector DB   │
│       (gemini-2.0-flash)           │   (local / persisted)  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│          SQLite (dev)  /  PostgreSQL (prod)                  │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, App Router, TypeScript, Tailwind CSS |
| Backend | Django 5.1, Django REST Framework |
| AI Orchestration | LangGraph |
| AI Model | Google Gemini `gemini-2.0-flash` (via `google-genai` SDK) |
| Vector Store | ChromaDB |
| Database (dev) | SQLite |
| Database (prod) | PostgreSQL |
| Auth (Gemini) | API key — `GEMINI_API_KEY` env var |

---

## 2. Backend Django Apps

### `research/` — Core Research Pipeline

**Purpose:** Orchestrates the AI-powered research workflow using LangGraph, manages research jobs, and stores all structured research results.

**Models:** `ResearchJob`, `ResearchReport`, `CompetitorCaseStudy`, `GapAnalysis`, `InternalOpsIntel`

**Services:**
- `gemini.py` → `GeminiClient` — central Gemini API wrapper, all AI calls
- `classifier.py` → `VerticalClassifier` — industry vertical classification
- `competitor.py` → `CompetitorSearchService` — competitor case study search
- `gap_analysis.py` → `GapAnalysisService` — technology/capability gap identification
- `internal_ops.py` → `InternalOpsService` — employee sentiment and operational intelligence
- `gap_correlation.py` → `GapCorrelationService` — cross-references gaps with internal ops

---

### `ideation/` — Use Case Generation Pipeline

**Purpose:** Generates AI/technology use cases from research, assesses feasibility, and creates refined sales plays.

**Models:** `UseCase`, `FeasibilityAssessment`, `RefinedPlay`

**Services:**
- `use_case_generator.py` → `UseCaseGenerator`
- `feasibility.py` → `FeasibilityService`
- `play_refiner.py` → `PlayRefiner`

**Status:** Backend complete — no frontend coverage.

---

### `assets/` — Sales Asset Generation

**Purpose:** Creates sales collateral: buyer personas, one-pagers, account plans, and source citations.

**Models:** `Persona`, `OnePager`, `AccountPlan`, `Citation`

**Services:**
- `persona.py` → `PersonaGenerator`
- `one_pager.py` → `OnePagerGenerator`
- `account_plan.py` → `AccountPlanGenerator`

**Status:** Backend complete — no frontend coverage.

---

### `projects/` — Iterative Workflow

**Purpose:** Manages multi-iteration research projects with context accumulation and comparison between runs.

**Models:** `Project`, `Iteration`, `WorkProduct`, `Annotation`

**Services:**
- `context.py` → `ContextAccumulator` — builds inherited context from previous iterations
- `comparison.py` → `IterationComparisonService` — diffs results between two iterations

**Status:** Backend and frontend complete.

---

### `memory/` — Vector Knowledge Base

**Purpose:** Persists intelligence from every research job into ChromaDB for semantic retrieval in future sessions.

**Models:** `ClientProfile`, `SalesPlay`, `MemoryEntry`

**Services:**
- `vectorstore.py` — ChromaDB operations (add, query, retrieve)
- `capture.py` → `MemoryCapture` — auto-captures from completed research jobs

**Status:** Backend complete — auto-capture runs on every job, but no frontend UI.

---

### `prompts/` — Prompt Template Management

**Purpose:** Manages configurable prompt templates. Currently supports a single default prompt that can be edited via the UI.

**Models:** `PromptTemplate`

**Status:** Backend and frontend complete (limited).

---

## 3. LangGraph Research Pipeline

The full research workflow runs as a compiled LangGraph graph. All nodes receive and return the full `ResearchState` dict. A `should_continue` conditional edge after each node exits early on `status == 'failed'`.

### Node Sequence

```
validate
   │
   ▼
research  ←── 4 parallel grounded Gemini calls (ThreadPoolExecutor)
   │
   ▼
classify
   │
   ▼
internal_ops
   │
   ▼
competitors
   │
   ▼
gap_analysis
   │
   ▼
correlate
   │
   ▼
finalize  ──→ DB persistence + memory auto-capture
```

### Node Responsibilities

| Node | Function | Key State Fields Written |
|---|---|---|
| `validate` | Check client_name + GEMINI_API_KEY | `status` |
| `research` | 4× grounded queries → synthesis → JSON | `result`, `research_report`, `web_sources` |
| `classify` | Industry vertical classification | `vertical` |
| `internal_ops` | Employee sentiment, LinkedIn, jobs, social | `internal_ops` |
| `competitors` | 3–5 competitor AI case studies | `competitor_case_studies` |
| `gap_analysis` | Technology/capability/process gaps | `gap_analysis` |
| `correlate` | Cross-reference gaps with internal ops | `gap_correlations` |
| `finalize` | Persist all records to DB | `status = 'completed'` |

### Phase 1 Parallel Queries

The `research` node fires 4 grounded Gemini calls simultaneously:

```python
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {
        'profile':    submit(grounded_query, PROFILE_QUERY_PROMPT),
        'news':       submit(grounded_query, NEWS_QUERY_PROMPT),
        'leadership': submit(grounded_query, LEADERSHIP_QUERY_PROMPT),
        'technology': submit(grounded_query, TECHNOLOGY_QUERY_PROMPT),
    }
```

Grounded calls use `types.Tool(google_search=types.GoogleSearch())` — Gemini fetches live web data and returns source URLs in grounding metadata.

---

## 4. Gemini API Call Map

All calls use `gemini-2.0-flash`. Two types:

### Type A — Grounded (Google Search enabled)
Used only in Phase 1 of `research` node. Returns grounding metadata (web sources).

### Type B — Plain completion
Used by all other services via `GeminiClient.generate_text()`.

| Call # | Node/Service | Type | Prompt |
|---|---|---|---|
| 1 | research — profile | **A (grounded)** | `PROFILE_QUERY_PROMPT` |
| 2 | research — news | **A (grounded)** | `NEWS_QUERY_PROMPT` |
| 3 | research — leadership | **A (grounded)** | `LEADERSHIP_QUERY_PROMPT` |
| 4 | research — technology | **A (grounded)** | `TECHNOLOGY_QUERY_PROMPT` |
| 5 | research — synthesis | B | `SYNTHESIS_PROMPT` |
| 6 | research — JSON format | B | `JSON_FORMAT_PROMPT` |
| 7 | classify | B | `VERTICAL_CLASSIFICATION_PROMPT` |
| 8 | competitors | B | `COMPETITOR_SEARCH_PROMPT` |
| 9 | gap_analysis | B | `GAP_ANALYSIS_PROMPT` |
| 10 | internal_ops | B | `INTERNAL_OPS_PROMPT` |
| 11 | correlate | B | `GAP_CORRELATION_PROMPT` |

**11 Gemini calls per core research job.** Additional on-demand calls for use cases, feasibility, plays, personas, one-pagers, and account plans.

---

## 5. Frontend Architecture

### Pages (Next.js 14 App Router)

| Route | File | Purpose |
|---|---|---|
| `/` | `app/page.tsx` | Quick single research job form |
| `/research/[id]` | `app/research/[id]/page.tsx` | Research results with tabs |
| `/research` | `app/research/page.tsx` | Research job history |
| `/projects` | `app/projects/page.tsx` | Project list |
| `/projects/new` | `app/projects/new/page.tsx` | Create project |
| `/projects/[id]` | `app/projects/[id]/page.tsx` | Project dashboard + iterations |
| `/projects/[id]/iterate` | `app/projects/[id]/iterate/page.tsx` | Run new iteration |

### Research Results Tabs

The `ResearchResults` component renders 7 tabs:

| Tab | Data Source | Service(s) |
|---|---|---|
| Overview | `report.*`, `decision_makers`, `pain_points`, `opportunities`, `talking_points` | GeminiClient Phase 1–3 |
| Deep Research | `report.*` full detail, `recent_news`, `strategic_goals`, `key_initiatives` | GeminiClient Phase 1–3 |
| Competitors | `competitor_case_studies[]` | CompetitorSearchService |
| Gap Analysis | `gap_analysis` | GapAnalysisService |
| Inside Intel | `internal_ops` + `gap_correlations` | InternalOpsService + GapCorrelationService |
| Sources | `report.web_sources[]` | Grounding metadata from Type A calls |
| Raw Output | `job.result` | Plain text result |

### API Client (`frontend/lib/api.ts`)

The `ApiClient` class wraps all backend calls. Currently covers:
- Research jobs (create, get, execute, poll, PDF export)
- Projects (CRUD)
- Iterations (CRUD, start, poll, compare, timeline)
- Work products (CRUD)
- Annotations (CRUD)
- Prompt templates (get default, update default)

**Zero calls** to `/api/ideation/`, `/api/assets/`, or `/api/memory/`.

---

## 6. Settings & Configuration

### Module Structure

```
backend/backend/settings/
├── base.py     ← shared settings (installed apps, middleware, DB, etc.)
├── dev.py      ← DEBUG=True, DRF browsable API, extends base.py
└── prod.py     ← production overrides, extends base.py
```

`DJANGO_SETTINGS_MODULE` defaults to `backend.settings.dev`.

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `SECRET_KEY` | Yes (prod) | dev hardcoded | Django secret key |
| `GEMINI_API_KEY` | Yes | — | Google AI API key |
| `CHROMA_PERSIST_DIR` | No | `./chroma_data` | ChromaDB storage path |
| `ALLOWED_HOSTS` | No | `localhost` | Comma-separated hosts |
| `CORS_ALLOWED_ORIGINS` | No | `localhost:3000` | Comma-separated CORS origins |
| `DEBUG` | No | `True` (dev) | Debug mode |

---

## 7. Architectural Observations

### Strengths

- **Clean app separation** — each Django app has a focused, single responsibility
- **Service layer** — all AI logic is in services, not views or models
- **LangGraph pipeline** — node-based workflow is easy to extend or reorder
- **Non-fatal failures** — pipeline continues gracefully when non-critical nodes fail
- **Immutable state** — each LangGraph node returns a new state dict, never mutates
- **Structured output** — dataclasses provide type safety for AI responses

### Concerns

**Security:**
- No authentication or authorisation on any endpoint (all `AllowAny`)
- No rate limiting
- API key auth only (not Vertex AI IAM)

**Scalability:**
- Research execution is synchronous — blocks a Django worker for 1–5 minutes
- No job queue (Celery/Redis) for background processing
- No caching layer
- Single Gemini model version hardcoded — no routing or fallback

**Data integrity:**
- All FK relationships use `CASCADE` delete — accidental deletion risk
- No audit trail or soft deletes
- No user ownership/tenancy on any model
- Heavy use of JSONField where relational tables might be better

**Testing:**
- Minimal test coverage
- No integration tests for the LangGraph pipeline
- No E2E tests

**Dark features (backend complete, no UI):**
- `ideation` — use cases, feasibility, refined plays
- `assets` — personas, one-pagers, account plans, citations
- `memory` — client profiles, sales play library, knowledge base
- `StarButton` component built but never rendered on any page

---

## 8. Recommended Next Steps

| Priority | Area | Action |
|---|---|---|
| High | Security | Add authentication (JWT or session) to all endpoints |
| High | Scalability | Move research execution to Celery background job |
| High | UI | Wire frontend to ideation and assets backend |
| Medium | Memory | Add memory browser UI — highest long-term value |
| Medium | Data | Add indexes on high-frequency query fields |
| Medium | Testing | Add integration tests for LangGraph pipeline |
| Low | Observability | Add Sentry, structured logging, APM |
| Low | Schema | Normalise key JSONFields (decision_makers, key_stakeholders) |
