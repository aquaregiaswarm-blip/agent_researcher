# Codemap: Assets App

**App:** `backend/assets/`
**Purpose:** Generate customer-facing and internal sales enablement documents (personas, one-pagers, account plans, citations)

**Last Updated:** 2026-03-10

---

## Overview

The Assets app generates four types of sales documents from research intelligence:

1. **Buyer Personas** — 2–3 detailed profiles of key stakeholders (roles, goals, challenges, objections, messaging)
2. **One-Pagers** — Concise customer-facing sales documents (headline, challenge, solution, benefits, CTA)
3. **Account Plans** — Strategic account plans with SWOT, stakeholder map, engagement strategy, action plan, milestones
4. **Citations** — Structured source references with type, author, publication date, excerpt, verification status

The first three are AI-generated (Gemini `gemini-2.0-flash`); the fourth is manually managed. **All are backend-complete with zero UI implementation** (except sources tab showing raw `web_sources`, not Citation records).

---

## Models (`models.py`)

### Persona (AGE-21)
| Field | Type | Purpose |
|-------|------|---------|
| `id` | UUID (PK) | Unique identifier |
| `research_job` | FK → ResearchJob | Parent research job |
| `name` | CharField (255) | Persona archetype (e.g., "The Risk-Averse CFO") |
| `title` | CharField (255) | Job title (e.g., "Chief Financial Officer") |
| `department` | CharField (100) | Department (Finance, IT, Operations, etc.) |
| `seniority_level` | CharField (50) | C-Level, VP, Director, Manager |
| `background` | TextField | Professional background narrative |
| `goals` | JSONField (list) | Professional goals and aspirations |
| `challenges` | JSONField (list) | Day-to-day pain points and obstacles |
| `motivations` | JSONField (list) | What drives their decisions |
| `decision_criteria` | JSONField (list) | Factors that matter in vendor evaluation |
| `preferred_communication` | CharField (100) | Email, In-person, Video, Async |
| `objections` | JSONField (list) | Common objections they raise |
| `content_preferences` | JSONField (list) | Preferred formats (whitepapers, case studies, demos) |
| `key_messages` | JSONField (list) | Messaging that resonates |
| `created_at`, `updated_at` | DateTimeField | Metadata |

**Relationship:** Many personas per research job (FK, not OneToOne)

---

### OnePager (AGE-22)
| Field | Type | Purpose |
|-------|------|---------|
| `id` | UUID (PK) | Unique identifier |
| `research_job` | FK → ResearchJob | Parent research job |
| `title` | CharField (255) | Document title |
| `headline` | CharField (500) | Compelling headline capturing value prop |
| `executive_summary` | TextField | 2–3 sentence summary for executives |
| `challenge_section` | TextField | Business challenges being addressed |
| `solution_section` | TextField | How solution addresses challenges |
| `benefits_section` | TextField | Key benefits and outcomes |
| `differentiators` | JSONField (list) | 3–5 differentiating points |
| `call_to_action` | TextField | Clear, direct CTA |
| `next_steps` | JSONField (list) | Ordered list of next actions |
| `html_content` | TextField | HTML rendering (populated by `/html/` endpoint) |
| `pdf_path` | CharField (500) | File path to exported PDF (if generated) |
| `status` | CharField (20) | draft / reviewed / approved / shared |
| `created_at`, `updated_at` | DateTimeField | Metadata |

**Relationship:** Many one-pagers per research job (FK, not OneToOne)

---

### AccountPlan (AGE-23)
| Field | Type | Purpose |
|-------|------|---------|
| `id` | UUID (PK) | Unique identifier |
| `research_job` | OneToOne FK → ResearchJob | One account plan per research job |
| `title` | CharField (255) | Plan title (e.g., "Account Plan: Acme Corp") |
| `executive_summary` | TextField | High-level account strategy |
| `account_overview` | TextField | Account description and relationship status |
| `strategic_objectives` | JSONField (list) | 3–5 strategic objectives |
| `key_stakeholders` | JSONField (list) | List of {name, title, role_in_decision, engagement_approach} |
| `opportunities` | JSONField (list) | List of {name, value, timeline, probability} |
| `competitive_landscape` | TextField | Competitive positioning analysis |
| `swot_analysis` | JSONField (dict) | {strengths, weaknesses, opportunities, threats} — 4 lists |
| `engagement_strategy` | TextField | Overall engagement narrative |
| `value_propositions` | JSONField (list) | Key value props tailored to account |
| `action_plan` | JSONField (list) | List of {action, owner, due_date, status} |
| `success_metrics` | JSONField (list) | How success is measured |
| `milestones` | JSONField (list) | List of {milestone, target_date, criteria} |
| `timeline` | TextField | Overall timeline narrative |
| `html_content` | TextField | HTML rendering (populated by `/html/` endpoint) |
| `pdf_path` | CharField (500) | File path to exported PDF (if generated) |
| `status` | CharField (20) | draft / in_progress / reviewed / approved / active |
| `created_at`, `updated_at` | DateTimeField | Metadata |

**Relationship:** One account plan per research job (OneToOne FK)

---

### Citation (AGE-24)
| Field | Type | Purpose |
|-------|------|---------|
| `id` | UUID (PK) | Unique identifier |
| `research_job` | FK → ResearchJob | Parent research job |
| `citation_type` | CharField (20) | news / website / report / social / financial / press_release / other |
| `title` | CharField (500) | Title of cited article or document |
| `source` | CharField (255) | Publication or domain name (TechCrunch, Reuters, etc.) |
| `url` | URLField | Direct URL to source |
| `author` | CharField (255) | Author name (if known) |
| `publication_date` | DateField | Date source was published |
| `excerpt` | TextField | Relevant excerpt or quote from source |
| `relevance_note` | TextField | Why this source is relevant to research |
| `verified` | BooleanField | Whether source has been manually verified |
| `verification_date` | DateTimeField | When verification occurred |
| `created_at` | DateTimeField | Auto-set on creation |

**Relationship:** Many citations per research job (FK, not OneToOne)

**Note:** Citations are NOT auto-populated. The `ResearchReport.web_sources` field contains raw grounding URLs; Citation is a richer model for curated attribution.

---

## Services (`services/`)

| Service | File | Key Functions | Inputs | Outputs |
|---------|------|----------------|--------|---------|
| **PersonaGenerator** | `persona.py` | `generate_personas(research_job, limit=3)` | ResearchReport (decision_makers, pain_points, opportunities, digital_maturity), client name, vertical | 2–3 `Persona` records |
| **OnePagerGenerator** | `one_pager.py` | `generate_one_pager(research_job, use_case_id=None)` | ResearchReport, optional UseCase | 1 `OnePager` record |
| **AccountPlanGenerator** | `account_plan.py` | `generate_account_plan(research_job)` | ResearchReport, GapAnalysis, CompetitorCaseStudy[], InternalOpsIntelligence | 1 `AccountPlan` record (OneToOne with job) |
| **HtmlRenderer** | `html_renderer.py` | `render_one_pager_html(one_pager)`, `render_account_plan_html(account_plan)` | OnePager or AccountPlan | HTML string; stored in model's `html_content` field |
| **PdfExporter** | `pdf_exporter.py` | `export_to_pdf(html_content, output_path)` | HTML string | PDF file; `pdf_path` stored in model |

---

## Views & API (`views.py`, `urls.py`)

### Persona Endpoints

| Endpoint | Method | Action | Request Body | Returns |
|----------|--------|--------|--------------|---------|
| `/api/assets/personas/` | GET | List all personas (filterable by research_job) | — | `Persona[]` |
| `/api/assets/personas/` | POST | Create persona (rarely used — use generate instead) | Persona fields | `Persona` |
| `/api/assets/personas/{id}/` | GET | Retrieve single persona | — | `Persona` |
| `/api/assets/personas/generate/` | POST | Generate personas from research job | `{ "research_job_id": "uuid" }` | `Persona[]` (2–3 records) |

### OnePager Endpoints

| Endpoint | Method | Action | Request Body | Returns |
|----------|--------|--------|--------------|---------|
| `/api/assets/one-pagers/` | GET | List all one-pagers (filterable by research_job) | — | `OnePager[]` |
| `/api/assets/one-pagers/generate/` | POST | Generate one-pager | `{ "research_job_id": "uuid", "use_case_id": "uuid" (optional) }` | `OnePager` |
| `/api/assets/one-pagers/{id}/` | GET | Retrieve single one-pager | — | `OnePager` |
| `/api/assets/one-pagers/{id}/html/` | GET | Get HTML-rendered version | — | HTML string (populated in `html_content` field) |

### AccountPlan Endpoints

| Endpoint | Method | Action | Request Body | Returns |
|----------|--------|--------|--------------|---------|
| `/api/assets/account-plans/generate/` | POST | Generate account plan | `{ "research_job_id": "uuid" }` | `AccountPlan` |
| `/api/assets/account-plans/{id}/` | GET | Retrieve single account plan | — | `AccountPlan` |
| `/api/assets/account-plans/{id}/html/` | GET | Get HTML-rendered version | — | HTML string (populated in `html_content` field) |

### Citation Endpoints

| Endpoint | Method | Action | Request Body | Returns |
|----------|--------|--------|--------------|---------|
| `/api/assets/citations/` | GET | List all citations (filterable by research_job) | — | `Citation[]` |
| `/api/assets/citations/{id}/` | GET | Retrieve single citation | — | `Citation` |

**Note:** No POST endpoint for auto-generation. Citations are manually created or expected to be programmatically migrated from `ResearchReport.web_sources`.

---

## Serializers (`serializers.py`)

| Serializer | Model | Nested Objects | Purpose |
|------------|-------|-----------------|---------|
| `PersonaSerializer` | Persona | — | Full persona with all goals, challenges, messages |
| `OnePagerSerializer` | OnePager | — | Full document with all sections and HTML/PDF paths |
| `AccountPlanSerializer` | AccountPlan | — | Full strategic plan with all sections, SWOT, action items, milestones |
| `CitationSerializer` | Citation | — | Full source record with type, author, publication date, excerpt, verification |

---

## Gemini Integration

All services use `GeminiClient.generate_text(prompt)` (Type B: plain completion, no grounding).

### Persona Generation Prompt (`PERSONA_PROMPT`)
**Inputs:** Client name, vertical, decision makers, pain points, strategic goals, digital maturity

**Output format:** JSON array of personas
```json
{
  "personas": [
    {
      "name": "The Risk-Averse CFO",
      "title": "Chief Financial Officer",
      "department": "Finance",
      "seniority_level": "C-Level",
      "goals": ["Reduce costs", "Improve margins"],
      "challenges": ["Budget constraints", "Legacy systems"],
      "objections": ["Can't afford this", "Too risky"],
      "key_messages": ["ROI-focused", "Risk mitigation"],
      "content_preferences": ["ROI calculators", "Case studies"]
    }
  ]
}
```

### One-Pager Generation Prompt (`ONE_PAGER_PROMPT`)
**Inputs:** Client name, vertical, pain points, opportunities, optional use case details

**Output format:** JSON object
```json
{
  "title": "AI-Powered Operations for Acme Corp",
  "headline": "Cut operational costs by 30% with intelligent automation",
  "executive_summary": "Narrative...",
  "challenge_section": "Narrative...",
  "solution_section": "Narrative...",
  "benefits_section": "Narrative...",
  "differentiators": ["diff1", "diff2", "diff3"],
  "call_to_action": "Schedule a 30-minute discovery call",
  "next_steps": ["Exploratory call", "Proof of concept", "Implementation"]
}
```

### Account Plan Generation Prompt (`ACCOUNT_PLAN_PROMPT`)
**Inputs:** Client name, vertical, company overview, decision makers, pain points, opportunities, strategic goals, digital maturity, competitors, gap analysis

**Output format:** JSON object
```json
{
  "title": "Account Plan: Acme Corp",
  "executive_summary": "Narrative...",
  "account_overview": "Narrative...",
  "strategic_objectives": ["obj1", "obj2", "obj3"],
  "key_stakeholders": [
    {
      "name": "John Doe",
      "title": "CTO",
      "role_in_decision": "Technical approval",
      "engagement_approach": "Executive briefing on ROI and tech stack"
    }
  ],
  "opportunities": [
    {
      "name": "AI Automation",
      "value": "$2M",
      "timeline": "Q3 2026",
      "probability": 0.7
    }
  ],
  "swot_analysis": {
    "strengths": ["str1"],
    "weaknesses": ["weak1"],
    "opportunities": ["opp1"],
    "threats": ["threat1"]
  },
  "engagement_strategy": "Narrative...",
  "action_plan": [
    {
      "action": "Initial executive discovery",
      "owner": "Account Executive",
      "due_date": "2026-03-20",
      "status": "pending"
    }
  ],
  "success_metrics": ["metric1", "metric2"],
  "milestones": [
    {
      "milestone": "Proof of concept approved",
      "target_date": "2026-05-01",
      "criteria": ">50% cost reduction in pilot"
    }
  ],
  "timeline": "Narrative..."
}
```

---

## Data Flow Example

```
1. Research job completes
   ↓
2. User triggers "Generate Personas" → POST /api/assets/personas/generate/
   └─ PersonaGenerator.generate_personas(research_job)
      ├─ Query ResearchReport, research_job.vertical
      ├─ Call Gemini with PERSONA_PROMPT
      ├─ Parse JSON response
      └─ Create 2–3 Persona records
   ↓
3. User triggers "Generate One-Pager" → POST /api/assets/one-pagers/generate/
   └─ OnePagerGenerator.generate_one_pager(research_job, use_case_id=None)
      ├─ Query ResearchReport
      ├─ Optionally query UseCase if provided
      ├─ Call Gemini with ONE_PAGER_PROMPT
      ├─ Parse JSON response
      ├─ Create OnePager record
      ├─ Call HtmlRenderer.render_one_pager_html(one_pager)
      └─ Store HTML in one_pager.html_content
   ↓
4. User can call GET /api/assets/one-pagers/{id}/html/
   → Returns one_pager.html_content for browser viewing or email embedding
   ↓
5. User can trigger PDF export → HtmlRenderer + PdfExporter
   └─ Generates PDF, stores path in one_pager.pdf_path
   ↓
6. User triggers "Generate Account Plan" → POST /api/assets/account-plans/generate/
   └─ AccountPlanGenerator.generate_account_plan(research_job)
      ├─ Query ResearchReport, GapAnalysis, CompetitorCaseStudy[], InternalOpsIntelligence
      ├─ Call Gemini with ACCOUNT_PLAN_PROMPT
      ├─ Parse JSON response
      ├─ Create AccountPlan record (OneToOne with job)
      ├─ Call HtmlRenderer.render_account_plan_html(account_plan)
      └─ Store HTML in account_plan.html_content
   ↓
7. Personas, one-pagers, account plans can be starred to Work Products
```

---

## Frontend Integration (Status: Zero UI)

**Backend Complete:**
- All endpoints exist
- All services production-ready
- All models fully defined
- PDF/HTML export infrastructure in place
- All Gemini prompts tested

**Frontend Missing (per `TODO.md`):**
- No "Generate Personas" button anywhere
- No persona card list or detail view
- No "Generate One-Pager" button
- No one-pager preview panel
- No "Generate Account Plan" button
- No account plan view
- HTML/PDF render endpoints never called
- No API calls in `frontend/lib/api.ts` for any `/api/assets/` endpoints
- No TypeScript types for Persona, OnePager, AccountPlan, Citation in `frontend/types/index.ts`
- Work Products sidebar has category icons but no content can reach it

**UI Build-Out Needed (per `TODO.md`):**
1. Add API client methods in `frontend/lib/api.ts`
2. Add TypeScript types in `frontend/types/index.ts`
3. Build PersonaCard component (grid layout with goals, challenges, objections)
4. Build OnePagerPreview component (styled document layout)
5. Build AccountPlanView component (multi-section strategic document)
6. Add "Open HTML" button for viewing rendered documents
7. Add "Download PDF" button for exporting
8. Wire StarButton to all asset cards
9. Add navigation / routing (tab on research results or standalone page)

---

## Status Lifecycles

### Persona Status
Currently no status field; personas are either created or deleted. Could add a `status` field for curation if needed.

### OnePager Status Flow
```
draft → reviewed → approved → shared
```
- `draft` — Initially created
- `reviewed` — Internal QA complete
- `approved` — Ready to share
- `shared` — Sent to prospect

### AccountPlan Status Flow
```
draft → in_progress → reviewed → approved → active
```
- `draft` — Initially created
- `in_progress` — Being executed
- `reviewed` — Internal alignment
- `approved` — Ready for execution
- `active` — Currently in use

### Citation Status
No workflow; just a `verified` boolean flag with `verification_date`.

---

## Related Areas

- **Research App** (`docs/CODEMAPS/research.md`) — Provides ResearchReport, GapAnalysis as inputs
- **Ideation App** (`docs/CODEMAPS/ideation.md`) — Provides UseCase as optional input for one-pager generation
- **Projects App** (`docs/CODEMAPS/projects.md`) — Work Products sidebar stores personas, one-pagers, account plans
- **Memory App** (`docs/CODEMAPS/memory.md`) — Could auto-capture best personas/plays for reuse

---

## Testing & Development

```bash
# Backend setup
cd backend
source venv/bin/activate
export GEMINI_API_KEY="your-key"
python manage.py runserver

# Generate personas
curl -X POST http://localhost:8000/api/assets/personas/generate/ \
  -H "Content-Type: application/json" \
  -d '{"research_job_id": "your-job-uuid"}'

# List personas
curl http://localhost:8000/api/assets/personas/

# Generate one-pager
curl -X POST http://localhost:8000/api/assets/one-pagers/generate/ \
  -H "Content-Type: application/json" \
  -d '{"research_job_id": "your-job-uuid"}'

# Get HTML rendering
curl http://localhost:8000/api/assets/one-pagers/{id}/html/

# Generate account plan
curl -X POST http://localhost:8000/api/assets/account-plans/generate/ \
  -H "Content-Type: application/json" \
  -d '{"research_job_id": "your-job-uuid"}'

# Run tests
pytest assets/tests/
```

---

## Known Issues & Notes

- **No auto-generation trigger:** Personas, one-pagers, account plans must be triggered manually via API.
- **Citation population:** Currently unused. Decision needed: auto-populate from ResearchReport.web_sources or keep manual?
- **HTML/PDF export fragile:** Gemini occasionally wraps JSON in markdown code fences; HTML renderer must strip these.
- **OneToOne constraint:** Only one account plan per research job (OneToOne FK). If regeneration is needed, must delete and recreate.
- **No versioning:** Multiple generations create new records; old records remain.

---

## Epic References

| Epic | Feature | Model | Status |
|------|---------|-------|--------|
| AGE-21 | Buyer Persona Generation | Persona | Complete |
| AGE-22 | One-Pager Generator | OnePager | Complete |
| AGE-23 | Account Plan Generator | AccountPlan | Complete |
| AGE-24 | Citations | Citation | Complete (but unused) |

