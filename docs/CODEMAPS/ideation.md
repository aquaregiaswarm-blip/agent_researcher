# Codemap: Ideation App

**App:** `backend/ideation/`
**Purpose:** Generate use cases, feasibility assessments, and refined sales plays from research intelligence

**Last Updated:** 2026-03-10

---

## Overview

The Ideation app transforms a completed research job into actionable sales content. It runs three stages in sequence:

1. **Use Case Generation** — 3–5 AI-generated use cases with business problems, proposed solutions, ROI, timeline, and feasibility scores
2. **Feasibility Assessment** — Technical viability analysis per use case (risks, mitigation, prerequisites)
3. **Play Refinement** — Sales-ready playbook cards with elevator pitches, discovery questions, objection handlers, proof points

All three are powered by Gemini (`gemini-2.0-flash`), and all are **backend-complete with zero UI implementation**.

---

## Models (`models.py`)

| Model | Purpose | Key Fields | Relationships |
|-------|---------|-----------|---------------|
| `UseCase` | AI/technology use case (AGE-18) | `title`, `description`, `business_problem`, `proposed_solution`, `expected_benefits[]`, `estimated_roi`, `time_to_value`, `technologies[]`, `data_requirements[]`, `integration_points[]`, `priority` (high/medium/low), `impact_score`, `feasibility_score`, `status` (draft/validated/refined/approved/rejected) | FK → `ResearchJob` (many use cases per job) |
| `FeasibilityAssessment` | Technical assessment per use case (AGE-19) | `overall_feasibility` (low/medium/high), `overall_score`, `technical_complexity`, `data_availability`, `integration_complexity`, `scalability_considerations`, `technical_risks[]`, `mitigation_strategies[]`, `prerequisites[]`, `dependencies[]`, `recommendations`, `next_steps[]` | OneToOne FK → `UseCase` |
| `RefinedPlay` | Sales play card per use case (AGE-20) | `title`, `elevator_pitch`, `value_proposition`, `key_differentiators[]`, `target_persona`, `target_vertical`, `company_size_fit`, `discovery_questions[]`, `objection_handlers[]` (list of {objection, response}), `proof_points[]`, `competitive_positioning`, `next_steps[]`, `success_metrics[]`, `status` (draft/reviewed/approved/active/archived) | OneToOne FK → `UseCase` |

---

## Services (`services/`)

| Service | File | Key Functions | Inputs | Outputs |
|---------|------|----------------|--------|---------|
| **UseCaseGenerator** | `use_case_generator.py` | `generate_use_cases(research_job, limit=5)` | ResearchReport (overview, pain_points, opportunities, digital_maturity, ai_adoption_stage, gap analysis summary), client name, vertical | 3–5 `UseCase` records |
| **FeasibilityService** | `feasibility.py` | `assess_feasibility(use_case)` | UseCase details, company context (digital_maturity, ai_adoption_stage) | 1 `FeasibilityAssessment` record; updates `UseCase.feasibility_score` and `UseCase.status` to 'validated' |
| **PlayRefiner** | `play_refiner.py` | `refine_play(use_case, feasibility_assessment=None)` | UseCase, optional FeasibilityAssessment, company context | 1 `RefinedPlay` record |

---

## Views & API (`views.py`, `urls.py`)

**UseCaseViewSet** (DRF ModelViewSet)

| Endpoint | Method | Action | Request Body | Returns |
|----------|--------|--------|--------------|---------|
| `/api/ideation/use-cases/` | GET | List all use cases (filterable by research_job) | — | `UseCase[]` |
| `/api/ideation/use-cases/` | POST | Create a use case (rarely used — use generate endpoint instead) | UseCase fields | `UseCase` |
| `/api/ideation/use-cases/{id}/` | GET | Retrieve a single use case (with nested feasibility_assessment if exists) | — | `UseCase` |
| `/api/ideation/use-cases/{id}/assess/` | POST | Trigger feasibility assessment | — | `UseCase` (status updated to 'validated') |
| `/api/ideation/use-cases/{id}/refine/` | POST | Trigger play refinement | — | `RefinedPlay` (created) |

**Custom Actions**

| Endpoint | Method | Action | Request Body | Returns |
|----------|--------|--------|--------------|---------|
| `/api/ideation/use-cases/generate/` | POST | Generate use cases from a research job | `{ "research_job_id": "uuid" }` | `UseCase[]` (3–5 records) |

---

## Serializers (`serializers.py`)

| Serializer | Model | Nested Objects | Purpose |
|------------|-------|-----------------|---------|
| `UseCaseSerializer` | UseCase | `feasibility_assessment` (if exists) | Full use case with nested assessment; includes status, priority, scores |
| `FeasibilityAssessmentSerializer` | FeasibilityAssessment | — | Full assessment details: feasibility level, score, risks, mitigation, prerequisites |
| `RefinedPlaySerializer` | RefinedPlay | — | Full play with pitch, questions, objection handlers, proof points, competitive positioning |

---

## Gemini Integration

All services use `GeminiClient.generate_text(prompt)` (Type B: plain completion, no grounding).

### Use Case Generation Prompt (`USE_CASE_PROMPT`)
**Inputs:** Client name, vertical, company overview, pain points, opportunities, digital maturity, AI adoption stage, gap analysis summary

**Output format:** JSON array of use cases
```json
{
  "use_cases": [
    {
      "title": "...",
      "business_problem": "...",
      "proposed_solution": "...",
      "expected_benefits": ["benefit1", "benefit2"],
      "estimated_roi": "2-3x within 12 months",
      "time_to_value": "3-6 months",
      "technologies": ["LLM", "RAG", "Vector DB"],
      "data_requirements": ["customer records", "product logs"],
      "integration_points": ["Salesforce", "SAP"],
      "impact_score": 0.85,
      "priority": "high"
    }
  ]
}
```

### Feasibility Assessment Prompt (`FEASIBILITY_PROMPT`)
**Inputs:** Use case details, company digital maturity, AI adoption stage, industry vertical

**Output format:** JSON object
```json
{
  "overall_feasibility": "medium",
  "overall_score": 0.68,
  "technical_complexity": "Narrative...",
  "data_availability": "Narrative...",
  "integration_complexity": "Narrative...",
  "technical_risks": ["risk1", "risk2"],
  "mitigation_strategies": ["mitigation1", "mitigation2"],
  "prerequisites": ["prerequisite1"],
  "recommendations": "Narrative...",
  "next_steps": ["step1", "step2"]
}
```

### Play Refinement Prompt (`PLAY_REFINER_PROMPT`)
**Inputs:** Use case, feasibility assessment (if exists), company context (name, vertical, digital maturity)

**Output format:** JSON object
```json
{
  "elevator_pitch": "30-second pitch...",
  "value_proposition": "Detailed narrative...",
  "key_differentiators": ["diff1", "diff2"],
  "target_persona": "CTO / Head of Operations",
  "target_vertical": "Technology / Healthcare",
  "discovery_questions": ["question1", "question2", "question3"],
  "objection_handlers": [
    {
      "objection": "We don't have the data",
      "response": "Our platform can ingest from..."
    }
  ],
  "proof_points": ["proof1", "proof2"],
  "competitive_positioning": "vs Competitor X...",
  "next_steps": ["step1", "step2"],
  "success_metrics": ["metric1", "metric2"]
}
```

---

## Data Flow Example

```
1. Research job completes (ResearchReport + GapAnalysis created)
   ↓
2. User triggers "Generate Use Cases" via API (or future UI button)
   ↓
3. POST /api/ideation/use-cases/generate/
   └─ UseCaseGenerator.generate_use_cases(research_job)
      ├─ Query ResearchReport, GapAnalysis for context
      ├─ Call Gemini with USE_CASE_PROMPT
      ├─ Parse JSON response
      └─ Create 3–5 UseCase records (status: 'draft')
   ↓
4. Reps review use cases, select one to assess
   ↓
5. User triggers "Assess Feasibility" via API
   ↓
6. POST /api/ideation/use-cases/{id}/assess/
   └─ FeasibilityService.assess_feasibility(use_case)
      ├─ Query company context (digital_maturity, ai_adoption_stage)
      ├─ Call Gemini with FEASIBILITY_PROMPT
      ├─ Parse JSON response
      ├─ Create FeasibilityAssessment record
      └─ Update UseCase.status to 'validated', set feasibility_score
   ↓
7. Rep proceeds to refinement or moves to next use case
   ↓
8. User triggers "Refine Play" via API
   ↓
9. POST /api/ideation/use-cases/{id}/refine/
   └─ PlayRefiner.refine_play(use_case, feasibility_assessment=None)
      ├─ Query UseCase, optional FeasibilityAssessment, company context
      ├─ Call Gemini with PLAY_REFINER_PROMPT
      ├─ Parse JSON response
      └─ Create RefinedPlay record (status: 'draft')
   ↓
10. RefinedPlay can be starred to Work Products sidebar (future UI)
```

---

## Frontend Integration (Status: Zero UI)

**Backend Complete:**
- All endpoints exist
- All services production-ready
- All models fully defined
- All Gemini prompts tested

**Frontend Missing (per `TODO.md`):**
- No "Generate Use Cases" button anywhere
- No use case card list or detail view
- No "Assess Feasibility" button
- No feasibility results panel
- No "Refine Play" button
- No play card view or detail page
- No API calls in `frontend/lib/api.ts` for any `/api/ideation/` endpoints
- No TypeScript types for UseCase, FeasibilityAssessment, RefinedPlay in `frontend/types/index.ts`
- No Work Products sidebar integration

**UI Build-Out Needed (per `TODO.md`):**
1. Add API client methods in `frontend/lib/api.ts`
2. Add TypeScript types in `frontend/types/index.ts`
3. Build use case list/card components
4. Build feasibility panel (expandable or modal)
5. Build play card component with accordion for objection handlers
6. Wire StarButton to all asset cards
7. Add navigation / routing for ideation section (new tab on research results or standalone page)

---

## Status Lifecycle

### UseCase Status Flow
```
draft → validated → refined → approved
                 ↘
                   → rejected
```
- `draft` — Initially created by generator
- `validated` — After FeasibilityAssessment runs
- `refined` — After PlayRefiner runs
- `approved` — After review (manual update)
- `rejected` — Determined not viable (manual update)

### FeasibilityAssessment
One-to-one with UseCase; created on-demand when user clicks "Assess Feasibility"

### RefinedPlay Status Flow
```
draft → reviewed → approved → active
                          ↘
                            → archived
```
- `draft` — Initially created by refiner
- `reviewed` — After internal QA
- `approved` — Ready for use
- `active` — Currently in use
- `archived` — Retired from rotation

---

## Related Areas

- **Research App** (`docs/CODEMAPS/research.md`) — Provides ResearchReport and GapAnalysis as inputs
- **Assets App** (`docs/CODEMAPS/assets.md`) — Uses RefinedPlay to generate personas, one-pagers, account plans
- **Projects App** (`docs/CODEMAPS/projects.md`) — Work Products sidebar stores use cases and plays
- **Memory App** (`docs/CODEMAPS/memory.md`) — Promotes RefinedPlay records into reusable SalesPlay library

---

## Testing & Development

```bash
# Backend setup
cd backend
source venv/bin/activate
export GEMINI_API_KEY="your-key"
python manage.py runserver

# Generate use cases (via API)
curl -X POST http://localhost:8000/api/ideation/use-cases/generate/ \
  -H "Content-Type: application/json" \
  -d '{"research_job_id": "your-job-uuid"}'

# List use cases
curl http://localhost:8000/api/ideation/use-cases/

# Assess feasibility
curl -X POST http://localhost:8000/api/ideation/use-cases/{id}/assess/ \
  -H "Content-Type: application/json"

# Refine play
curl -X POST http://localhost:8000/api/ideation/use-cases/{id}/refine/ \
  -H "Content-Type: application/json"

# Run tests
pytest ideation/tests/
```

---

## Known Issues & Notes

- **No auto-generation trigger:** Use cases, assessments, and plays must be triggered manually via API. No automatic pipeline after research completes (unlike Memory auto-capture).
- **No UI visibility:** Backend is fully functional but invisible in frontend — Work Products sidebar has placeholder icons but nothing gets stored there.
- **Feasibility score decay:** UseCase initially has `feasibility_score: 0.0`; only updated after FeasibilityAssessment runs. Until then, impact_score is the only quality signal.
- **No versioning:** If a user triggers assessment or refinement multiple times, new records are created (not updated). Previous records remain in DB.

---

## Epic References

| Epic | Feature | Model | Status |
|------|---------|-------|--------|
| AGE-18 | Use Case Generation | UseCase | Complete |
| AGE-19 | Feasibility Assessment | FeasibilityAssessment | Complete |
| AGE-20 | Refined Sales Play | RefinedPlay | Complete |

