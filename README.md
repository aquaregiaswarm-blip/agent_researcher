# Deep Prospecting Engine

An AI-powered prospect research platform that automatically gathers comprehensive intelligence on potential clients, powered by Google Gemini and LangGraph. Built with Django 5 (backend), Next.js 14 (frontend), and organized around iterative project workflows.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Features](#features)
6. [API Overview](#api-overview)
7. [Environment Variables](#environment-variables)
8. [Documentation Index](#documentation-index)
9. [Development](#development)
10. [Tech Stack](#tech-stack)

---

## What It Does

The Deep Prospecting Engine automates sales research by gathering intelligence on prospects from multiple angles:

- **Deep Company Research** — Overview, leadership team, financial data, recent news, strategic goals
- **Digital Maturity Assessment** — Technology adoption level, AI readiness, digital transformation stage
- **Internal Operations Intelligence** — Employee sentiment, hiring trends, LinkedIn activity, social media presence, news coverage
- **Competitor Case Studies** — Find relevant AI/technology implementations from competitors in the same space
- **Gap Analysis** — Identify technology, capability, and process gaps with recommendations
- **Sales Readiness** — Generate actionable sales content (use cases, personas, one-pagers, account plans)

All research runs through a **LangGraph AI orchestration pipeline** that chains multiple Gemini API calls to gather structured, cross-referenced intelligence. Results can be organized into **iterative projects** where each round of research builds on prior findings.

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Google Gemini API key (free tier available)

### Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Run migrations
python manage.py migrate

# Start development server (runs on http://localhost:8000)
python manage.py runserver
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure environment (optional — defaults to http://localhost:8000)
# cp .env.example .env.local

# Start development server (runs on http://localhost:3000)
npm run dev
```

### Access the Application

- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000/api/
- **Django Admin:** http://localhost:8000/admin/

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend                             │
│                    Next.js 14 + React 18                     │
│         TypeScript + Tailwind CSS + React Hook Form          │
└─────────────────────────┬───────────────────────────────────┘
                          │ REST API
┌─────────────────────────┴───────────────────────────────────┐
│                         Backend                              │
│                    Django 5 + DRF                            │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              LangGraph Research Pipeline             │   │
│  │  (8 stages: validate → research → classify →        │   │
│  │   internal_ops → competitors → gaps → correlate →   │   │
│  │   finalize)                                          │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Google Gemini API (gemini-2.0-flash)              │   │
│  │  - Type A: Grounded queries (with Google Search)   │   │
│  │  - Type B: Plain completions (structured output)   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────┐ │
│  │  Django Apps    │  │  PostgreSQL DB   │  │  ChromaDB  │ │
│  │                 │  │                  │  │  (Vector   │ │
│  │  • research     │  │  • Research Jobs │  │   Store)   │ │
│  │  • ideation     │  │  • Projects      │  │            │ │
│  │  • assets       │  │  • Use Cases     │  │  Memory    │ │
│  │  • projects     │  │  • Personas      │  │  auto-     │ │
│  │  • memory       │  │  • One-Pagers    │  │  captures  │ │
│  │  • prompts      │  │  • Account Plans │  │            │ │
│  │                 │  │  • Annotations   │  │            │ │
│  └─────────────────┘  └──────────────────┘  └────────────┘ │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Project Structure

### Backend (`backend/`)

```
backend/
├── backend/                          # Django project settings
│   ├── settings/
│   │   ├── base.py                  # Shared settings
│   │   ├── dev.py                   # Development overrides (DEBUG=True, DRF browsable API)
│   │   └── prod.py                  # Production config
│   └── urls.py                      # URL routing
│
├── research/                         # Core AI research pipeline (AGE-10)
│   ├── models.py                    # ResearchJob, ResearchReport, CompetitorCaseStudy, GapAnalysis, InternalOpsIntelligence, GapCorrelation
│   ├── services/
│   │   ├── gemini.py               # GeminiClient (wraps google-genai API, Type A/B calls)
│   │   ├── classifier.py           # Vertical classification (18 industries)
│   │   ├── competitor.py           # Competitor case study search
│   │   ├── gap_analysis.py         # Gap analysis service
│   │   ├── internal_ops.py         # Employee sentiment, hiring, LinkedIn, social, news
│   │   └── gap_correlation.py      # Gap correlation with internal ops evidence
│   ├── graph/
│   │   ├── state.py               # ResearchState dataclass
│   │   ├── nodes.py               # 8 pipeline node functions
│   │   └── workflow.py            # LangGraph workflow builder
│   ├── views.py                    # ResearchJobViewSet (DRF)
│   ├── urls.py                     # API routes
│   └── serializers.py              # DRF serializers
│
├── ideation/                        # Use cases, feasibility, refined plays (AGE-18, 19, 20)
│   ├── models.py                   # UseCase, FeasibilityAssessment, RefinedPlay
│   ├── services/
│   │   ├── use_case_generator.py
│   │   ├── feasibility.py
│   │   └── play_refiner.py
│   ├── views.py, urls.py, serializers.py
│
├── assets/                          # Personas, one-pagers, account plans, citations (AGE-21, 22, 23, 24)
│   ├── models.py                   # Persona, OnePager, AccountPlan, Citation
│   ├── services/
│   │   ├── persona.py
│   │   ├── one_pager.py
│   │   ├── account_plan.py
│   │   ├── html_renderer.py        # HTML export
│   │   └── pdf_exporter.py         # PDF export
│   ├── views.py, urls.py, serializers.py
│
├── projects/                        # Project-based iterative workflow
│   ├── models.py                   # Project, Iteration, WorkProduct (generic FK), Annotation (generic FK)
│   ├── services/
│   │   ├── context.py              # ContextAccumulator (inject prior findings into new iterations)
│   │   └── comparison.py           # IterationComparator (side-by-side diff)
│   ├── views.py, urls.py, serializers.py
│
├── memory/                          # ChromaDB-backed vector store (AGE-14, 15, 16, 17)
│   ├── models.py                   # ClientProfile, SalesPlay, MemoryEntry
│   ├── services/
│   │   ├── vectorstore.py          # ChromaDB wrapper
│   │   ├── capture.py              # MemoryCapture (auto-run at end of research)
│   │   ├── context.py              # ContextRetriever (semantic search)
│   │   └── play_library.py         # PlayLibraryManager (reusable plays)
│   ├── views.py, urls.py, serializers.py
│
├── prompts/                         # Prompt template management
│   ├── models.py                   # PromptTemplate
│   └── views.py, urls.py
│
├── manage.py                        # Django CLI
├── requirements.txt                 # Python dependencies
├── conftest.py                      # Pytest config
└── .env.example                     # Environment template
```

### Frontend (`frontend/`)

```
frontend/
├── app/                             # Next.js App Router
│   ├── layout.tsx                  # Root layout (Navigation)
│   ├── page.tsx                    # Home page (quick research)
│   ├── research/
│   │   ├── page.tsx                # Research job list
│   │   └── [id]/page.tsx           # Research job detail + results
│   └── projects/
│       ├── page.tsx                # Project list
│       ├── new/page.tsx            # Create new project
│       └── [id]/
│           ├── page.tsx            # Project dashboard (iterations, work products, annotations)
│           └── iterate/page.tsx    # Start new iteration
│
├── components/
│   ├── ResearchForm.tsx            # Research job creation form
│   ├── ResearchResults.tsx         # Tabbed results view (Overview, Deep Research, Competitors, Gap Analysis, Inside Intel, Sources, Raw Output)
│   ├── Navigation.tsx              # Top navigation bar
│   ├── projects/
│   │   ├── ProjectList.tsx
│   │   ├── ProjectDetail.tsx       # Project dashboard layout
│   │   ├── IterationTimeline.tsx
│   │   ├── ComparisonView.tsx      # Side-by-side iteration diff
│   │   ├── WorkProductsSidebar.tsx # Starred items
│   │   └── AnnotationPanel.tsx     # Notes
│   └── common/
│       ├── StarButton.tsx          # Star/unstar button (built but unplaced)
│       ├── Button.tsx, Card.tsx, Modal.tsx, Loading.tsx
│
├── lib/
│   └── api.ts                      # API client class (fetch wrapper + all endpoints)
│
├── types/
│   └── index.ts                    # TypeScript interfaces
│
├── styles/
│   └── globals.css                 # Tailwind + global styles
│
├── next.config.js
├── package.json
├── tsconfig.json
└── .env.example                    # Environment template
```

---

## Features

### Live in UI

| Feature | Backend | Frontend | Page | Status |
|---------|---------|----------|------|--------|
| Quick single research job | ✅ | ✅ | Home (`/`) | **Live** |
| Research results tabs (8 tabs) | ✅ | ✅ | `/research/[id]` | **Live** |
| Project-based research | ✅ | ✅ | `/projects` | **Live** |
| Iterative research | ✅ | ✅ | `/projects/[id]` | **Live** |
| Context accumulation between iterations | ✅ | ✅ | `/projects/[id]/iterate` | **Live** |
| Iteration comparison (side-by-side diff) | ✅ | ✅ | `/projects/[id]` | **Live** |
| Work Products sidebar (star/save items) | ✅ | ✅ | `/projects/[id]` | **Live** |
| Annotations (user notes) | ✅ | ✅ | `/projects/[id]` | **Live** |
| PDF export of research results | ✅ | ✅ | `/research/[id]` | **Live** |

### Backend-Complete, No UI

| Feature | Backend | Frontend | Epic | Docs |
|---------|---------|----------|------|------|
| Use Case Generation | ✅ | ❌ | AGE-18 | `docs/feature-use-cases.md` |
| Feasibility Assessment | ✅ | ❌ | AGE-19 | `docs/feature-feasibility.md` |
| Refined Sales Plays | ✅ | ❌ | AGE-20 | `docs/feature-refined-play.md` |
| Buyer Personas | ✅ | ❌ | AGE-21 | `docs/feature-personas.md` |
| One-Pager Generator | ✅ | ❌ | AGE-22 | `docs/feature-one-pager.md` |
| Account Plan Generator | ✅ | ❌ | AGE-23 | `docs/feature-account-plan.md` |
| Citations (source tracking) | ✅ | ❌ | AGE-24 | `docs/feature-citations.md` |
| Memory / Knowledge Base | ✅ | ❌ | AGE-14/15/16/17 | `docs/feature-memory.md` |

**See `TODO.md` for full UI build-out roadmap.**

---

## API Overview

### Core Endpoints

#### Research
```
POST   /api/research/              Create research job
GET    /api/research/{id}/         Get job status & results
POST   /api/research/{id}/execute/ Start pipeline
GET    /api/research/{id}/export/pdf/  Export as PDF
```

#### Projects
```
GET    /api/projects/              List all projects
POST   /api/projects/              Create new project
GET    /api/projects/{id}/         Get project detail
PUT    /api/projects/{id}/         Update project
DELETE /api/projects/{id}/         Delete project

GET    /api/projects/{id}/iterations/         List iterations
POST   /api/projects/{id}/iterations/         Create iteration
GET    /api/projects/{id}/iterations/{seq}/   Get iteration
POST   /api/projects/{id}/iterations/{seq}/start/  Start research

GET    /api/projects/{id}/work-products/      List work products
POST   /api/projects/{id}/work-products/      Star item
DELETE /api/projects/{id}/work-products/{id}/ Unstar item

GET    /api/projects/{id}/annotations/        List annotations
POST   /api/projects/{id}/annotations/        Add note
DELETE /api/projects/{id}/annotations/{id}/   Delete note

GET    /api/projects/{id}/compare/?a=1&b=2   Compare iterations
GET    /api/projects/{id}/timeline/           Timeline view
```

#### Ideation (Backend only, zero UI)
```
POST   /api/ideation/use-cases/generate/    Generate use cases
GET    /api/ideation/use-cases/             List use cases
GET    /api/ideation/use-cases/{id}/        Get use case
POST   /api/ideation/use-cases/{id}/assess/ Assess feasibility
POST   /api/ideation/use-cases/{id}/refine/ Generate play

GET    /api/ideation/plays/                 List plays
GET    /api/ideation/plays/{id}/            Get play
```

#### Assets (Backend only, zero UI)
```
POST   /api/assets/personas/generate/           Generate personas
GET    /api/assets/personas/                    List personas
GET    /api/assets/personas/{id}/               Get persona

POST   /api/assets/one-pagers/generate/        Generate one-pager
GET    /api/assets/one-pagers/                 List one-pagers
GET    /api/assets/one-pagers/{id}/            Get one-pager
GET    /api/assets/one-pagers/{id}/html/       Get HTML version

POST   /api/assets/account-plans/generate/     Generate account plan
GET    /api/assets/account-plans/{id}/         Get account plan
GET    /api/assets/account-plans/{id}/html/    Get HTML version

GET    /api/assets/citations/                  List citations
GET    /api/assets/citations/{id}/             Get citation
```

#### Memory (Backend only, zero UI)
```
GET    /api/memory/profiles/                    List client profiles
GET    /api/memory/profiles/{id}/               Get profile
GET    /api/memory/plays/                       List sales plays
GET    /api/memory/plays/{id}/                  Get play
GET    /api/memory/entries/                     List memory entries
GET    /api/memory/entries/{id}/                Get entry
POST   /api/memory/context/                     Query for prior context
POST   /api/memory/capture/{id}/                Manually capture from research
```

---

## Environment Variables

### Backend (`backend/.env`)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SECRET_KEY` | String | Required | Django secret key |
| `DEBUG` | Boolean | `True` | Debug mode (set to `False` in production) |
| `GEMINI_API_KEY` | String | Required | Google Gemini API key |
| `CHROMA_PERSIST_DIR` | String | `./chroma_data` | ChromaDB persistence directory |
| `ALLOWED_HOSTS` | String | `localhost,127.0.0.1` | Comma-separated allowed hosts |
| `CORS_ALLOWED_ORIGINS` | String | `http://localhost:3000` | CORS allowed origins |
| `DATABASE_URL` | String | SQLite | Database connection string |

### Frontend (`frontend/.env.local`)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | String | `http://localhost:8000` | Backend API base URL |

---

## Documentation Index

| Document | Purpose | Location |
|----------|---------|----------|
| **Codemaps** | Architectural reference for each app | `docs/CODEMAPS/` |
| — Research app | 8-stage LangGraph pipeline, 6 models, Gemini integration | `docs/CODEMAPS/research.md` |
| — Ideation app | Use cases, feasibility, refined plays | `docs/CODEMAPS/ideation.md` |
| — Assets app | Personas, one-pagers, account plans, citations | `docs/CODEMAPS/assets.md` |
| — Projects app | Project workflow, iterations, context, comparison | `docs/CODEMAPS/projects.md` |
| — Memory app | ChromaDB knowledge base, auto-capture, semantic search | `docs/CODEMAPS/memory.md` |
| — Frontend | Next.js pages, components, types, API client | `docs/CODEMAPS/frontend.md` |
| **Feature Docs** | Detailed feature specifications and API endpoints | `docs/` |
| — Use Cases | Feature specification for use case generation | `docs/feature-use-cases.md` |
| — Feasibility | Feature specification for feasibility assessment | `docs/feature-feasibility.md` |
| — Refined Plays | Feature specification for sales play generation | `docs/feature-refined-play.md` |
| — Personas | Feature specification for persona generation | `docs/feature-personas.md` |
| — One-Pagers | Feature specification for one-pager generation | `docs/feature-one-pager.md` |
| — Account Plans | Feature specification for account plan generation | `docs/feature-account-plan.md` |
| — Citations | Feature specification for source tracking | `docs/feature-citations.md` |
| — Memory | Feature specification for knowledge base | `docs/feature-memory.md` |
| **Data Dictionary** | Complete database schema and field documentation | `docs/data-dictionary.md` |
| **Architecture** | System design and component interactions | `docs/architecture.md` |
| **TODO** | UI build-out roadmap (8+ features dark) | `TODO.md` |
| **API Summary** | Complete API call reference and UI panel mapping | `API_Research_Summary.md` |

---

## Development

### Running Tests

```bash
# Backend tests
cd backend
pytest                                  # Run all tests
pytest research/tests/                  # Test specific app
pytest -k "test_name"                   # Test by name
pytest -v                               # Verbose output

# Frontend tests
cd frontend
npm run test                            # Run Vitest
npm run test:ui                         # With UI
```

### Building for Production

```bash
# Backend
cd backend
python manage.py collectstatic
# Then deploy using your hosting platform

# Frontend
cd frontend
npm run build
npm start  # Or deploy to Vercel, Netlify, etc.
```

### Useful Commands

**Backend:**
```bash
cd backend
source venv/bin/activate

# Migrations
python manage.py makemigrations
python manage.py migrate

# Django shell
python manage.py shell

# Create superuser (for Django admin)
python manage.py createsuperuser

# Flush database (WARNING: deletes all data)
python manage.py flush
```

**Frontend:**
```bash
cd frontend

# Format code
npm run lint

# Type check
npm run type-check
```

---

## Tech Stack

### Backend
- **Framework:** Django 5, Django REST Framework
- **AI Orchestration:** LangGraph
- **LLM:** Google Gemini API (`gemini-2.0-flash`)
- **Vector Database:** ChromaDB (knowledge base)
- **Database:** PostgreSQL (production) / SQLite (development)
- **Language:** Python 3.11+
- **Task Queue:** Optional (for async tasks)

### Frontend
- **Framework:** Next.js 14 with App Router
- **UI Library:** React 18
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **Form Handling:** React Hook Form
- **HTTP Client:** Fetch API (via custom wrapper)
- **Testing:** Vitest

### Infrastructure
- **Hosting Options:**
  - Backend: Django on any Python-capable host (GCP Cloud Run, AWS Lambda, Heroku, etc.)
  - Frontend: Next.js on Vercel, Netlify, or any static host
- **API:** REST (DRF)
- **Authentication:** (Not yet implemented)

---

## Known Limitations

1. **No authentication** — All endpoints accessible without auth
2. **Ideation/Assets/Memory dark** — Backend complete but zero frontend UI (see `TODO.md`)
3. **Single Gemini model** — `gemini-2.0-flash` hardcoded; no model routing
4. **No streaming** — All API calls synchronous; no WebSocket updates
5. **ChromaDB in-memory by default** — Data lost on server restart unless persistence enabled
6. **Fragile JSON parsing** — Gemini occasionally wraps JSON in markdown code fences
7. **Limited error handling** — Some edge cases not covered
8. **No rate limiting** — Can hit Gemini API limits quickly
9. **Cascade deletes** — Deleting a project deletes all associated research data permanently
10. **GenericForeignKey inefficient** — Work Products queries can be N+1 without optimization

---

## Roadmap

**Priority 1: Complete UI for Backend-Complete Features**
- [ ] Build ideation section (use cases, feasibility, plays)
- [ ] Build assets section (personas, one-pagers, account plans)
- [ ] Build memory browser (knowledge base, sales play library)
- [ ] Wire StarButton to all asset cards

**Priority 2: Production Readiness**
- [ ] Add authentication and authorization
- [ ] Implement rate limiting
- [ ] Add error boundary components
- [ ] Comprehensive error handling
- [ ] Mobile/responsive optimization

**Priority 3: Performance & Scale**
- [ ] Optimize GenericForeignKey queries
- [ ] Implement caching strategy
- [ ] Add database indices
- [ ] Consider async task queue for long-running jobs

---

## Support & Contribution

Internal project — contact the development team for contribution guidelines.

---

## License

Proprietary — All rights reserved.

---

**Last Updated:** 2026-03-10
