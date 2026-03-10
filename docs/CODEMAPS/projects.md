# Codemap: Projects App

**App:** `backend/projects/`
**Purpose:** Wrap research jobs in iterative project workflow with context accumulation, work products sidebar, annotations, and iteration comparison

**Last Updated:** 2026-03-10

---

## Overview

The Projects app organizes sales research into project-based engagements. Each project contains multiple research iterations with the option to build context from previous findings. Key features:

- **Iterative Research** — Run multiple research cycles on the same prospect, optionally building on prior findings
- **Context Accumulation** — Automatically inject insights from previous iterations into new research prompts
- **Work Products** — Star and save important findings (use cases, plays, personas, one-pagers) to a project sidebar
- **Annotations** — Add notes to research outputs
- **Iteration Comparison** — Side-by-side diff view between two iterations

All UI for this is **complete and functional**. This is the primary workflow in the application.

---

## Models (`models.py`)

### Project
| Field | Type | Purpose |
|-------|------|---------|
| `id` | UUID (PK) | Unique identifier |
| `name` | CharField (255) | Project name (e.g., "Acme Corp - Q1 2026") |
| `client_name` | CharField (255) | Company name being researched |
| `description` | TextField | Optional project description |
| `context_mode` | CharField (20) | `accumulate` (build on prior) or `fresh` (start clean) |
| `created_at`, `updated_at` | DateTimeField | Metadata |

**Methods:**
- `latest_iteration` — Property returning most recent iteration
- `get_iteration_count()` — Number of iterations in project

**Relationships:**
- `iterations` (reverse FK from Iteration)
- `work_products` (reverse FK from WorkProduct)
- `annotations` (reverse FK from Annotation)

---

### Iteration
| Field | Type | Purpose |
|-------|------|---------|
| `id` | UUID (PK) | Unique identifier |
| `project` | FK → Project | Parent project |
| `sequence` | PositiveInteger | Iteration number (1, 2, 3, ...) |
| `name` | CharField (255, blank) | Optional label (e.g., "v2 - Focus on AI") |
| `sales_history` | TextField | Sales context for this iteration |
| `prompt_override` | TextField | Additional guidance beyond default prompt |
| `status` | CharField (20) | pending / running / completed / failed |
| `accumulated_context` | TextField | Context injected from previous iterations (if context_mode='accumulate') |
| `created_at`, `updated_at` | DateTimeField | Metadata |

**Relationships:**
- `project` (FK)
- `research_job` (OneToOne reverse from ResearchJob) — Created when iteration starts

---

### WorkProduct
| Field | Type | Purpose |
|-------|------|---------|
| `id` | UUID (PK) | Unique identifier |
| `project` | FK → Project | Parent project |
| `content_type` | ForeignKey (Django ContentType) | Polymorphic: UseCase, Persona, OnePager, etc. |
| `object_id` | CharField (255) | ID of the content object |
| `content_object` | GenericForeignKey (content_type + object_id) | Actual content (UseCase, Persona, etc.) |
| `title` | CharField (255) | Display title for sidebar |
| `summary` | TextField | Brief summary for preview |
| `product_type` | CharField (20) | `use_case`, `play`, `persona`, `one_pager`, `account_plan`, `gap`, `insight` (for search/filter) |
| `created_at`, `updated_at` | DateTimeField | Metadata |

**Purpose:** Generic relation to allow any model to be starred/saved to project sidebar.

---

### Annotation
| Field | Type | Purpose |
|-------|------|---------|
| `id` | UUID (PK) | Unique identifier |
| `project` | FK → Project | Parent project |
| `content_type` | ForeignKey (Django ContentType) | Polymorphic: can annotate any model |
| `object_id` | CharField (255) | ID of the content being annotated |
| `content_object` | GenericForeignKey | Actual content object |
| `text` | TextField | The annotation text (user note) |
| `created_at`, `updated_at` | DateTimeField | Metadata |

**Purpose:** Generic relation for user-created notes on any research output.

---

## Services (`services/`)

| Service | File | Key Functions | Purpose |
|---------|------|----------------|---------|
| **ContextAccumulator** | `context.py` | `get_context_from_previous_iteration(iteration)`, `inject_into_prompt(prompt, context)` | Retrieves insights from prior iteration's research report; injects into new prompt if context_mode='accumulate' |
| **IterationComparator** | `comparison.py` | `compare_iterations(iteration_a, iteration_b)` | Generates side-by-side diff of two iterations' research results |

---

## Views & API (`views.py`, `urls.py`)

### Project Endpoints

| Endpoint | Method | Action | Request Body | Returns |
|----------|--------|--------|--------------|---------|
| `/api/projects/` | GET | List all projects (paginated, ordered by -updated_at) | — | `ProjectListItem[]` |
| `/api/projects/` | POST | Create new project | `{ "name", "client_name", "description", "context_mode" }` | `Project` |
| `/api/projects/{id}/` | GET | Retrieve full project with iterations, work products, annotations | — | `Project` |
| `/api/projects/{id}/` | PUT | Update project fields | `{ "name", "description", "context_mode" }` | `Project` |
| `/api/projects/{id}/` | DELETE | Delete project (cascades to iterations, work products, annotations) | — | 204 No Content |

### Iteration Endpoints

| Endpoint | Method | Action | Request Body | Returns |
|----------|--------|--------|--------------|---------|
| `/api/projects/{id}/iterations/` | GET | List all iterations in project (ordered by sequence) | — | `Iteration[]` |
| `/api/projects/{id}/iterations/` | POST | Create new iteration (auto-increments sequence) | `{ "name", "sales_history", "prompt_override" }` | `Iteration` |
| `/api/projects/{id}/iterations/{seq}/` | GET | Retrieve iteration + linked research job + research results | — | `Iteration` (with nested ResearchJob + report) |
| `/api/projects/{id}/iterations/{seq}/start/` | POST | Start research for this iteration | — | `ResearchJob` (status: pending, then runs async) |

### Work Products Endpoints

| Endpoint | Method | Action | Request Body | Returns |
|----------|--------|--------|--------------|---------|
| `/api/projects/{id}/work-products/` | GET | List all starred items in project (filterable by product_type) | — | `WorkProduct[]` |
| `/api/projects/{id}/work-products/` | POST | Star/save an item to project | `{ "content_type": "use_case", "object_id": "uuid", "title", "summary" }` | `WorkProduct` |
| `/api/projects/{id}/work-products/{product_id}/` | DELETE | Unstar an item | — | 204 No Content |

### Annotations Endpoints

| Endpoint | Method | Action | Request Body | Returns |
|----------|--------|--------|--------------|---------|
| `/api/projects/{id}/annotations/` | GET | List all notes in project | — | `Annotation[]` |
| `/api/projects/{id}/annotations/` | POST | Add note to an item | `{ "content_type": "research_report", "object_id": "uuid", "text" }` | `Annotation` |
| `/api/projects/{id}/annotations/{annotation_id}/` | DELETE | Delete a note | — | 204 No Content |

### Comparison Endpoints

| Endpoint | Method | Action | Query Params | Returns |
|----------|--------|--------|--------------|---------|
| `/api/projects/{id}/compare/` | GET | Compare two iterations side-by-side | `a=seq1&b=seq2` | `IterationComparison` (diff of research reports) |

### Timeline Endpoints

| Endpoint | Method | Action | Returns |
|----------|--------|--------|---------|
| `/api/projects/{id}/timeline/` | GET | Get timeline view data (all iterations with key milestones) | `TimelineData` (formatted for timeline component) |

---

## Serializers (`serializers.py`)

| Serializer | Model | Nested Objects | Purpose |
|------------|-------|-----------------|---------|
| `ProjectListItemSerializer` | Project | Minimal nesting | For list view (lightweight) |
| `ProjectDetailSerializer` | Project | `iterations`, `work_products`, `annotations` | Full project with all nested objects |
| `IterationSerializer` | Iteration | `research_job` (if exists), linked `ResearchJob.report` + all nested research data | Iteration with full research context |
| `WorkProductSerializer` | WorkProduct | Generic FK to content object | Starred item with content |
| `AnnotationSerializer` | Annotation | Generic FK to annotated object | Note with target object |
| `IterationComparisonSerializer` | (custom) | Two iterations' reports | Side-by-side diff data |

---

## Context Accumulation Flow

```
Project context_mode: 'accumulate'
     ↓
Iteration 1 completes
├─ ResearchReport1 created (company overview, pain points, opportunities, etc.)
├─ GapAnalysis1 created
└─ InternalOpsIntelligence1 created
     ↓
User triggers Iteration 2
     ↓
ContextAccumulator.get_context_from_previous_iteration(iteration_2)
     ↓
Extracts key insights from Iteration1's ResearchReport1:
├─ pain_points
├─ opportunities
├─ strategic_goals
├─ digital_maturity
├─ ai_adoption_stage
└─ key_initiatives
     ↓
Constructs accumulated_context narrative: "Based on previous research...
   Key pain points identified: [...]
   Strategic goals: [...]
   Digital maturity: [...]"
     ↓
ContextAccumulator.inject_into_prompt(prompt, context)
     ↓
New prompt for Iteration2:
"[DEFAULT_PROMPT]

PREVIOUS RESEARCH CONTEXT:
{accumulated_context}"
     ↓
Iteration2 research job starts with enriched prompt
     ↓
Iteration2 ResearchReport2 created (considers prior findings)
```

---

## Data Flow Example

```
1. User creates project "Acme Corp - 2026 Engagement"
   ├─ Project (name, client_name, context_mode='accumulate')
   └─ No iterations yet
     ↓
2. User clicks "Start First Iteration"
   ├─ Creates Iteration 1 (sequence=1, name='Initial Research')
   ├─ POST /api/projects/{id}/iterations/{seq}/start/
   ├─ Creates ResearchJob, links to Iteration
   └─ Async research begins (same pipeline as single-job research)
     ↓
3. Iteration 1 completes
   ├─ ResearchReport1, GapAnalysis1, CompetitorCaseStudy1[] created
   ├─ Frontend polls, shows research results
   ├─ User reviews findings
   └─ User stars a use case to Work Products sidebar
     ↓
4. User clicks "New Iteration"
   ├─ Creates Iteration 2 (sequence=2, name='Dive into AI gap')
   ├─ ContextAccumulator pulls key insights from Iteration1
   ├─ accumulated_context field populated
   └─ POST /api/projects/{id}/iterations/{seq}/start/
     ↓
5. Iteration 2 research starts with enriched prompt
   ├─ ResearchJob linked to Iteration2
   ├─ Prompt includes accumulated context
   └─ Async research completes
     ↓
6. Iteration 2 results appear
   ├─ Different focus, but informed by Iteration1
   ├─ User can side-by-side compare Iteration1 vs Iteration2
   └─ GET /api/projects/{id}/compare/?a=1&b=2
     ↓
7. UI shows diff:
   ├─ What changed in pain points
   ├─ What changed in opportunities
   ├─ New insights found in Iteration2
   └─ What stayed the same
```

---

## Frontend Integration (Status: Complete & Functional)

**What's wired up:**
- Project list page with create button
- Project detail page showing all iterations
- Iteration results displayed via same ResearchResults component as single-job research
- Work Products sidebar (uses GenericForeignKey to display any starred item)
- Annotations system for user notes
- Iteration comparison view (diff rendering)
- Timeline view

**What's missing:**
- No ideation section (use cases, plays) — Backend complete, UI zero (but can be starred)
- No asset generation (personas, one-pagers, account plans) — Backend complete, UI zero (but can be starred)
- No memory browser — Auto-captures, UI invisible

---

## Polymorphic GenericForeignKey Pattern

Work Products and Annotations use Django's ContentType framework for polymorphism:

```python
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType

class WorkProduct(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    object_id = models.CharField(max_length=36)  # UUID string
    content_object = GenericForeignKey('content_type', 'object_id')
    # ...

# Usage:
# Star a UseCase
work_product = WorkProduct.objects.create(
    project=my_project,
    content_type=ContentType.objects.get_for_model(UseCase),
    object_id=use_case.id,
    title=use_case.title,
    summary=use_case.description,
    product_type='use_case',
)

# Retrieve
work_product.content_object  # Returns the UseCase instance
```

---

## Related Areas

- **Research App** (`docs/CODEMAPS/research.md`) — Provides ResearchJob that Iteration links to
- **Ideation App** (`docs/CODEMAPS/ideation.md`) — Generated use cases can be starred to Work Products
- **Assets App** (`docs/CODEMAPS/assets.md`) — Generated personas, one-pagers, account plans can be starred
- **Memory App** (`docs/CODEMAPS/memory.md`) — Provides context for ContextAccumulator; could auto-retrieve prior research

---

## Testing & Development

```bash
# Backend setup
cd backend
source venv/bin/activate
python manage.py runserver

# Create project
curl -X POST http://localhost:8000/api/projects/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Project",
    "client_name": "Test Corp",
    "description": "Test project description",
    "context_mode": "accumulate"
  }'

# List projects
curl http://localhost:8000/api/projects/

# Create iteration
curl -X POST http://localhost:8000/api/projects/{project_id}/iterations/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Iteration 1",
    "sales_history": "Previous sales context"
  }'

# Start iteration research
curl -X POST http://localhost:8000/api/projects/{project_id}/iterations/{seq}/start/ \
  -H "Content-Type: application/json"

# Star item to Work Products
curl -X POST http://localhost:8000/api/projects/{project_id}/work-products/ \
  -H "Content-Type: application/json" \
  -d '{
    "content_type": "use_case",
    "object_id": "use-case-uuid",
    "title": "Use Case Title",
    "summary": "Brief summary",
    "product_type": "use_case"
  }'

# Add annotation
curl -X POST http://localhost:8000/api/projects/{project_id}/annotations/ \
  -H "Content-Type: application/json" \
  -d '{
    "content_type": "research_report",
    "object_id": "report-uuid",
    "text": "Important note about this finding"
  }'

# Compare iterations
curl "http://localhost:8000/api/projects/{project_id}/compare/?a=1&b=2"

# Run tests
pytest projects/tests/
```

---

## Known Issues & Notes

- **GenericForeignKey queries inefficient:** To retrieve all work products with their content objects requires N+1 queries. Consider prefetching or caching.
- **Content types brittle:** If a model is renamed or moved, ContentType references break. Document model naming conventions.
- **No transaction management:** Cascade deletes can delete research data if project is deleted. Consider soft deletes or cascade=SET_NULL for critical relations.
- **Context accumulation simplistic:** Just concatenates prior insights into prompt; no semantic ranking or keyword extraction for relevance.
- **Iteration sequence auto-increment:** Uses PositiveInteger, not Django's built-in sequence. Manual management required if editing/deleting iterations.

---

## Database Schema Notes

```sql
-- Key indices for performance
CREATE INDEX idx_project_client_name ON projects_project(client_name);
CREATE INDEX idx_iteration_project_sequence ON projects_iteration(project_id, sequence);
CREATE INDEX idx_work_product_project_type ON projects_workproduct(project_id, product_type);
CREATE INDEX idx_annotation_project_content ON projects_annotation(project_id, content_type_id, object_id);
```

---

## Version History

| Date | Changes |
|------|---------|
| 2026-03-10 | Initial codemap — Project, Iteration, WorkProduct, Annotation, context accumulation, comparison |
| 2026-02-15 | Added iteration timeline view |
| 2026-01-20 | Core project workflow launched |

