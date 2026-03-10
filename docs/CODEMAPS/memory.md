# Codemap: Memory App

**App:** `backend/memory/`
**Purpose:** Persistent knowledge base backed by ChromaDB vector store for semantic search and auto-capture of research intelligence

**Last Updated:** 2026-03-10

---

## Overview

The Memory app builds a persistent, searchable knowledge base from every research job. It automatically captures insights at the end of each research pipeline and stores them as semantic embeddings in ChromaDB. Three distinct data stores:

1. **Client Profiles** — One profile per company researched, with industry, size, key contacts, summary
2. **Sales Plays** — Reusable sales plays (pitches, objection handlers, discovery questions) tagged by vertical
3. **Memory Entries** — Flexible knowledge records (research insights, deal outcomes, lessons learned)

All three are backed by ChromaDB vectors for semantic similarity search. **Backend is complete and auto-capture runs silently; UI is zero** (no memory browser, no search integration).

---

## Models (`models.py`)

### ClientProfile
| Field | Type | Purpose |
|-------|------|---------|
| `id` | UUID (PK) | Unique identifier |
| `client_name` | CharField (255, unique) | Company name — one profile per company |
| `industry` | CharField (100) | Industry vertical |
| `company_size` | CharField (50) | Size descriptor (e.g., "Enterprise", "1,000–5,000 employees") |
| `region` | CharField (100) | Geographic region (e.g., "North America", "EMEA") |
| `key_contacts` | JSONField (list) | List of {name, title, department, linkedin_url} |
| `summary` | TextField | Full-text summary used to generate embedding |
| `vector_id` | CharField (255) | ChromaDB document ID for retrieval |
| `created_at`, `updated_at` | DateTimeField | Metadata |

**Relationship:** One-to-one with Company (implicit via unique client_name)

---

### SalesPlay
| Field | Type | Purpose |
|-------|------|---------|
| `id` | UUID (PK) | Unique identifier |
| `title` | CharField (255) | Play title (e.g., "Objection Handler: Budget Constraints") |
| `play_type` | CharField (50) | Type: `pitch`, `objection_handler`, `value_proposition`, `case_study`, `competitive_response`, `discovery_question` |
| `content` | TextField | Full play content (actual text to use) |
| `context` | TextField | Guidance on when/how to use this play |
| `industry` | CharField (100) | Industry this play is for |
| `vertical` | CharField (50) | Specific vertical (e.g., "healthcare", "finance", "retail") |
| `usage_count` | IntegerField | How many times this play has been used |
| `success_rate` | FloatField (0.0–1.0) | Effectiveness score (intended for tracking over time) |
| `vector_id` | CharField (255) | ChromaDB document ID for semantic retrieval |
| `created_at`, `updated_at` | DateTimeField | Metadata |

**Promotion Flow:** Best `RefinedPlay` records (from ideation app) promoted into `SalesPlay` library for reuse.

---

### MemoryEntry
| Field | Type | Purpose |
|-------|------|---------|
| `id` | UUID (PK) | Unique identifier |
| `entry_type` | CharField (50) | Type: `research_insight`, `client_interaction`, `deal_outcome`, `best_practice`, `lesson_learned` |
| `title` | CharField (255) | Short title for the entry |
| `content` | TextField | Full text of the memory (used to generate embedding) |
| `client_name` | CharField (255) | Company this relates to (if applicable) |
| `industry` | CharField (100) | Industry context |
| `tags` | JSONField (list) | Free-form tags for filtering (e.g., ["AI adoption", "cost pressure"]) |
| `source_type` | CharField (50) | Where it came from (e.g., "research_job", "manual", "deal_review") |
| `source_id` | CharField (255) | ID of source record (ResearchJob UUID, deal ID, etc.) |
| `vector_id` | CharField (255) | ChromaDB document ID for semantic retrieval |
| `created_at`, `updated_at` | DateTimeField | Metadata |

**Purpose:** Flexible store for any knowledge — auto-captured from research, or manually entered.

---

## Services (`services/`)

| Service | File | Key Functions | Purpose |
|---------|------|----------------|---------|
| **VectorStore** | `vectorstore.py` | `init_client()`, `add_document(collection, text, metadata)`, `query(collection, query_text, top_k)`, `delete_document(collection, vector_id)` | ChromaDB operations — creates/manages collections, embeddings, semantic search |
| **MemoryCapture** | `capture.py` | `capture_from_research(research_job)` | Auto-run at end of research pipeline; creates ClientProfile, MemoryEntry records from ResearchReport data |
| **ContextRetriever** | `context.py` | `get_context_for_company(client_name, query, top_k=5)` | Query memory store for prior intelligence on a company |
| **PlayLibraryManager** | `play_library.py` | `promote_play(refined_play)`, `query_plays(vertical, play_type, query_text)` | Manage SalesPlay library; promote best RefinedPlay; semantic search |

---

## Views & API (`views.py`, `urls.py`)

### ClientProfile Endpoints

| Endpoint | Method | Action | Returns |
|----------|--------|--------|---------|
| `/api/memory/profiles/` | GET | List all client profiles | `ClientProfile[]` |
| `/api/memory/profiles/{id}/` | GET | Retrieve single profile | `ClientProfile` |
| `/api/memory/profiles/?client_name=Acme` | GET | Search by company name | `ClientProfile[]` |

### SalesPlay Endpoints

| Endpoint | Method | Action | Query Params | Returns |
|----------|--------|--------|--------------|---------|
| `/api/memory/plays/` | GET | List all sales plays | `?vertical=healthcare&type=objection_handler` | `SalesPlay[]` |
| `/api/memory/plays/{id}/` | GET | Retrieve single play | — | `SalesPlay` |
| `/api/memory/plays/search/` | POST | Semantic search in plays library | `{ "query": "budget concerns" }` | `SalesPlay[]` (ranked by similarity) |

### MemoryEntry Endpoints

| Endpoint | Method | Action | Query Params | Returns |
|----------|--------|--------|--------------|---------|
| `/api/memory/entries/` | GET | List all memory entries | `?entry_type=research_insight&client_name=Acme` | `MemoryEntry[]` |
| `/api/memory/entries/{id}/` | GET | Retrieve single entry | — | `MemoryEntry` |
| `/api/memory/entries/search/` | POST | Semantic search in entries | `{ "query": "AI adoption challenges" }` | `MemoryEntry[]` (ranked by similarity) |

### Context Query Endpoint

| Endpoint | Method | Action | Request Body | Returns |
|----------|--------|--------|--------------|---------|
| `/api/memory/context/` | POST | Query memory for context on a company (before new research) | `{ "client_name": "Acme", "query": "prior findings" }` | `{ "profiles": ClientProfile[], "entries": MemoryEntry[] }` |

### Manual Capture Endpoint

| Endpoint | Method | Action | Returns |
|----------|--------|--------|---------|
| `/api/memory/capture/{research_job_id}/` | POST | Manually trigger capture from completed research job | `{ "created": int, "profile_updated": bool }` |

---

## Serializers (`serializers.py`)

| Serializer | Model | Purpose |
|------------|-------|---------|
| `ClientProfileSerializer` | ClientProfile | Full profile with all contacts and metadata |
| `SalesPlaySerializer` | SalesPlay | Full play with content, context, effectiveness data |
| `MemoryEntrySerializer` | MemoryEntry | Full entry with type, tags, source tracking |

---

## ChromaDB Integration

VectorStore wraps ChromaDB client with three persistent collections:

```python
# Collections (each persists independently)
1. "client_profiles" — One doc per company, vectorized from summary
2. "sales_plays" — One doc per reusable play, vectorized from content
3. "memory_entries" — One doc per knowledge entry, vectorized from content
```

### Collection Schema Example

```python
{
  "ids": ["profile_001", "profile_002", ...],
  "embeddings": [[0.123, 0.456, ...], [0.789, ...], ...],
  "metadatas": [
    {
      "client_name": "Acme Corp",
      "industry": "Technology",
      "company_size": "Enterprise",
      "region": "North America"
    },
    ...
  ],
  "documents": [
    "Acme Corp is a global technology company...",
    ...
  ]
}
```

### Semantic Search Example

```python
# Query: "What do we know about AI adoption in Acme?"
results = vectorstore.query(
    collection="client_profiles",
    query_text="Acme AI adoption",
    top_k=3
)
# Returns: [
#   {"id": "profile_001", "score": 0.92, "text": "...", "metadata": {...}},
#   {"id": "profile_002", "score": 0.78, "text": "...", "metadata": {...}},
# ]
```

---

## Auto-Capture Flow

**Triggered at end of research pipeline** (`research/graph/nodes.py::finalize_result`):

```
ResearchJob completes (status='completed')
     ↓
finalize_result() node executes
     ↓
MemoryCapture.capture_from_research(research_job)
     ↓
1. Extract ClientProfile data:
   ├─ client_name, vertical, employee_count, headquarters (region)
   ├─ Generate summary from: overview, pain_points, opportunities, strategic_goals
   ├─ Extract key_contacts from decision_makers
   └─ Add to ChromaDB "client_profiles" collection
     ↓
2. Create MemoryEntry records:
   ├─ Type: 'research_insight'
   ├─ For each: pain_point, opportunity, strategic_goal, key_initiative
   ├─ Tag with client_name, industry, relevant keywords
   ├─ Source: research_job
   └─ Add to ChromaDB "memory_entries" collection
     ↓
3. Update ClientProfile if company previously researched:
   ├─ Check unique constraint on client_name
   ├─ If found: update summary, key_contacts, industry (if changed)
   ├─ If not found: create new profile
     ↓
4. Silent completion (no UI notification)
```

---

## Data Flow Example: Context-Aware Research

```
1. Rep starting new research on "Acme Corp" again
   ├─ Navigate to /projects/new
   ├─ Enter client_name: "Acme Corp"
     ↓
2. (Future UI) Context search bar suggests:
   "We've researched Acme before. Load prior context?"
   ├─ POST /api/memory/context/
   ├─ Request: { "client_name": "Acme Corp", "query": "prior findings" }
     ↓
3. API queries ChromaDB:
   ├─ Find ClientProfile for "Acme Corp"
   ├─ Find MemoryEntry records tagged with "Acme Corp"
   ├─ Rank by relevance, return top 5
     ↓
4. Response:
   {
     "profiles": [
       {
         "client_name": "Acme Corp",
         "industry": "Technology",
         "summary": "Global tech firm, 5,000+ employees...",
         "key_contacts": [...]
       }
     ],
     "entries": [
       {
         "title": "AI adoption: Pilot stage",
         "content": "Acme running LLM pilots in...",
         "tags": ["AI adoption", "pilot"],
         "created_at": "2026-02-15"
       },
       ...
     ]
   }
     ↓
5. UI displays prior context to rep:
   ├─ "Acme Corp — Technology, Enterprise"
   ├─ Key prior findings: [list of MemoryEntry titles]
   ├─ Previous contacts: [list of decision_makers]
     ↓
6. Rep starts new research
   ├─ System injects accumulated context into prompt:
      "Based on previous research conducted on 2026-02-15:
       Key findings: [...]
       Prior pain points: [...]
       Digital maturity: Advanced"
     ↓
7. New research job runs with enriched prompt
     ↓
8. New findings captured (auto-run)
   ├─ ClientProfile for Acme updated
   ├─ New MemoryEntry records created
```

---

## Play Library Workflow

```
1. RefinedPlay generated from ideation pipeline
   ├─ Title: "Objection Handler: Budget Constraints"
   ├─ Content: "Here's how to position ROI..."
   ├─ Status: 'draft'
     ↓
2. Sales leadership reviews play
   ├─ Validates messaging quality
   ├─ Tests in sales conversations
   ├─ Marks as 'approved'
     ↓
3. (Future UI) "Add to Play Library" button
   ├─ POST /api/memory/plays/promote/
   ├─ Request: { "refined_play_id": "uuid" }
     ↓
4. PlayLibraryManager.promote_play(refined_play)
   ├─ Create SalesPlay record
   ├─ Copy content from RefinedPlay
   ├─ Set vertical, play_type, initial success_rate
   ├─ Add to ChromaDB "sales_plays" collection
   └─ Link original RefinedPlay for tracking
     ↓
5. Play now in library, available for reuse
   ├─ GET /api/memory/plays/?vertical=healthcare&type=objection_handler
   ├─ POST /api/memory/plays/search/ with "budget concerns"
   └─ Other reps can find and reuse
```

---

## Frontend Integration (Status: Zero UI)

**Backend Complete:**
- All endpoints exist
- ChromaDB auto-capture running silently
- Semantic search working
- Play library infrastructure ready

**Frontend Missing (per `TODO.md`):**
- No memory browser page anywhere
- No search interface for memory entries or plays
- No client profile viewer
- No context pre-population in research form
- No "Add to Play Library" button on RefinedPlay
- No API calls in `frontend/lib/api.ts` for any `/api/memory/` endpoints
- No TypeScript types for memory models in `frontend/types/index.ts`
- No visibility into auto-capture (runs silently)

**UI Build-Out Needed (per `TODO.md`):**
1. Add API client methods in `frontend/lib/api.ts`
2. Add TypeScript types in `frontend/types/index.ts`
3. Build Memory Browser page with three tabs:
   - **Client Profiles** — Searchable list of all researched companies
   - **Sales Play Library** — Filterable/searchable plays by type and vertical
   - **Memory Entries** — Searchable feed of all captured insights
4. Add context search bar to research creation form
5. Pre-populate research with prior context if available
6. Add "Add to Play Library" button on RefinedPlay cards
7. Show memory capture confirmation in research status pipeline

---

## Related Areas

- **Research App** (`docs/CODEMAPS/research.md`) — Triggers auto-capture at end of pipeline
- **Ideation App** (`docs/CODEMAPS/ideation.md`) — RefinedPlay records promoted to SalesPlay library
- **Projects App** (`docs/CODEMAPS/projects.md`) — Could use context retrieval for iteration accumulation
- **ContextAccumulator** (projects/services/context.py) — Could use memory context for enrichment

---

## Testing & Development

```bash
# Backend setup
cd backend
source venv/bin/activate
export CHROMA_PERSIST_DIR="./chroma_data"  # Optional: enable persistence
python manage.py runserver

# List client profiles
curl http://localhost:8000/api/memory/profiles/

# Search profiles by company name
curl "http://localhost:8000/api/memory/profiles/?client_name=Acme"

# Query context before new research
curl -X POST http://localhost:8000/api/memory/context/ \
  -H "Content-Type: application/json" \
  -d '{"client_name": "Acme Corp", "query": "prior findings"}'

# List sales plays
curl http://localhost:8000/api/memory/plays/?vertical=healthcare

# Semantic search in plays
curl -X POST http://localhost:8000/api/memory/plays/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "budget concerns"}'

# List memory entries
curl http://localhost:8000/api/memory/entries/

# Search memory entries
curl -X POST http://localhost:8000/api/memory/entries/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "AI adoption challenges"}'

# Manually trigger capture
curl -X POST http://localhost:8000/api/memory/capture/{research_job_id}/ \
  -H "Content-Type: application/json"

# Run tests
pytest memory/tests/

# View ChromaDB files (if persistence enabled)
ls -la ./chroma_data/
```

---

## Known Issues & Notes

- **Silent auto-capture:** No UI confirmation or progress indication. User doesn't know what's being captured.
- **ChromaDB persistence:** By default, in-memory only. To persist across server restarts, set `CHROMA_PERSIST_DIR` env var.
- **Similarity scoring opaque:** ChromaDB's default embedding model (Sentence Transformers) produces scores between 0–1; meaning is not always obvious.
- **No duplicate detection:** If same company researched multiple times, multiple ClientProfile + MemoryEntry records created. Consider deduplication logic.
- **Promotion flow manual:** No automatic "top plays" aggregation. Promotion to SalesPlay is manual via API or future UI.
- **No temporal relevance:** Older entries get same search weight as recent. Consider recency boosting.
- **Scaling concerns:** As MemoryEntry count grows (thousands of entries), ChromaDB queries may slow. Consider implementing archival/pruning.

---

## Database Schema Notes

```python
# ChromaDB collections (in addition to Django models)
client_profiles_collection = client.create_collection(name="client_profiles")
sales_plays_collection = client.create_collection(name="sales_plays")
memory_entries_collection = client.create_collection(name="memory_entries")

# Each has embeddings, documents, metadatas, and ids
```

---

## Version History

| Date | Changes |
|------|---------|
| 2026-03-10 | Initial codemap — ClientProfile, SalesPlay, MemoryEntry, auto-capture, semantic search |
| 2026-01-30 | Memory system launched with ChromaDB backend |

