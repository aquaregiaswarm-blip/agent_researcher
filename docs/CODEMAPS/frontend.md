# Codemap: Frontend (Next.js)

**Framework:** Next.js 14 + React 18 + TypeScript + Tailwind CSS
**Location:** `frontend/`
**Purpose:** Customer-facing UI for research job creation, project management, results visualization, and work products

**Last Updated:** 2026-03-10

---

## Overview

The frontend is a Next.js 14 application using App Router. It implements two main workflows:

1. **Quick Research** — Single-shot research job on home page (`/`)
2. **Project-Based Research** — Multi-iteration research within projects (`/projects/*`)

Most backend features (ideation, assets, memory) have **zero UI implementation** — endpoints exist but are never called from the frontend.

---

## Project Structure

```
frontend/
├── app/                              # Next.js App Router pages
│   ├── layout.tsx                   # Root layout with Navigation
│   ├── page.tsx                     # Home page (quick research)
│   ├── research/
│   │   ├── page.tsx                 # Research job list (rarely used)
│   │   └── [id]/
│   │       └── page.tsx             # Research job detail + results
│   └── projects/
│       ├── page.tsx                 # Project list
│       ├── new/
│       │   └── page.tsx             # Create new project
│       └── [id]/
│           ├── page.tsx             # Project dashboard
│           └── iterate/
│               └── page.tsx         # Start new iteration
├── components/
│   ├── ResearchForm.tsx             # Form to create research job
│   ├── ResearchResults.tsx          # Tabbed results view (Overview, Deep Research, etc.)
│   ├── Navigation.tsx               # Top nav bar
│   ├── projects/
│   │   ├── ProjectList.tsx          # Project cards grid
│   │   ├── ProjectDetail.tsx        # Project dashboard with sidebar
│   │   ├── IterationTimeline.tsx    # Timeline view of iterations
│   │   ├── ComparisonView.tsx       # Side-by-side iteration diff
│   │   ├── WorkProductsSidebar.tsx  # Starred items panel
│   │   └── AnnotationPanel.tsx      # Notes panel
│   └── common/
│       ├── Button.tsx
│       ├── Card.tsx
│       ├── Modal.tsx
│       ├── Loading.tsx
│       └── StarButton.tsx           # Star/unstar button (built but not placed)
├── lib/
│   └── api.ts                       # API client (fetch wrapper + all endpoints)
├── types/
│   └── index.ts                     # TypeScript interfaces
├── styles/
│   └── globals.css                  # Tailwind + global styles
├── next.config.js                   # Next.js config
└── package.json                     # Dependencies (Next.js, React, TypeScript, Tailwind, etc.)
```

---

## Pages & Routes

### Home Page (`/`)
**Component:** `app/page.tsx` + `components/ResearchForm.tsx`

**What it does:**
- Displays quick research form (client name + optional sales history)
- Submits to POST `/api/research/`
- Redirects to `/research/[id]` after job creation
- Shows loading state while research runs

**UI Elements:**
- ResearchForm component (text inputs, submit button)
- Redirect to research detail on submit

---

### Research Detail (`/research/[id]`)
**Component:** `app/research/[id]/page.tsx` + `components/ResearchResults.tsx`

**What it does:**
- Polls `/api/research/{id}/` for job status
- Displays tabs when completed:
  - **Overview** — Quick stats (company, founded year, employees, revenue, digital maturity, decision makers)
  - **Deep Research** — Full details (headquarters, website, strategic goals, AI adoption, key initiatives)
  - **Competitors** — Competitor case studies (3–5 cards with relevance score)
  - **Gap Analysis** — Technology, capability, process gaps with recommendations
  - **Inside Intel** — Internal ops intelligence (employee sentiment, hiring, LinkedIn, social, news, gap correlations)
  - **Sources** — Web sources from Google Search grounding
  - **Raw Output** — Full plain-text research report

**UI Elements:**
- Tab navigation (Overview, Deep Research, Competitors, Gap Analysis, Inside Intel, Sources, Raw Output)
- Status badge (Pending, Running, Completed, Failed)
- PDF download button
- Loading spinners during polling
- Error display

**Missing Features:**
- No "Generate Use Cases" button
- No "Generate Personas" button
- No "Generate One-Pager" button
- No "Generate Account Plan" button
- No ideation section tab
- No asset generation buttons
- StarButton not placed anywhere

---

### Projects List (`/projects`)
**Component:** `app/projects/page.tsx` + `components/projects/ProjectList.tsx`

**What it does:**
- Lists all projects (paginated, ordered by -updated_at)
- Displays project cards with:
  - Project name
  - Client name
  - Description snippet
  - Last updated date
  - Iteration count
  - Create/Edit/Delete buttons
- "New Project" button redirects to `/projects/new`

**UI Elements:**
- Project card grid layout
- Search/filter by client name or project name (optional)
- Pagination controls
- "New Project" button

---

### Create Project (`/projects/new`)
**Component:** `app/projects/new/page.tsx`

**What it does:**
- Form to create new project
- Fields:
  - Project name (required)
  - Client name (required)
  - Description (optional)
  - Context mode selector (Build on Previous vs Fresh Start)
- Submits to POST `/api/projects/`
- Redirects to project detail on success

**UI Elements:**
- Form inputs with validation
- Radio buttons for context_mode
- Submit and cancel buttons
- Error handling

---

### Project Dashboard (`/projects/[id]`)
**Component:** `app/projects/[id]/page.tsx` + `components/projects/ProjectDetail.tsx`

**What it does:**
- Main project view with:
  - Project header (name, client, context mode, description)
  - Iteration timeline (all iterations with status badges)
  - Research results area (same ResearchResults tabs as single-job research)
  - Work Products sidebar (starred items)
  - Annotations sidebar (notes)
  - Comparison button (if multiple iterations exist)

**Left Sidebar (Work Products):**
- Title: "Saved Items"
- Grouped by product_type: Use Cases, Plays, Personas, One-Pagers, Gaps, Insights
- Each item shows: title, summary preview, created date
- Hover: show star icon to unstar
- Click: (future) navigate to full item view or expand inline

**Right Sidebar (Annotations):**
- Title: "Notes"
- List of all annotations on this project
- Each shows: text, created date, delete button
- Input to add new annotation

**UI Elements:**
- Header with project info
- Iteration timeline (horizontal scroll or vertical list)
- "Start New Iteration" button
- Tabs for iteration results (Overview, Deep Research, Competitors, Gap Analysis, Inside Intel, Sources, Raw Output)
- Two sidebars (left: work products, right: annotations)
- Loading states

---

### Iterate (`/projects/[id]/iterate`)
**Component:** `app/projects/[id]/iterate/page.tsx`

**What it does:**
- Form to create new iteration on a project
- Fields:
  - Iteration name (optional label like "v2 - Focus on AI")
  - Sales history / context update
  - Prompt override (optional additional guidance)
- If context_mode='accumulate': shows "Context will be injected from previous iteration"
- Submits to POST `/api/projects/{id}/iterations/`
- Then calls POST `/api/projects/{id}/iterations/{seq}/start/` to trigger research
- Redirects to project detail, starts polling

**UI Elements:**
- Form inputs
- Context preview (if accumulate mode)
- Submit button
- Cancel button

---

## Components

### ResearchForm (`components/ResearchForm.tsx`)

**Props:**
- `onSubmit: (data) => void`
- `isLoading: boolean`

**What it does:**
- Text input for client_name
- TextArea for sales_history (optional)
- Submit button (disabled while loading)
- Error display

**Uses:**
- React Hook Form for form management
- API client to POST `/api/research/`

---

### ResearchResults (`components/ResearchResults.tsx`)

**Props:**
- `research: ResearchJob` (includes nested report, competitors, gaps, internal_ops, correlations)
- `isLoading: boolean`
- `onRefresh: () => void`

**What it does:**
- Renders 8 tabs: Overview, Deep Research, Competitors, Gap Analysis, Inside Intel, Sources, Raw Output
- Each tab displays research data in a structured format
- Shows status badge (Pending, Running, Completed, Failed)
- Shows PDF download button
- Polling hook to refresh job status every 2 seconds until completed

**Tabs:**

| Tab | Component | Data Source | UI |
|-----|-----------|-------------|-----|
| **Overview** | OverviewTab | ResearchReport fields | Quick stats cards (company, founder year, employees, revenue, digital maturity), Decision makers list, Pain points, Opportunities |
| **Deep Research** | DeepResearchTab | ResearchReport full fields | Detailed narrative sections (headquarters, website, digital maturity, AI adoption, strategic goals, key initiatives) |
| **Competitors** | CompetitorsTab | CompetitorCaseStudy[] | 3–5 cards with company, vertical, case study title, technologies, outcomes, relevance score % |
| **Gap Analysis** | GapAnalysisTab | GapAnalysis | Priority areas list, Technology gaps (red), Capability gaps (orange), Process gaps (purple), Recommendations, confidence % |
| **Inside Intel** | InsideIntelTab | InternalOpsIntelligence + GapCorrelation[] | 6 panels (employee sentiment, job postings, LinkedIn, social media, news, gap correlation insights) + footer metadata |
| **Sources** | SourcesTab | ResearchReport.web_sources | List of URIs with titles, clickable links |
| **Raw Output** | RawOutputTab | ResearchJob.result | Plain text rendering |

---

### Navigation (`components/Navigation.tsx`)

**What it does:**
- Top nav bar with logo
- Links: Home, Projects, (future: Memory, Ideation)
- User menu (if auth implemented)

**Links:**
- Home → `/`
- Projects → `/projects`

---

### ProjectList (`components/projects/ProjectList.tsx`)

**Props:**
- `projects: ProjectListItem[]`
- `onDelete: (id) => void`
- `onEdit: (id) => void`

**What it does:**
- Grid of project cards
- Each card shows: project name, client name, description snippet, last updated, iteration count
- Buttons: View, Edit, Delete
- "New Project" button

---

### ProjectDetail (`components/projects/ProjectDetail.tsx`)

**Props:**
- `project: Project` (includes iterations, work_products, annotations)

**What it does:**
- Header with project info
- Iteration timeline (IterationTimeline component)
- Research results area (ResearchResults for latest iteration)
- Two sidebars: WorkProductsSidebar (left), AnnotationPanel (right)

**Uses:**
- IterationTimeline component
- ResearchResults component
- WorkProductsSidebar component
- AnnotationPanel component

---

### IterationTimeline (`components/projects/IterationTimeline.tsx`)

**Props:**
- `iterations: Iteration[]`
- `selectedSequence: number`
- `onSelect: (sequence) => void`

**What it does:**
- Horizontal or vertical timeline of all iterations
- Each iteration shows: sequence number, name, status badge, created date
- Click to select iteration and show its results
- "Start New Iteration" button

---

### ComparisonView (`components/projects/ComparisonView.tsx`)

**Props:**
- `comparison: IterationComparison`
- `iterationA: Iteration`
- `iterationB: Iteration`

**What it does:**
- Side-by-side comparison of two iterations
- Highlights what changed:
  - Pain points: added/removed/unchanged
  - Opportunities: added/removed/unchanged
  - Digital maturity: before/after
  - Key insights: new findings
- Uses color coding (green=new, red=removed, gray=unchanged)

---

### WorkProductsSidebar (`components/projects/WorkProductsSidebar.tsx`)

**Props:**
- `workProducts: WorkProduct[]`
- `onUnstar: (id) => void`

**What it does:**
- Displays all starred items grouped by product_type
- Groups: Use Cases, Plays, Personas, One-Pagers, Gaps, Insights
- Each item shows: title, summary preview, created date
- Hover to show delete button
- (Future) Click to expand inline or navigate to full view

**Missing:**
- Not populated (no API calls to create work products)
- No ability to star items from research results

---

### AnnotationPanel (`components/projects/AnnotationPanel.tsx`)

**Props:**
- `annotations: Annotation[]`
- `onAdd: (text) => void`
- `onDelete: (id) => void`

**What it does:**
- List of all notes on this project
- Each note shows: text, created date, author (if tracked), delete button
- Input at bottom to add new note

**UI:**
- Textarea input
- "Add Note" button
- List of previous notes with delete buttons

---

### StarButton (`components/common/StarButton.tsx`)

**Props:**
- `item: any` (UseCase, Persona, OnePager, etc.)
- `isStarred: boolean`
- `onToggle: () => void`

**What it does:**
- Star/unstar icon button
- Shows filled star if starred, outline if not
- Calls onToggle when clicked

**Status:** Built but never placed anywhere on the page. Not in research results, not in ideation section, not in asset section.

---

## TypeScript Types (`frontend/types/index.ts`)

### Current Types Defined:

```typescript
// Research types
interface ResearchJob {
  id: string;
  client_name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  result: string;
  error: string;
  created_at: string;
  updated_at: string;
  report?: ResearchReport;
  vertical?: string;
}

interface ResearchReport {
  id: string;
  company_overview: string;
  founded_year?: number;
  headquarters: string;
  // ... more fields
}

interface CompetitorCaseStudy {
  id: string;
  competitor_name: string;
  // ... more fields
}

interface GapAnalysis {
  id: string;
  technology_gaps: string[];
  // ... more fields
}

// Internal operations intelligence
interface InternalOpsIntelligence {
  id: string;
  employee_sentiment: EmployeeSentiment;
  // ... more fields
}

interface GapCorrelation {
  id: string;
  gap_type: string;
  // ... more fields
}

// Project types
interface Project {
  id: string;
  name: string;
  client_name: string;
  context_mode: 'accumulate' | 'fresh';
  iterations?: Iteration[];
  work_products?: WorkProduct[];
  annotations?: Annotation[];
  // ... more fields
}

interface Iteration {
  id: string;
  sequence: number;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  research_job?: ResearchJob;
  // ... more fields
}

interface WorkProduct {
  id: string;
  project_id: string;
  title: string;
  summary: string;
  product_type: 'use_case' | 'play' | 'persona' | 'one_pager' | 'account_plan' | 'gap' | 'insight';
  // ... more fields
}

interface Annotation {
  id: string;
  project_id: string;
  text: string;
  content_type: string;
  object_id: string;
  created_at: string;
}
```

### Missing Types (Backend-complete, UI zero):

```typescript
// Ideation types (not defined)
interface UseCase { /* ... */ }
interface FeasibilityAssessment { /* ... */ }
interface RefinedPlay { /* ... */ }

// Assets types (not defined)
interface Persona { /* ... */ }
interface OnePager { /* ... */ }
interface AccountPlan { /* ... */ }
interface Citation { /* ... */ }

// Memory types (not defined)
interface ClientProfile { /* ... */ }
interface SalesPlay { /* ... */ }
interface MemoryEntry { /* ... */ }
```

---

## API Client (`frontend/lib/api.ts`)

### Methods Implemented

**Research endpoints:**
- `listResearch()` → GET `/api/research/jobs/`
- `createResearch(data)` → POST `/api/research/`
- `executeResearch(id)` → POST `/api/research/{id}/execute/`
- `getResearch(id)` → GET `/api/research/{id}/`
- `downloadResearchPdf(id)` → GET `/api/research/{id}/export/pdf/` (downloads blob)
- `pollResearch(id, onUpdate, intervalMs)` → Polls until completed/failed

**Prompt endpoints:**
- `getDefaultPrompt()` → GET `/api/prompts/default/`
- `updateDefaultPrompt(content)` → PUT `/api/prompts/default/`

**Project endpoints:**
- `listProjects()` → GET `/api/projects/`
- `getProject(id)` → GET `/api/projects/{id}/`
- `createProject(data)` → POST `/api/projects/`
- `updateProject(id, data)` → PUT `/api/projects/{id}/`
- `deleteProject(id)` → DELETE `/api/projects/{id}/`

**Iteration endpoints:**
- `listIterations(projectId)` → GET `/api/projects/{id}/iterations/`
- `createIteration(projectId, data)` → POST `/api/projects/{id}/iterations/`
- `getIteration(projectId, sequence)` → GET `/api/projects/{id}/iterations/{seq}/`
- `startIteration(projectId, sequence)` → POST `/api/projects/{id}/iterations/{seq}/start/`

**Work Products endpoints:**
- `listWorkProducts(projectId, productType?)` → GET `/api/projects/{id}/work-products/`
- `createWorkProduct(projectId, data)` → POST `/api/projects/{id}/work-products/`
- `deleteWorkProduct(projectId, productId)` → DELETE `/api/projects/{id}/work-products/{id}/`

**Annotations endpoints:**
- `listAnnotations(projectId)` → GET `/api/projects/{id}/annotations/`
- `createAnnotation(projectId, data)` → POST `/api/projects/{id}/annotations/`
- `deleteAnnotation(projectId, annotationId)` → DELETE `/api/projects/{id}/annotations/{id}/`

**Comparison endpoints:**
- `compareIterations(projectId, seqA, seqB)` → GET `/api/projects/{id}/compare/?a={a}&b={b}`

**Timeline endpoints:**
- `getTimeline(projectId)` → GET `/api/projects/{id}/timeline/`

### Methods NOT Implemented (Missing)

**Ideation endpoints:**
- `generateUseCases(researchJobId)`
- `listUseCases(researchJobId?)`
- `getUseCase(id)`
- `assessFeasibility(useCaseId)`
- `refinePlay(useCaseId)`

**Assets endpoints:**
- `generatePersonas(researchJobId)`
- `listPersonas(researchJobId?)`
- `getPersona(id)`
- `generateOnePager(researchJobId, useCaseId?)`
- `getOnePager(id)`
- `getOnePagerHtml(id)`
- `generateAccountPlan(researchJobId)`
- `getAccountPlan(id)`
- `getAccountPlanHtml(id)`
- `listCitations(researchJobId?)`
- `getCitation(id)`

**Memory endpoints:**
- `listClientProfiles()`
- `getClientProfile(id)`
- `listSalesPlays(vertical?, type?)`
- `getSalesPlay(id)`
- `queryContextBefore(clientName, query)`
- `searchPlays(query)`
- `listMemoryEntries(type?, clientName?)`
- `getMemoryEntry(id)`
- `searchMemoryEntries(query)`
- `captureFromResearchJob(id)`

---

## Styling & Tailwind

**Tailwind CSS Configuration:**
- Color palette: Slate/Blue primary colors
- Responsive breakpoints: Mobile-first
- Custom utilities for cards, buttons, badges

**Global Styles (`frontend/styles/globals.css`):**
- Tailwind imports
- Custom component classes (.card, .btn, .badge, etc.)
- Dark mode support (optional)

---

## Frontend Features Status

| Feature | Backend | Frontend | Status |
|---------|---------|----------|--------|
| Quick Research (single job) | ✅ | ✅ | **Live** |
| Project-based research | ✅ | ✅ | **Live** |
| Iteration management | ✅ | ✅ | **Live** |
| Context accumulation | ✅ | ✅ | **Live** |
| Iteration comparison | ✅ | ✅ | **Live** |
| Work Products sidebar | ✅ | ✅ | **Live** |
| Annotations / Notes | ✅ | ✅ | **Live** |
| Research results tabs | ✅ | ✅ | **Live** |
| PDF export | ✅ | ✅ | **Live** |
| Use Case Generation | ✅ | ❌ | **Dark** |
| Feasibility Assessment | ✅ | ❌ | **Dark** |
| Refined Sales Plays | ✅ | ❌ | **Dark** |
| Buyer Personas | ✅ | ❌ | **Dark** |
| One-Pager Generator | ✅ | ❌ | **Dark** |
| Account Plan Generator | ✅ | ❌ | **Dark** |
| Citations / Sources tab | ✅ | ⚠️ | **Partial** (shows web_sources, not Citations) |
| Memory Browser | ✅ | ❌ | **Dark** |
| Context search (before research) | ✅ | ❌ | **Dark** |

---

## Development Setup

```bash
cd frontend

# Install dependencies
npm install

# Development server (http://localhost:3000)
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Linting
npm run lint

# Tests (Vitest)
npm run test
npm run test:ui  # With UI

# Environment variables (.env.local)
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Known Issues & Limitations

- **StarButton orphaned:** Component built but never placed anywhere
- **No type definitions for ideation/assets/memory:** Can't develop UI without manual type definitions
- **Polling hardcoded to 2s intervals:** No configurable polling strategy
- **Work Products sidebar basic:** Doesn't expand to full content, no navigation to asset detail pages
- **ResearchResults tabs monolithic:** Should be split into separate components for maintainability
- **No error boundary:** App crashes if API calls fail unexpectedly
- **No authentication:** All endpoints accessible without auth
- **No mobile optimization:** Responsive classes exist but not thoroughly tested

---

## Next Steps (per `TODO.md`)

1. Add ideation API methods to `api.ts`
2. Add asset generation API methods to `api.ts`
3. Add memory API methods to `api.ts`
4. Define TypeScript types for all missing models
5. Build ideation section UI (use cases, plays, feasibility)
6. Build assets section UI (personas, one-pagers, account plans)
7. Build memory browser page
8. Place StarButton on all asset cards
9. Expand Work Products sidebar to show full content
10. Add context search bar to research creation form

---

## Version History

| Date | Changes |
|------|---------|
| 2026-03-10 | Initial codemap — all implemented features documented, missing UI listed |
| 2026-02-20 | Iteration comparison and timeline views added |
| 2026-02-01 | Project workflow launched |
| 2026-01-15 | Quick research home page launched |

