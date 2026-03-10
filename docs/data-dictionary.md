# Deep Prospecting Engine - Database Data Dictionary

**Generated:** 2026-03-10
**Project:** Deep Prospecting Engine
**Database:** PostgreSQL (via Django ORM)
**Django Version:** 5.x / 6.x

---

## Table of Contents

1. [Overview](#overview)
2. [Database Architecture](#database-architecture)
3. [Detailed Table Documentation](#detailed-table-documentation)
   - [Research App](#research-app)
   - [Projects App](#projects-app)
   - [Ideation App](#ideation-app)
   - [Assets App](#assets-app)
   - [Memory App](#memory-app)
   - [Prompts App](#prompts-app)
4. [Cross-Table Analysis](#cross-table-analysis)
5. [JSONField Audit](#jsonfield-audit)
6. [Index Audit](#index-audit)
7. [Enum/Choice Fields](#enumchoice-fields)
8. [Nullable Field Analysis](#nullable-field-analysis)
9. [Data Completeness Assessment](#data-completeness-assessment)
10. [Schema Concerns](#schema-concerns)

---

## Overview

The Deep Prospecting Engine is an AI-powered sales research system using Django + DRF backend with Google Gemini for AI processing and LangGraph for workflow orchestration. The database stores research jobs, structured reports, generated assets, and knowledge memory across six Django apps.

**Primary Key Strategy:** UUID v4 for all tables (except `prompts.PromptTemplate` which uses Django's default BigAutoField)

**Timestamp Strategy:** All models include `created_at` (auto_now_add) and `updated_at` (auto_now) except where noted

**Relationships:** Predominantly cascade deletes (`on_delete=models.CASCADE`) with some `SET_NULL` for work product tracking

---

## Database Architecture

### Entity Relationship Overview

```
Project (1) ──── (M) Iteration (1) ──── (1) ResearchJob
                     │                        │
                     │                        ├─── (1) ResearchReport
                     │                        ├─── (M) CompetitorCaseStudy
                     │                        ├─── (1) GapAnalysis
                     │                        ├─── (1) InternalOpsIntel
                     │                        ├─── (M) UseCase
                     │                        ├─── (M) Persona
                     │                        ├─── (M) OnePager
                     │                        ├─── (1) AccountPlan
                     │                        └─── (M) Citation
                     │
                     └─── (M) WorkProduct (GenericFK to any model)
                     └─── (M) Annotation (GenericFK to any model)

UseCase (1) ──── (1) FeasibilityAssessment
           └─── (1) RefinedPlay

ClientProfile ──── (indexed by client_name, no FK)
SalesPlay ──── (no FK relationships)
MemoryEntry ──── (no FK relationships)
PromptTemplate ──── (no FK relationships)
```

**Core Relationship Patterns:**
- **Project → Iteration → ResearchJob** - Primary workflow hierarchy
- **ResearchJob as Hub** - Most domain entities FK to ResearchJob
- **GenericFK Pattern** - WorkProduct and Annotation use ContentType framework for polymorphic relationships
- **OneToOne Relationships** - Report, GapAnalysis, InternalOpsIntel, AccountPlan link 1:1 with ResearchJob
- **Memory Module** - Standalone tables with no FK relationships (vector store references only)

---

## Detailed Table Documentation

## Research App

### `research_researchjob`

**Purpose:** Core entity tracking research job lifecycle, status, and linking to iteration workflow.

**Relationships:**
- OneToOne with `projects.Iteration` (nullable, for project-based research)
- OneToOne reverse: `ResearchReport`, `GapAnalysis`, `InternalOpsIntel`, `AccountPlan`
- ForeignKey reverse: `CompetitorCaseStudy`, `UseCase`, `Persona`, `OnePager`, `Citation`

| Field | DB Column | Type | Constraints | Description |
|-------|-----------|------|-------------|-------------|
| id | id | UUID | PRIMARY KEY, default=uuid.uuid4 | Primary key |
| client_name | client_name | VARCHAR(255) | NOT NULL | Name of the prospect company |
| sales_history | sales_history | TEXT | blank=True | Historical sales context/notes |
| prompt | prompt | TEXT | blank=True | Custom research prompt override |
| status | status | VARCHAR(20) | NOT NULL, default='pending', CHOICES | Job status: pending, running, completed, failed |
| result | result | TEXT | blank=True | Plain text research result (backward compatibility) |
| error | error | TEXT | blank=True | Error message if status=failed |
| vertical | vertical | VARCHAR(50) | NULL, blank=True, CHOICES | Industry vertical classification (see Vertical enum) |
| iteration_id | iteration_id | UUID | NULL, blank=True, UNIQUE FK | Link to projects.Iteration (nullable for standalone jobs) |
| created_at | created_at | TIMESTAMP | NOT NULL, auto_now_add | Creation timestamp |
| updated_at | updated_at | TIMESTAMP | NOT NULL, auto_now | Last update timestamp |

**Meta:**
- Ordering: `['-created_at']`

**Indexes:**
- PRIMARY KEY on `id` (automatic)
- UNIQUE INDEX on `iteration_id` (automatic from OneToOne)
- No explicit index on `vertical` (potential concern for filtering)

---

### `research_researchreport`

**Purpose:** Structured deep research report with company profile, decision makers, and strategic insights.

**Relationships:**
- OneToOne with `ResearchJob` (CASCADE delete)

| Field | DB Column | Type | Constraints | Description |
|-------|-----------|------|-------------|-------------|
| id | id | UUID | PRIMARY KEY, default=uuid.uuid4 | Primary key |
| research_job_id | research_job_id | UUID | NOT NULL, UNIQUE FK | Link to ResearchJob |
| company_overview | company_overview | TEXT | blank=True | Company description and overview |
| founded_year | founded_year | INTEGER | NULL, blank=True | Year company was founded |
| headquarters | headquarters | VARCHAR(255) | blank=True | HQ location |
| employee_count | employee_count | VARCHAR(100) | blank=True | Employee count range (e.g., "1,000-5,000") |
| annual_revenue | annual_revenue | VARCHAR(100) | blank=True | Revenue range (e.g., "$500M - $1B") |
| website | website | URL | blank=True | Company website URL |
| recent_news | recent_news | JSONB | NOT NULL, default=list | Array of news items (see JSONField Audit) |
| decision_makers | decision_makers | JSONB | NOT NULL, default=list | Array of decision maker objects |
| pain_points | pain_points | JSONB | NOT NULL, default=list | Array of pain point strings |
| opportunities | opportunities | JSONB | NOT NULL, default=list | Array of opportunity strings |
| digital_maturity | digital_maturity | VARCHAR(20) | blank=True, CHOICES | Digital maturity level (see DigitalMaturityLevel enum) |
| ai_footprint | ai_footprint | TEXT | blank=True | AI/ML technology footprint description |
| ai_adoption_stage | ai_adoption_stage | VARCHAR(20) | blank=True, CHOICES | AI adoption stage (see AIAdoptionStage enum) |
| strategic_goals | strategic_goals | JSONB | NOT NULL, default=list | Array of strategic goal strings |
| key_initiatives | key_initiatives | JSONB | NOT NULL, default=list | Array of initiative strings |
| talking_points | talking_points | JSONB | NOT NULL, default=list | Array of sales talking point strings |
| web_sources | web_sources | JSONB | NOT NULL, default=list | Array of web source objects from Google Search grounding |
| created_at | created_at | TIMESTAMP | NOT NULL, auto_now_add | Creation timestamp |
| updated_at | updated_at | TIMESTAMP | NOT NULL, auto_now | Last update timestamp |

**Meta:**
- Ordering: `['-created_at']`

**Indexes:**
- PRIMARY KEY on `id` (automatic)
- UNIQUE INDEX on `research_job_id` (automatic from OneToOne)

---

### `research_competitorcasestudy`

**Purpose:** Competitor AI case studies discovered during research for competitive intelligence.

**Relationships:**
- ForeignKey to `ResearchJob` (CASCADE delete)

| Field | DB Column | Type | Constraints | Description |
|-------|-----------|------|-------------|-------------|
| id | id | UUID | PRIMARY KEY, default=uuid.uuid4 | Primary key |
| research_job_id | research_job_id | UUID | NOT NULL, FK | Link to ResearchJob |
| competitor_name | competitor_name | VARCHAR(255) | NOT NULL | Name of competitor company |
| vertical | vertical | VARCHAR(50) | blank=True, CHOICES | Industry vertical of competitor |
| case_study_title | case_study_title | VARCHAR(500) | NOT NULL | Title of case study |
| summary | summary | TEXT | NOT NULL | Case study summary |
| technologies_used | technologies_used | JSONB | NOT NULL, default=list | Array of technology strings |
| outcomes | outcomes | JSONB | NOT NULL, default=list | Array of outcome strings |
| source_url | source_url | URL | blank=True | Source URL for case study |
| relevance_score | relevance_score | FLOAT | NOT NULL, default=0.0 | Relevance score (0.0-1.0) |
| created_at | created_at | TIMESTAMP | NOT NULL, auto_now_add | Creation timestamp |

**Meta:**
- Ordering: `['-relevance_score', '-created_at']`
- Verbose name plural: 'Competitor case studies'

**Indexes:**
- PRIMARY KEY on `id` (automatic)
- INDEX on `research_job_id` (automatic from ForeignKey)
- No explicit index on `relevance_score` (used in ordering)

---

### `research_gapanalysis`

**Purpose:** Technology, capability, and process gap analysis derived from sales history.

**Relationships:**
- OneToOne with `ResearchJob` (CASCADE delete)

| Field | DB Column | Type | Constraints | Description |
|-------|-----------|------|-------------|-------------|
| id | id | UUID | PRIMARY KEY, default=uuid.uuid4 | Primary key |
| research_job_id | research_job_id | UUID | NOT NULL, UNIQUE FK | Link to ResearchJob |
| technology_gaps | technology_gaps | JSONB | NOT NULL, default=list | Array of technology gap objects |
| capability_gaps | capability_gaps | JSONB | NOT NULL, default=list | Array of capability gap objects |
| process_gaps | process_gaps | JSONB | NOT NULL, default=list | Array of process gap objects |
| recommendations | recommendations | JSONB | NOT NULL, default=list | Array of recommendation strings |
| priority_areas | priority_areas | JSONB | NOT NULL, default=list | Array of priority area strings |
| confidence_score | confidence_score | FLOAT | NOT NULL, default=0.0 | Confidence in analysis (0.0-1.0) |
| analysis_notes | analysis_notes | TEXT | blank=True | Additional analysis notes |
| created_at | created_at | TIMESTAMP | NOT NULL, auto_now_add | Creation timestamp |
| updated_at | updated_at | TIMESTAMP | NOT NULL, auto_now | Last update timestamp |

**Meta:**
- Ordering: `['-created_at']`
- Verbose name plural: 'Gap analyses'

**Indexes:**
- PRIMARY KEY on `id` (automatic)
- UNIQUE INDEX on `research_job_id` (automatic from OneToOne)

---

### `research_internalopsintel`

**Purpose:** Internal Operations Intelligence from public sources (employee sentiment, LinkedIn, job postings, etc.).

**Relationships:**
- OneToOne with `ResearchJob` (CASCADE delete)

| Field | DB Column | Type | Constraints | Description |
|-------|-----------|------|-------------|-------------|
| id | id | UUID | PRIMARY KEY, default=uuid.uuid4 | Primary key |
| research_job_id | research_job_id | UUID | NOT NULL, UNIQUE FK | Link to ResearchJob |
| employee_sentiment | employee_sentiment | JSONB | NOT NULL, default=dict | Employee sentiment object (see JSONField Audit) |
| linkedin_presence | linkedin_presence | JSONB | NOT NULL, default=dict | LinkedIn presence object |
| social_media_mentions | social_media_mentions | JSONB | NOT NULL, default=list | Array of social media mention objects |
| job_postings | job_postings | JSONB | NOT NULL, default=dict | Job postings analysis object |
| news_sentiment | news_sentiment | JSONB | NOT NULL, default=dict | News sentiment analysis object |
| key_insights | key_insights | JSONB | NOT NULL, default=list | Array of key insight strings |
| gap_correlations | gap_correlations | JSONB | NOT NULL, default=list | Array of gap correlation objects (cross-ref with GapAnalysis) |
| confidence_score | confidence_score | FLOAT | NOT NULL, default=0.0 | Confidence in analysis (0.0-1.0) |
| data_freshness | data_freshness | VARCHAR(50) | blank=True | Data freshness indicator (e.g., "last_30_days") |
| analysis_notes | analysis_notes | TEXT | blank=True | Additional analysis notes |
| created_at | created_at | TIMESTAMP | NOT NULL, auto_now_add | Creation timestamp |
| updated_at | updated_at | TIMESTAMP | NOT NULL, auto_now | Last update timestamp |

**Meta:**
- Ordering: `['-created_at']`
- Verbose name: 'Internal Operations Intelligence'
- Verbose name plural: 'Internal Operations Intelligence'

**Indexes:**
- PRIMARY KEY on `id` (automatic)
- UNIQUE INDEX on `research_job_id` (automatic from OneToOne)

---

## Projects App

### `projects_project`

**Purpose:** Top-level engagement wrapper for iterative research workflow.

**Relationships:**
- ForeignKey reverse: `Iteration`, `WorkProduct`, `Annotation`

| Field | DB Column | Type | Constraints | Description |
|-------|-----------|------|-------------|-------------|
| id | id | UUID | PRIMARY KEY, default=uuid.uuid4 | Primary key |
| name | name | VARCHAR(255) | NOT NULL | Project name |
| client_name | client_name | VARCHAR(255) | NOT NULL | Client/prospect name |
| description | description | TEXT | blank=True | Project description |
| context_mode | context_mode | VARCHAR(20) | NOT NULL, default='accumulate', CHOICES | How iterations build: accumulate or fresh |
| created_at | created_at | TIMESTAMP | NOT NULL, auto_now_add | Creation timestamp |
| updated_at | updated_at | TIMESTAMP | NOT NULL, auto_now | Last update timestamp |

**Meta:**
- Ordering: `['-updated_at']`

**Indexes:**
- PRIMARY KEY on `id` (automatic)
- No explicit index on `client_name` (potential concern for searching)

**Methods:**
- `latest_iteration` - Property returning most recent iteration
- `get_iteration_count()` - Returns iteration count

---

### `projects_iteration`

**Purpose:** Single research iteration within a project, carries context between iterations.

**Relationships:**
- ForeignKey to `Project` (CASCADE delete)
- OneToOne reverse: `ResearchJob`
- ForeignKey reverse: `WorkProduct`

| Field | DB Column | Type | Constraints | Description |
|-------|-----------|------|-------------|-------------|
| id | id | UUID | PRIMARY KEY, default=uuid.uuid4 | Primary key |
| project_id | project_id | UUID | NOT NULL, FK | Link to Project |
| sequence | sequence | INTEGER | NOT NULL, UNIQUE TOGETHER with project_id | Iteration sequence number (1, 2, 3...) |
| name | name | VARCHAR(255) | blank=True | Optional iteration label (e.g., "v2 - Focus on AI") |
| sales_history | sales_history | TEXT | blank=True | Sales history for this iteration |
| prompt_override | prompt_override | TEXT | blank=True | Additional iteration-specific guidance |
| status | status | VARCHAR(20) | NOT NULL, default='pending', CHOICES | Iteration status: pending, running, completed, failed |
| inherited_context | inherited_context | JSONB | NOT NULL, default=dict | Context inherited from previous iteration |
| created_at | created_at | TIMESTAMP | NOT NULL, auto_now_add | Creation timestamp |

**Meta:**
- Ordering: `['sequence']`
- Unique together: `('project', 'sequence')`

**Indexes:**
- PRIMARY KEY on `id` (automatic)
- INDEX on `project_id` (automatic from ForeignKey)
- UNIQUE INDEX on `(project_id, sequence)` (from unique_together)

**Methods:**
- `save()` - Auto-assigns sequence number if not set

---

### `projects_workproduct`

**Purpose:** Items starred/saved as 'keepers' across iterations. Uses Generic Foreign Key to reference any model.

**Relationships:**
- ForeignKey to `Project` (CASCADE delete)
- ForeignKey to `Iteration` (SET_NULL on delete)
- GenericFK to any model via ContentType framework

| Field | DB Column | Type | Constraints | Description |
|-------|-----------|------|-------------|-------------|
| id | id | UUID | PRIMARY KEY, default=uuid.uuid4 | Primary key |
| project_id | project_id | UUID | NOT NULL, FK | Link to Project |
| source_iteration_id | source_iteration_id | UUID | NULL, blank=True, FK | Iteration where this was created |
| content_type_id | content_type_id | INTEGER | NOT NULL, FK | ContentType for GenericFK |
| object_id | object_id | UUID | NOT NULL | UUID of referenced object |
| category | category | VARCHAR(50) | NOT NULL, CHOICES | Category: play, persona, insight, one_pager, case_study, use_case, gap, other |
| starred | starred | BOOLEAN | NOT NULL, default=True | Whether item is starred |
| notes | notes | TEXT | blank=True | User notes on work product |
| created_at | created_at | TIMESTAMP | NOT NULL, auto_now_add | Creation timestamp |

**Meta:**
- Ordering: `['-created_at']`

**Indexes:**
- PRIMARY KEY on `id` (automatic)
- INDEX on `project_id` (automatic from ForeignKey)
- INDEX on `source_iteration_id` (automatic from ForeignKey)
- INDEX on `content_type_id` (automatic from ForeignKey)
- No composite index on `(content_type_id, object_id)` for GenericFK lookups (CONCERN)

---

### `projects_annotation`

**Purpose:** User notes attached to any object via Generic Foreign Key.

**Relationships:**
- ForeignKey to `Project` (CASCADE delete)
- GenericFK to any model via ContentType framework

| Field | DB Column | Type | Constraints | Description |
|-------|-----------|------|-------------|-------------|
| id | id | UUID | PRIMARY KEY, default=uuid.uuid4 | Primary key |
| project_id | project_id | UUID | NOT NULL, FK | Link to Project |
| content_type_id | content_type_id | INTEGER | NOT NULL, FK | ContentType for GenericFK |
| object_id | object_id | UUID | NOT NULL | UUID of referenced object |
| text | text | TEXT | NOT NULL | Annotation text |
| created_at | created_at | TIMESTAMP | NOT NULL, auto_now_add | Creation timestamp |
| updated_at | updated_at | TIMESTAMP | NOT NULL, auto_now | Last update timestamp |

**Meta:**
- Ordering: `['-created_at']`

**Indexes:**
- PRIMARY KEY on `id` (automatic)
- INDEX on `project_id` (automatic from ForeignKey)
- INDEX on `content_type_id` (automatic from ForeignKey)
- No composite index on `(content_type_id, object_id)` for GenericFK lookups (CONCERN)

---

## Ideation App

### `ideation_usecase`

**Purpose:** AI/technology use case generated during ideation loop.

**Relationships:**
- ForeignKey to `ResearchJob` (CASCADE delete)
- OneToOne reverse: `FeasibilityAssessment`, `RefinedPlay`

| Field | DB Column | Type | Constraints | Description |
|-------|-----------|------|-------------|-------------|
| id | id | UUID | PRIMARY KEY, default=uuid.uuid4 | Primary key |
| research_job_id | research_job_id | UUID | NOT NULL, FK | Link to ResearchJob |
| title | title | VARCHAR(255) | NOT NULL | Use case title |
| description | description | TEXT | NOT NULL | Use case description |
| business_problem | business_problem | TEXT | NOT NULL | Business problem addressed |
| proposed_solution | proposed_solution | TEXT | NOT NULL | AI/technology solution overview |
| expected_benefits | expected_benefits | JSONB | NOT NULL, default=list | Array of benefit strings |
| estimated_roi | estimated_roi | VARCHAR(100) | blank=True | ROI estimate |
| time_to_value | time_to_value | VARCHAR(100) | blank=True | Time to value estimate |
| technologies | technologies | JSONB | NOT NULL, default=list | Array of technology strings |
| data_requirements | data_requirements | JSONB | NOT NULL, default=list | Array of data requirement strings |
| integration_points | integration_points | JSONB | NOT NULL, default=list | Array of integration point strings |
| priority | priority | VARCHAR(20) | NOT NULL, default='medium', CHOICES | Priority: high, medium, low |
| impact_score | impact_score | FLOAT | NOT NULL, default=0.0 | Impact score (0.0-1.0) |
| feasibility_score | feasibility_score | FLOAT | NOT NULL, default=0.0 | Feasibility score (0.0-1.0) |
| status | status | VARCHAR(20) | NOT NULL, default='draft', CHOICES | Status: draft, validated, refined, approved, rejected |
| created_at | created_at | TIMESTAMP | NOT NULL, auto_now_add | Creation timestamp |
| updated_at | updated_at | TIMESTAMP | NOT NULL, auto_now | Last update timestamp |

**Meta:**
- Ordering: `['-priority', '-impact_score', '-created_at']`

**Indexes:**
- PRIMARY KEY on `id` (automatic)
- INDEX on `research_job_id` (automatic from ForeignKey)
- No explicit indexes on `priority` or `impact_score` (used in ordering)

---

### `ideation_feasibilityassessment`

**Purpose:** Technical feasibility assessment for a use case.

**Relationships:**
- OneToOne with `UseCase` (CASCADE delete)

| Field | DB Column | Type | Constraints | Description |
|-------|-----------|------|-------------|-------------|
| id | id | UUID | PRIMARY KEY, default=uuid.uuid4 | Primary key |
| use_case_id | use_case_id | UUID | NOT NULL, UNIQUE FK | Link to UseCase |
| overall_feasibility | overall_feasibility | VARCHAR(20) | NOT NULL, default='medium', CHOICES | Overall feasibility: low, medium, high |
| overall_score | overall_score | FLOAT | NOT NULL, default=0.0 | Overall feasibility score (0.0-1.0) |
| technical_complexity | technical_complexity | TEXT | blank=True | Technical complexity assessment |
| data_availability | data_availability | TEXT | blank=True | Data availability assessment |
| integration_complexity | integration_complexity | TEXT | blank=True | Integration complexity assessment |
| scalability_considerations | scalability_considerations | TEXT | blank=True | Scalability considerations |
| technical_risks | technical_risks | JSONB | NOT NULL, default=list | Array of technical risk strings |
| mitigation_strategies | mitigation_strategies | JSONB | NOT NULL, default=list | Array of mitigation strategy strings |
| prerequisites | prerequisites | JSONB | NOT NULL, default=list | Array of prerequisite strings |
| dependencies | dependencies | JSONB | NOT NULL, default=list | Array of dependency strings |
| recommendations | recommendations | TEXT | blank=True | Recommendations |
| next_steps | next_steps | JSONB | NOT NULL, default=list | Array of next step strings |
| created_at | created_at | TIMESTAMP | NOT NULL, auto_now_add | Creation timestamp |
| updated_at | updated_at | TIMESTAMP | NOT NULL, auto_now | Last update timestamp |

**Meta:**
- Ordering: `['-created_at']`

**Indexes:**
- PRIMARY KEY on `id` (automatic)
- UNIQUE INDEX on `use_case_id` (automatic from OneToOne)

---

### `ideation_refinedplay`

**Purpose:** Refined sales play generated from use case with sales enablement content.

**Relationships:**
- OneToOne with `UseCase` (CASCADE delete)

| Field | DB Column | Type | Constraints | Description |
|-------|-----------|------|-------------|-------------|
| id | id | UUID | PRIMARY KEY, default=uuid.uuid4 | Primary key |
| use_case_id | use_case_id | UUID | NOT NULL, UNIQUE FK | Link to UseCase |
| title | title | VARCHAR(255) | NOT NULL | Play title |
| elevator_pitch | elevator_pitch | TEXT | NOT NULL | 30-second pitch |
| value_proposition | value_proposition | TEXT | NOT NULL | Value proposition |
| key_differentiators | key_differentiators | JSONB | NOT NULL, default=list | Array of differentiator strings |
| target_persona | target_persona | VARCHAR(255) | blank=True | Target persona |
| target_vertical | target_vertical | VARCHAR(100) | blank=True | Target vertical |
| company_size_fit | company_size_fit | VARCHAR(100) | blank=True | Company size fit |
| discovery_questions | discovery_questions | JSONB | NOT NULL, default=list | Array of discovery question strings |
| objection_handlers | objection_handlers | JSONB | NOT NULL, default=list | Array of objection handler strings |
| proof_points | proof_points | JSONB | NOT NULL, default=list | Array of proof point strings |
| competitive_positioning | competitive_positioning | TEXT | blank=True | Competitive positioning statement |
| next_steps | next_steps | JSONB | NOT NULL, default=list | Array of next step strings |
| success_metrics | success_metrics | JSONB | NOT NULL, default=list | Array of success metric strings |
| status | status | VARCHAR(20) | NOT NULL, default='draft', CHOICES | Status: draft, reviewed, approved, active, archived |
| created_at | created_at | TIMESTAMP | NOT NULL, auto_now_add | Creation timestamp |
| updated_at | updated_at | TIMESTAMP | NOT NULL, auto_now | Last update timestamp |

**Meta:**
- Ordering: `['-created_at']`
- Verbose name plural: 'Refined plays'

**Indexes:**
- PRIMARY KEY on `id` (automatic)
- UNIQUE INDEX on `use_case_id` (automatic from OneToOne)

---

## Assets App

### `assets_persona`

**Purpose:** Buyer persona generated from research for sales targeting.

**Relationships:**
- ForeignKey to `ResearchJob` (CASCADE delete)

| Field | DB Column | Type | Constraints | Description |
|-------|-----------|------|-------------|-------------|
| id | id | UUID | PRIMARY KEY, default=uuid.uuid4 | Primary key |
| research_job_id | research_job_id | UUID | NOT NULL, FK | Link to ResearchJob |
| name | name | VARCHAR(255) | NOT NULL | Persona name |
| title | title | VARCHAR(255) | NOT NULL | Job title |
| department | department | VARCHAR(100) | blank=True | Department |
| seniority_level | seniority_level | VARCHAR(50) | blank=True | Seniority level |
| background | background | TEXT | blank=True | Professional background |
| goals | goals | JSONB | NOT NULL, default=list | Array of goal strings |
| challenges | challenges | JSONB | NOT NULL, default=list | Array of challenge strings |
| motivations | motivations | JSONB | NOT NULL, default=list | Array of motivation strings |
| decision_criteria | decision_criteria | JSONB | NOT NULL, default=list | Array of decision criteria strings |
| preferred_communication | preferred_communication | VARCHAR(100) | blank=True | Preferred communication style |
| objections | objections | JSONB | NOT NULL, default=list | Array of objection strings |
| content_preferences | content_preferences | JSONB | NOT NULL, default=list | Array of content preference strings |
| key_messages | key_messages | JSONB | NOT NULL, default=list | Array of key message strings |
| created_at | created_at | TIMESTAMP | NOT NULL, auto_now_add | Creation timestamp |
| updated_at | updated_at | TIMESTAMP | NOT NULL, auto_now | Last update timestamp |

**Meta:**
- Ordering: `['-created_at']`

**Indexes:**
- PRIMARY KEY on `id` (automatic)
- INDEX on `research_job_id` (automatic from ForeignKey)

---

### `assets_onepager`

**Purpose:** One-page sales document generated from research.

**Relationships:**
- ForeignKey to `ResearchJob` (CASCADE delete)

| Field | DB Column | Type | Constraints | Description |
|-------|-----------|------|-------------|-------------|
| id | id | UUID | PRIMARY KEY, default=uuid.uuid4 | Primary key |
| research_job_id | research_job_id | UUID | NOT NULL, FK | Link to ResearchJob |
| title | title | VARCHAR(255) | NOT NULL | Document title |
| headline | headline | VARCHAR(500) | NOT NULL | Headline/tagline |
| executive_summary | executive_summary | TEXT | NOT NULL | Executive summary |
| challenge_section | challenge_section | TEXT | NOT NULL | Challenge section content |
| solution_section | solution_section | TEXT | NOT NULL | Solution section content |
| benefits_section | benefits_section | TEXT | NOT NULL | Benefits section content |
| differentiators | differentiators | JSONB | NOT NULL, default=list | Array of differentiator strings |
| call_to_action | call_to_action | TEXT | blank=True | Call to action |
| next_steps | next_steps | JSONB | NOT NULL, default=list | Array of next step strings |
| html_content | html_content | TEXT | blank=True | Rendered HTML content for export |
| pdf_path | pdf_path | VARCHAR(500) | blank=True | Path to exported PDF |
| status | status | VARCHAR(20) | NOT NULL, default='draft', CHOICES | Status: draft, reviewed, approved, shared |
| created_at | created_at | TIMESTAMP | NOT NULL, auto_now_add | Creation timestamp |
| updated_at | updated_at | TIMESTAMP | NOT NULL, auto_now | Last update timestamp |

**Meta:**
- Ordering: `['-created_at']`
- Verbose name plural: 'One pagers'

**Indexes:**
- PRIMARY KEY on `id` (automatic)
- INDEX on `research_job_id` (automatic from ForeignKey)

---

### `assets_accountplan`

**Purpose:** Strategic account plan document generated from research.

**Relationships:**
- OneToOne with `ResearchJob` (CASCADE delete)

| Field | DB Column | Type | Constraints | Description |
|-------|-----------|------|-------------|-------------|
| id | id | UUID | PRIMARY KEY, default=uuid.uuid4 | Primary key |
| research_job_id | research_job_id | UUID | NOT NULL, UNIQUE FK | Link to ResearchJob |
| title | title | VARCHAR(255) | NOT NULL | Plan title |
| executive_summary | executive_summary | TEXT | NOT NULL | Executive summary |
| account_overview | account_overview | TEXT | NOT NULL | Account overview |
| strategic_objectives | strategic_objectives | JSONB | NOT NULL, default=list | Array of strategic objective strings |
| key_stakeholders | key_stakeholders | JSONB | NOT NULL, default=list | Array of stakeholder objects |
| opportunities | opportunities | JSONB | NOT NULL, default=list | Array of opportunity objects |
| competitive_landscape | competitive_landscape | TEXT | blank=True | Competitive landscape analysis |
| swot_analysis | swot_analysis | JSONB | NOT NULL, default=dict | SWOT analysis object |
| engagement_strategy | engagement_strategy | TEXT | blank=True | Engagement strategy |
| value_propositions | value_propositions | JSONB | NOT NULL, default=list | Array of value proposition strings |
| action_plan | action_plan | JSONB | NOT NULL, default=list | Array of action item objects |
| success_metrics | success_metrics | JSONB | NOT NULL, default=list | Array of success metric strings |
| milestones | milestones | JSONB | NOT NULL, default=list | Array of milestone objects |
| timeline | timeline | TEXT | blank=True | Timeline description |
| html_content | html_content | TEXT | blank=True | Rendered HTML content for export |
| pdf_path | pdf_path | VARCHAR(500) | blank=True | Path to exported PDF |
| status | status | VARCHAR(20) | NOT NULL, default='draft', CHOICES | Status: draft, in_progress, reviewed, approved, active |
| created_at | created_at | TIMESTAMP | NOT NULL, auto_now_add | Creation timestamp |
| updated_at | updated_at | TIMESTAMP | NOT NULL, auto_now | Last update timestamp |

**Meta:**
- Ordering: `['-created_at']`

**Indexes:**
- PRIMARY KEY on `id` (automatic)
- UNIQUE INDEX on `research_job_id` (automatic from OneToOne)

---

### `assets_citation`

**Purpose:** Source citation for generated content. **NOTE: Appears to be unused in application code.**

**Relationships:**
- ForeignKey to `ResearchJob` (CASCADE delete)

| Field | DB Column | Type | Constraints | Description |
|-------|-----------|------|-------------|-------------|
| id | id | UUID | PRIMARY KEY, default=uuid.uuid4 | Primary key |
| research_job_id | research_job_id | UUID | NOT NULL, FK | Link to ResearchJob |
| citation_type | citation_type | VARCHAR(50) | NOT NULL, CHOICES | Type: news, website, report, social, financial, press_release, other |
| title | title | VARCHAR(500) | NOT NULL | Citation title |
| source | source | VARCHAR(255) | NOT NULL | Source name |
| url | url | URL | blank=True | Source URL |
| author | author | VARCHAR(255) | blank=True | Author name |
| publication_date | publication_date | DATE | NULL, blank=True | Publication date |
| excerpt | excerpt | TEXT | blank=True | Relevant excerpt |
| relevance_note | relevance_note | TEXT | blank=True | Why this citation is relevant |
| verified | verified | BOOLEAN | NOT NULL, default=False | Whether citation has been verified |
| verification_date | verification_date | TIMESTAMP | NULL, blank=True | When citation was verified |
| created_at | created_at | TIMESTAMP | NOT NULL, auto_now_add | Creation timestamp |

**Meta:**
- Ordering: `['-created_at']`

**Indexes:**
- PRIMARY KEY on `id` (automatic)
- INDEX on `research_job_id` (automatic from ForeignKey)

---

## Memory App

### `memory_clientprofile`

**Purpose:** Client profile stored in vector database for context retrieval. No FK relationships - indexed by client_name.

**Relationships:**
- None (standalone, references ChromaDB via `vector_id`)

| Field | DB Column | Type | Constraints | Description |
|-------|-----------|------|-------------|-------------|
| id | id | UUID | PRIMARY KEY, default=uuid.uuid4 | Primary key |
| client_name | client_name | VARCHAR(255) | NOT NULL, UNIQUE | Client/company name |
| industry | industry | VARCHAR(100) | blank=True | Industry classification |
| company_size | company_size | VARCHAR(50) | blank=True | Company size range |
| region | region | VARCHAR(100) | blank=True | Geographic region |
| key_contacts | key_contacts | JSONB | NOT NULL, default=list | Array of contact objects |
| summary | summary | TEXT | blank=True | Profile summary for embedding |
| vector_id | vector_id | VARCHAR(255) | blank=True | Reference ID in ChromaDB |
| created_at | created_at | TIMESTAMP | NOT NULL, auto_now_add | Creation timestamp |
| updated_at | updated_at | TIMESTAMP | NOT NULL, auto_now | Last update timestamp |

**Meta:**
- Ordering: `['-updated_at']`

**Indexes:**
- PRIMARY KEY on `id` (automatic)
- UNIQUE INDEX on `client_name` (automatic)

---

### `memory_salesplay`

**Purpose:** Sales play/strategy stored for retrieval and reuse. No FK relationships.

**Relationships:**
- None (standalone, references ChromaDB via `vector_id`)

| Field | DB Column | Type | Constraints | Description |
|-------|-----------|------|-------------|-------------|
| id | id | UUID | PRIMARY KEY, default=uuid.uuid4 | Primary key |
| title | title | VARCHAR(255) | NOT NULL | Play title |
| play_type | play_type | VARCHAR(50) | NOT NULL, CHOICES | Type: pitch, objection_handler, value_proposition, case_study, competitive_response, discovery_question |
| content | content | TEXT | NOT NULL | Play content |
| context | context | TEXT | blank=True | When to use this play |
| industry | industry | VARCHAR(100) | blank=True | Target industry |
| vertical | vertical | VARCHAR(50) | blank=True | Target vertical |
| usage_count | usage_count | INTEGER | NOT NULL, default=0 | Number of times used |
| success_rate | success_rate | FLOAT | NOT NULL, default=0.0 | Success rate (0.0-1.0) |
| vector_id | vector_id | VARCHAR(255) | blank=True | Reference ID in ChromaDB |
| created_at | created_at | TIMESTAMP | NOT NULL, auto_now_add | Creation timestamp |
| updated_at | updated_at | TIMESTAMP | NOT NULL, auto_now | Last update timestamp |

**Meta:**
- Ordering: `['-success_rate', '-usage_count']`
- Verbose name plural: 'Sales plays'

**Indexes:**
- PRIMARY KEY on `id` (automatic)
- No explicit indexes on `success_rate` or `usage_count` (used in ordering)

---

### `memory_memoryentry`

**Purpose:** Generic memory entry for storing various types of knowledge. No FK relationships.

**Relationships:**
- None (standalone, references ChromaDB via `vector_id`)

| Field | DB Column | Type | Constraints | Description |
|-------|-----------|------|-------------|-------------|
| id | id | UUID | PRIMARY KEY, default=uuid.uuid4 | Primary key |
| entry_type | entry_type | VARCHAR(50) | NOT NULL, CHOICES | Type: research_insight, client_interaction, deal_outcome, best_practice, lesson_learned |
| title | title | VARCHAR(255) | NOT NULL | Entry title |
| content | content | TEXT | NOT NULL | Entry content |
| client_name | client_name | VARCHAR(255) | blank=True | Associated client name |
| industry | industry | VARCHAR(100) | blank=True | Associated industry |
| tags | tags | JSONB | NOT NULL, default=list | Array of tag strings |
| source_type | source_type | VARCHAR(50) | blank=True | Source type |
| source_id | source_id | VARCHAR(255) | blank=True | Source reference ID |
| vector_id | vector_id | VARCHAR(255) | blank=True | Reference ID in ChromaDB |
| created_at | created_at | TIMESTAMP | NOT NULL, auto_now_add | Creation timestamp |
| updated_at | updated_at | TIMESTAMP | NOT NULL, auto_now | Last update timestamp |

**Meta:**
- Ordering: `['-created_at']`
- Verbose name plural: 'Memory entries'

**Indexes:**
- PRIMARY KEY on `id` (automatic)
- No explicit index on `client_name` (used for filtering)
- No explicit index on `industry` (used for filtering)

---

## Prompts App

### `prompts_prompttemplate`

**Purpose:** Configurable prompt template management. Uses Django default BigAutoField PK.

**Relationships:**
- None (standalone)

| Field | DB Column | Type | Constraints | Description |
|-------|-----------|------|-------------|-------------|
| id | id | BIGINT | PRIMARY KEY, auto_increment | Primary key (Django default) |
| name | name | VARCHAR(100) | NOT NULL, UNIQUE | Template name |
| content | content | TEXT | NOT NULL | Template content (supports {placeholders}) |
| is_default | is_default | BOOLEAN | NOT NULL, default=False | Whether this is the default template |
| created_at | created_at | TIMESTAMP | NOT NULL, auto_now_add | Creation timestamp |
| updated_at | updated_at | TIMESTAMP | NOT NULL, auto_now | Last update timestamp |

**Meta:**
- Ordering: `['-is_default', '-updated_at']`

**Indexes:**
- PRIMARY KEY on `id` (automatic)
- UNIQUE INDEX on `name` (automatic)

**Methods:**
- `save()` - Ensures only one default prompt exists
- `get_default()` - Class method to retrieve default prompt
- `get_default_content()` - Static method with default prompt text

---

## Cross-Table Analysis

### Entity Relationship Overview

**Primary Workflow:**
```
Project → Iteration → ResearchJob → [Report, GapAnalysis, InternalOpsIntel, CompetitorCaseStudy, ...]
```

**Asset Generation:**
```
ResearchJob → [UseCase → FeasibilityAssessment, RefinedPlay]
ResearchJob → [Persona, OnePager, AccountPlan, Citation]
```

**Work Product Tracking:**
```
Project → WorkProduct (GenericFK) → [Any of above entities]
Project → Annotation (GenericFK) → [Any of above entities]
```

**Memory/Knowledge Base (Standalone):**
```
ClientProfile (no FK, indexed by client_name)
SalesPlay (no FK, vector store reference)
MemoryEntry (no FK, vector store reference)
PromptTemplate (no FK, global configuration)
```

### Key Relationships

| From | To | Type | On Delete | Notes |
|------|-----|------|-----------|-------|
| Project | Iteration | ForeignKey (1:M) | CASCADE | Project can have multiple iterations |
| Iteration | ResearchJob | OneToOne | CASCADE | Each iteration has one research job |
| ResearchJob | ResearchReport | OneToOne | CASCADE | 1:1 structured report |
| ResearchJob | GapAnalysis | OneToOne | CASCADE | 1:1 gap analysis |
| ResearchJob | InternalOpsIntel | OneToOne | CASCADE | 1:1 internal ops intel |
| ResearchJob | AccountPlan | OneToOne | CASCADE | 1:1 account plan |
| ResearchJob | CompetitorCaseStudy | ForeignKey (1:M) | CASCADE | Multiple case studies per job |
| ResearchJob | UseCase | ForeignKey (1:M) | CASCADE | Multiple use cases per job |
| ResearchJob | Persona | ForeignKey (1:M) | CASCADE | Multiple personas per job |
| ResearchJob | OnePager | ForeignKey (1:M) | CASCADE | Multiple one-pagers per job |
| ResearchJob | Citation | ForeignKey (1:M) | CASCADE | Multiple citations per job |
| UseCase | FeasibilityAssessment | OneToOne | CASCADE | 1:1 feasibility assessment |
| UseCase | RefinedPlay | OneToOne | CASCADE | 1:1 refined play |
| Project | WorkProduct | ForeignKey (1:M) | CASCADE | Multiple starred items per project |
| Project | Annotation | ForeignKey (1:M) | CASCADE | Multiple annotations per project |
| Iteration | WorkProduct | ForeignKey (1:M) | SET_NULL | Track source iteration, preserve if iteration deleted |
| WorkProduct | (any model) | GenericFK | N/A | Polymorphic reference via ContentType |
| Annotation | (any model) | GenericFK | N/A | Polymorphic reference via ContentType |

---

## JSONField Audit

### Structure Documentation

This section documents the expected structure of all JSONField columns based on service code analysis.

#### `research_researchreport`

**recent_news** (Array of objects):
```json
[
  {
    "title": "string",
    "summary": "string",
    "date": "string (YYYY-MM-DD or descriptive)",
    "source": "string",
    "url": "string (URL)"
  }
]
```

**decision_makers** (Array of objects):
```json
[
  {
    "name": "string",
    "title": "string",
    "background": "string",
    "linkedin_url": "string (URL)"
  }
]
```

**pain_points** (Array of strings):
```json
["Pain point 1", "Pain point 2"]
```

**opportunities** (Array of strings):
```json
["Opportunity 1", "Opportunity 2"]
```

**strategic_goals** (Array of strings):
```json
["Strategic goal 1", "Strategic goal 2"]
```

**key_initiatives** (Array of strings):
```json
["Initiative 1", "Initiative 2"]
```

**talking_points** (Array of strings):
```json
["Talking point 1", "Talking point 2"]
```

**web_sources** (Array of objects from Google Search grounding):
```json
[
  {
    "uri": "string (URL)",
    "title": "string"
  }
]
```

#### `research_competitorcasestudy`

**technologies_used** (Array of strings):
```json
["Technology 1", "Technology 2"]
```

**outcomes** (Array of strings):
```json
["Outcome 1", "Outcome 2"]
```

#### `research_gapanalysis`

**technology_gaps** (Array of objects):
```json
[
  {
    "gap": "string",
    "description": "string",
    "severity": "string (high/medium/low)"
  }
]
```

**capability_gaps** (Array of objects):
```json
[
  {
    "gap": "string",
    "description": "string",
    "severity": "string"
  }
]
```

**process_gaps** (Array of objects):
```json
[
  {
    "gap": "string",
    "description": "string",
    "severity": "string"
  }
]
```

**recommendations** (Array of strings):
```json
["Recommendation 1", "Recommendation 2"]
```

**priority_areas** (Array of strings):
```json
["Priority area 1", "Priority area 2"]
```

#### `research_internalopsintel`

**employee_sentiment** (Object):
```json
{
  "overall_rating": 0.0,
  "work_life_balance": 0.0,
  "compensation": 0.0,
  "culture": 0.0,
  "management": 0.0,
  "recommend_pct": 0,
  "positive_themes": ["theme1", "theme2"],
  "negative_themes": ["theme1", "theme2"],
  "trend": "stable|improving|declining"
}
```

**linkedin_presence** (Object):
```json
{
  "follower_count": 0,
  "engagement_level": "low|medium|high",
  "recent_posts": [{"title": "string", "summary": "string", "date": "string"}],
  "employee_trend": "growing|shrinking|stable",
  "notable_changes": ["change1", "change2"]
}
```

**social_media_mentions** (Array of objects):
```json
[
  {
    "platform": "reddit|twitter|facebook",
    "summary": "string",
    "sentiment": "positive|negative|neutral|mixed",
    "topic": "string"
  }
]
```

**job_postings** (Object):
```json
{
  "total_openings": 0,
  "departments_hiring": {"dept_name": count},
  "skills_sought": ["skill1", "skill2"],
  "seniority_distribution": {"level": count},
  "urgency_signals": ["signal1", "signal2"],
  "insights": "string"
}
```

**news_sentiment** (Object):
```json
{
  "overall_sentiment": "positive|negative|neutral|mixed",
  "coverage_volume": "high|medium|low",
  "topics": ["topic1", "topic2"],
  "headlines": ["headline1", "headline2"]
}
```

**key_insights** (Array of strings):
```json
["Insight 1", "Insight 2"]
```

**gap_correlations** (Array of objects - cross-references GapAnalysis):
```json
[
  {
    "gap_type": "technology|capability|process",
    "description": "string",
    "evidence": "string",
    "evidence_type": "employee_sentiment|job_postings|news|social",
    "confidence": 0.0,
    "sales_implication": "string"
  }
]
```

#### `projects_iteration`

**inherited_context** (Object - free-form, accumulated from previous iterations):
```json
{
  "previous_findings": ["finding1", "finding2"],
  "accumulated_insights": ["insight1", "insight2"],
  "key_contacts": [{"name": "string", "role": "string"}],
  "custom_data": {}
}
```

#### `ideation_usecase`

**expected_benefits** (Array of strings):
```json
["Benefit 1", "Benefit 2"]
```

**technologies** (Array of strings):
```json
["Technology 1", "Technology 2"]
```

**data_requirements** (Array of strings):
```json
["Requirement 1", "Requirement 2"]
```

**integration_points** (Array of strings):
```json
["Integration point 1", "Integration point 2"]
```

#### `ideation_feasibilityassessment`

**technical_risks** (Array of strings):
```json
["Risk 1", "Risk 2"]
```

**mitigation_strategies** (Array of strings):
```json
["Strategy 1", "Strategy 2"]
```

**prerequisites** (Array of strings):
```json
["Prerequisite 1", "Prerequisite 2"]
```

**dependencies** (Array of strings):
```json
["Dependency 1", "Dependency 2"]
```

**next_steps** (Array of strings):
```json
["Step 1", "Step 2"]
```

#### `ideation_refinedplay`

**key_differentiators** (Array of strings):
```json
["Differentiator 1", "Differentiator 2"]
```

**discovery_questions** (Array of strings):
```json
["Question 1", "Question 2"]
```

**objection_handlers** (Array of strings):
```json
["Handler 1", "Handler 2"]
```

**proof_points** (Array of strings):
```json
["Proof point 1", "Proof point 2"]
```

**next_steps** (Array of strings):
```json
["Step 1", "Step 2"]
```

**success_metrics** (Array of strings):
```json
["Metric 1", "Metric 2"]
```

#### `assets_persona`

**goals** (Array of strings):
```json
["Goal 1", "Goal 2"]
```

**challenges** (Array of strings):
```json
["Challenge 1", "Challenge 2"]
```

**motivations** (Array of strings):
```json
["Motivation 1", "Motivation 2"]
```

**decision_criteria** (Array of strings):
```json
["Criteria 1", "Criteria 2"]
```

**objections** (Array of strings):
```json
["Objection 1", "Objection 2"]
```

**content_preferences** (Array of strings):
```json
["Preference 1", "Preference 2"]
```

**key_messages** (Array of strings):
```json
["Message 1", "Message 2"]
```

#### `assets_onepager`

**differentiators** (Array of strings):
```json
["Differentiator 1", "Differentiator 2"]
```

**next_steps** (Array of strings):
```json
["Step 1", "Step 2"]
```

#### `assets_accountplan`

**strategic_objectives** (Array of strings):
```json
["Objective 1", "Objective 2"]
```

**key_stakeholders** (Array of objects):
```json
[
  {
    "name": "string",
    "title": "string",
    "influence": "high|medium|low",
    "relationship": "string"
  }
]
```

**opportunities** (Array of objects):
```json
[
  {
    "title": "string",
    "description": "string",
    "value": "string",
    "timeline": "string"
  }
]
```

**swot_analysis** (Object):
```json
{
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"],
  "opportunities": ["opportunity1", "opportunity2"],
  "threats": ["threat1", "threat2"]
}
```

**value_propositions** (Array of strings):
```json
["Proposition 1", "Proposition 2"]
```

**action_plan** (Array of objects):
```json
[
  {
    "action": "string",
    "owner": "string",
    "due_date": "string",
    "status": "string"
  }
]
```

**success_metrics** (Array of strings):
```json
["Metric 1", "Metric 2"]
```

**milestones** (Array of objects):
```json
[
  {
    "title": "string",
    "date": "string",
    "description": "string"
  }
]
```

#### `memory_clientprofile`

**key_contacts** (Array of objects):
```json
[
  {
    "name": "string",
    "title": "string",
    "email": "string",
    "phone": "string"
  }
]
```

#### `memory_memoryentry`

**tags** (Array of strings):
```json
["tag1", "tag2", "tag3"]
```

### JSONField Normalization Assessment

**Should Remain JSON:**
- `research_researchreport.recent_news` - Variable structure, temporal data
- `research_researchreport.web_sources` - External grounding metadata
- `research_internalopsintel.*` - Complex nested structures from AI analysis
- `projects_iteration.inherited_context` - Free-form accumulated context
- `assets_accountplan.swot_analysis` - Fixed structure, always together
- Simple string arrays (pain_points, opportunities, talking_points, etc.)

**Consider Normalizing:**
- `research_researchreport.decision_makers` → New `DecisionMaker` table
  - **Reason:** Structured entities that could be queried, updated independently
  - **Benefit:** Enable queries like "all decision makers in title='CTO'", track updates over time
- `assets_accountplan.key_stakeholders` → New `Stakeholder` table
  - **Reason:** Similar to decision makers, structured entities
  - **Benefit:** Track relationship changes, influence scoring over time
- `memory_clientprofile.key_contacts` → New `Contact` table
  - **Reason:** Core CRM-like data, should be first-class entities
  - **Benefit:** Enable contact-based queries, relationship tracking

**Low Priority for Normalization:**
- Most other JSONFields are appropriate for their use cases
- Arrays of simple strings are fine as JSON
- Complex AI-generated structures (InternalOpsIntel) are best kept as JSON

---

## Index Audit

### Explicit Indexes

**None found.** The schema relies entirely on Django's automatic indexes.

### Automatic Indexes (Django ORM)

Django automatically creates indexes for:
- PRIMARY KEY fields (all tables)
- UNIQUE constraints (`prompts_prompttemplate.name`, `memory_clientprofile.client_name`)
- UNIQUE TOGETHER constraints (`projects_iteration` on `(project_id, sequence)`)
- ForeignKey fields (all FK relationships)
- OneToOneField fields (all 1:1 relationships)

### Foreign Key Index Coverage

All ForeignKey fields automatically have indexes:
- `research_researchjob.iteration_id` ✓
- `research_researchreport.research_job_id` ✓
- `research_competitorcasestudy.research_job_id` ✓
- `research_gapanalysis.research_job_id` ✓
- `research_internalopsintel.research_job_id` ✓
- `projects_iteration.project_id` ✓
- `projects_workproduct.project_id` ✓
- `projects_workproduct.source_iteration_id` ✓
- `projects_workproduct.content_type_id` ✓
- `projects_annotation.project_id` ✓
- `projects_annotation.content_type_id` ✓
- `ideation_usecase.research_job_id` ✓
- `ideation_feasibilityassessment.use_case_id` ✓
- `ideation_refinedplay.use_case_id` ✓
- `assets_persona.research_job_id` ✓
- `assets_onepager.research_job_id` ✓
- `assets_accountplan.research_job_id` ✓
- `assets_citation.research_job_id` ✓

### Missing Indexes (CONCERNS)

#### HIGH Priority

1. **`projects_workproduct` & `projects_annotation` GenericFK lookups**
   - **Missing:** Composite index on `(content_type_id, object_id)`
   - **Reason:** GenericFK queries always filter by both columns
   - **Query pattern:** `WHERE content_type_id = X AND object_id = Y`
   - **Impact:** Full table scans when looking up annotations/work products for a specific object

2. **`research_researchjob.vertical`**
   - **Missing:** Index on `vertical` column
   - **Reason:** Used for filtering by industry vertical
   - **Query pattern:** `WHERE vertical = 'healthcare'`
   - **Impact:** Full table scans when filtering research jobs by vertical

3. **`projects_project.client_name`**
   - **Missing:** Index on `client_name` column
   - **Reason:** Used for searching projects by client
   - **Query pattern:** `WHERE client_name LIKE '%Acme%'` or `WHERE client_name = 'Acme Corp'`
   - **Impact:** Full table scans when searching projects (partial match LIKE won't use index anyway, but equality would)

#### MEDIUM Priority

4. **`research_competitorcasestudy.relevance_score`**
   - **Missing:** Index on `relevance_score` column
   - **Reason:** Used in Meta ordering `['-relevance_score']`
   - **Query pattern:** `ORDER BY relevance_score DESC`
   - **Impact:** Sort operations on large result sets

5. **`ideation_usecase.priority` and `ideation_usecase.impact_score`**
   - **Missing:** Composite index on `(priority, impact_score)`
   - **Reason:** Used in Meta ordering `['-priority', '-impact_score']`
   - **Query pattern:** `ORDER BY priority DESC, impact_score DESC`
   - **Impact:** Sort operations on large result sets

6. **`memory_salesplay.success_rate` and `memory_salesplay.usage_count`**
   - **Missing:** Composite index on `(success_rate, usage_count)`
   - **Reason:** Used in Meta ordering `['-success_rate', '-usage_count']`
   - **Query pattern:** `ORDER BY success_rate DESC, usage_count DESC`
   - **Impact:** Sort operations on large result sets

7. **`memory_memoryentry.client_name`**
   - **Missing:** Index on `client_name` column
   - **Reason:** Used for filtering memory entries by client
   - **Query pattern:** `WHERE client_name = 'Acme Corp'`
   - **Impact:** Full table scans when retrieving memories for a client

8. **`memory_memoryentry.industry`**
   - **Missing:** Index on `industry` column
   - **Reason:** Used for filtering memory entries by industry
   - **Query pattern:** `WHERE industry = 'Healthcare'`
   - **Impact:** Full table scans when retrieving memories by industry

#### LOW Priority (Timestamp indexes)

9. **Various `created_at` columns used in ordering**
   - Already indexed implicitly by B-tree structure and likely won't need explicit index
   - Most queries are on recent data (range scans work well)
   - Exception: If `WHERE created_at > X` is used frequently, consider adding

### Recommended Index Additions

```sql
-- HIGH PRIORITY
CREATE INDEX idx_workproduct_generic_fk ON projects_workproduct (content_type_id, object_id);
CREATE INDEX idx_annotation_generic_fk ON projects_annotation (content_type_id, object_id);
CREATE INDEX idx_researchjob_vertical ON research_researchjob (vertical) WHERE vertical IS NOT NULL;
CREATE INDEX idx_project_client_name ON projects_project (client_name);

-- MEDIUM PRIORITY
CREATE INDEX idx_competitor_relevance ON research_competitorcasestudy (relevance_score DESC);
CREATE INDEX idx_usecase_priority_impact ON ideation_usecase (priority DESC, impact_score DESC);
CREATE INDEX idx_salesplay_success_usage ON memory_salesplay (success_rate DESC, usage_count DESC);
CREATE INDEX idx_memoryentry_client ON memory_memoryentry (client_name);
CREATE INDEX idx_memoryentry_industry ON memory_memoryentry (industry);
```

---

## Enum/Choice Fields

### `research.constants.Vertical`

**Used in:** `research_researchjob.vertical`, `research_competitorcasestudy.vertical`

**Values:**
- `healthcare` - Healthcare
- `finance` - Finance
- `retail` - Retail
- `manufacturing` - Manufacturing
- `technology` - Technology
- `energy` - Energy
- `telecommunications` - Telecommunications
- `media_entertainment` - Media Entertainment
- `transportation` - Transportation
- `real_estate` - Real Estate
- `professional_services` - Professional Services
- `education` - Education
- `government` - Government
- `hospitality` - Hospitality
- `agriculture` - Agriculture
- `construction` - Construction
- `nonprofit` - Nonprofit
- `other` - Other

### `research.constants.DigitalMaturityLevel`

**Used in:** `research_researchreport.digital_maturity`

**Values:**
- `nascent` - Nascent
- `developing` - Developing
- `maturing` - Maturing
- `advanced` - Advanced
- `leading` - Leading

### `research.constants.AIAdoptionStage`

**Used in:** `research_researchreport.ai_adoption_stage`

**Values:**
- `exploring` - Exploring
- `experimenting` - Experimenting
- `implementing` - Implementing
- `scaling` - Scaling
- `optimizing` - Optimizing

### `research_researchjob.status`

**Values:**
- `pending` - Pending
- `running` - Running
- `completed` - Completed
- `failed` - Failed

### `projects_project.context_mode`

**Values:**
- `accumulate` - Build on Previous
- `fresh` - Fresh Start

### `projects_iteration.status`

**Values:**
- `pending` - Pending
- `running` - Running
- `completed` - Completed
- `failed` - Failed

### `projects_workproduct.category`

**Values:**
- `play` - Refined Play
- `persona` - Persona
- `insight` - Insight
- `one_pager` - One Pager
- `case_study` - Case Study
- `use_case` - Use Case
- `gap` - Gap Analysis
- `other` - Other

### `ideation_usecase.priority`

**Values:**
- `high` - High
- `medium` - Medium
- `low` - Low

### `ideation_usecase.status`

**Values:**
- `draft` - Draft
- `validated` - Validated
- `refined` - Refined
- `approved` - Approved
- `rejected` - Rejected

### `ideation_feasibilityassessment.overall_feasibility`

**Values:**
- `low` - Low - Significant challenges
- `medium` - Medium - Some challenges
- `high` - High - Readily achievable

### `ideation_refinedplay.status`

**Values:**
- `draft` - Draft
- `reviewed` - Reviewed
- `approved` - Approved
- `active` - Active
- `archived` - Archived

### `assets_onepager.status`

**Values:**
- `draft` - Draft
- `reviewed` - Reviewed
- `approved` - Approved
- `shared` - Shared

### `assets_accountplan.status`

**Values:**
- `draft` - Draft
- `in_progress` - In Progress
- `reviewed` - Reviewed
- `approved` - Approved
- `active` - Active

### `assets_citation.citation_type`

**Values:**
- `news` - News Article
- `website` - Company Website
- `report` - Industry Report
- `social` - Social Media
- `financial` - Financial Filing
- `press_release` - Press Release
- `other` - Other

### `memory_salesplay.play_type`

**Values:**
- `pitch` - Pitch
- `objection_handler` - Objection Handler
- `value_proposition` - Value Proposition
- `case_study` - Case Study
- `competitive_response` - Competitive Response
- `discovery_question` - Discovery Question

### `memory_memoryentry.entry_type`

**Values:**
- `research_insight` - Research Insight
- `client_interaction` - Client Interaction
- `deal_outcome` - Deal Outcome
- `best_practice` - Best Practice
- `lesson_learned` - Lesson Learned

---

## Nullable Field Analysis

### Appropriately Nullable

**OneToOne/ForeignKey relationships (nullable for optional relationships):**
- `research_researchjob.iteration_id` - NULL for standalone research jobs (not part of a project)
- `projects_workproduct.source_iteration_id` - SET_NULL, preserve work product if iteration deleted
- `assets_citation.publication_date` - Legitimate unknown publication date
- `assets_citation.verification_date` - Only set when verified

**Optional metadata/context:**
- `research_researchjob.vertical` - Unknown until classified
- `research_researchreport.founded_year` - May be unavailable from research
- All `blank=True` TEXT/VARCHAR fields - Optional descriptive content

### Potentially Inappropriate Nullable

**None identified.** The schema appropriately uses `blank=True` for empty strings on TEXT/VARCHAR fields rather than NULL, following Django best practices.

### Fields that Should NOT be Nullable

All non-nullable fields are appropriate:
- Primary keys (UUID, always generated)
- Foreign keys for required relationships
- Status fields (have defaults)
- Timestamps (auto_now_add/auto_now)
- JSONFields with default factories

---

## Data Completeness Assessment

### Fully Populated by Application

**Core Workflow Models:**
- `research_researchjob` ✓ - Created by API, updated by LangGraph workflow
- `research_researchreport` ✓ - Populated by `conduct_research` node via Gemini
- `research_competitorcasestudy` ✓ - Populated by `find_competitors` node
- `research_gapanalysis` ✓ - Populated by `analyze_gaps` node
- `research_internalopsintel` ✓ - Populated by `gather_internal_ops` node
- `projects_project` ✓ - Created by projects API
- `projects_iteration` ✓ - Created by projects API, linked to ResearchJob
- `prompts_prompttemplate` ✓ - Managed via Django admin or API

### Partially Populated by Application

**Generated Assets (depends on user request):**
- `ideation_usecase` ~ - Generated by ideation service (Epic 4)
- `ideation_feasibilityassessment` ~ - Generated for selected use cases
- `ideation_refinedplay` ~ - Generated from validated use cases
- `assets_persona` ~ - Generated by persona service (Epic 5)
- `assets_onepager` ~ - Generated by one-pager service (Epic 5)
- `assets_accountplan` ~ - Generated by account plan service (Epic 5)

**Work Product Tracking (user-initiated):**
- `projects_workproduct` ~ - Created when user stars/saves items
- `projects_annotation` ~ - Created when user adds notes

### Appears NEVER Populated

**`assets_citation` ⚠️**
- **Status:** Model exists, migration created, but no service code populates it
- **Evidence:**
  - No service file in `assets/services/` for citations
  - Not referenced in `research/graph/nodes.py` workflow
  - Not mentioned in serializers or views
  - `web_sources` field on ResearchReport serves similar purpose (grounding metadata)
- **Assessment:** Likely planned feature (AGE-24) that was not implemented, superseded by `web_sources` field

### Memory Module (External Workflow)

**Uncertain - No Evidence of Population:**
- `memory_clientprofile` ? - Intended for vector store, no clear population code
- `memory_salesplay` ? - Intended for vector store, no clear population code
- `memory_memoryentry` ? - Intended for vector store, no clear population code

These models have service files (`memory/services/vectorstore.py`, `memory/services/capture.py`) but unclear if they're actively used in the main workflow.

---

## Schema Concerns

### CRITICAL Issues

1. **Missing GenericFK Composite Indexes**
   - **Tables:** `projects_workproduct`, `projects_annotation`
   - **Issue:** No composite index on `(content_type_id, object_id)` for GenericFK lookups
   - **Impact:** Full table scans when retrieving work products or annotations for a specific object
   - **Fix:** Add composite indexes (see Index Audit section)

2. **Citation Model Appears Unused**
   - **Table:** `assets_citation`
   - **Issue:** Model defined and migrated but never populated by application code
   - **Impact:** Dead code, unused database space
   - **Fix:** Either implement citation capture or remove model/table
   - **Note:** `web_sources` field on ResearchReport may have superseded this

### HIGH Priority Issues

3. **Missing Vertical Classification Index**
   - **Table:** `research_researchjob`
   - **Field:** `vertical`
   - **Issue:** No index on frequently-filtered column
   - **Impact:** Full table scans when filtering by vertical
   - **Fix:** Add partial index `WHERE vertical IS NOT NULL`

4. **No Index on Project Client Name**
   - **Table:** `projects_project`
   - **Field:** `client_name`
   - **Issue:** No index on search column
   - **Impact:** Full table scans when searching projects by client
   - **Fix:** Add index on `client_name` (note: won't help ILIKE/LIKE searches)

5. **Potential Decision Maker Denormalization**
   - **Table:** `research_researchreport`
   - **Field:** `decision_makers` (JSONField)
   - **Issue:** Structured entities stored as JSON, not queryable
   - **Impact:** Cannot query across all decision makers (e.g., "find all CTOs"), cannot track updates over time
   - **Fix:** Consider new `DecisionMaker` table with FK to ResearchReport

### MEDIUM Priority Issues

6. **Missing Ordering Field Indexes**
   - **Tables:** `research_competitorcasestudy`, `ideation_usecase`, `memory_salesplay`
   - **Fields:** `relevance_score`, `priority`/`impact_score`, `success_rate`/`usage_count`
   - **Issue:** Fields used in Meta ordering lack indexes
   - **Impact:** Slow sort operations on large result sets
   - **Fix:** Add indexes on ordering fields

7. **Memory Module Tables Have No FK Relationships**
   - **Tables:** `memory_clientprofile`, `memory_salesplay`, `memory_memoryentry`
   - **Issue:** No foreign key relationships, rely on `client_name` string matching
   - **Impact:** Data integrity concerns, orphaned records, inconsistent naming
   - **Fix:** Consider FK to `projects_project` or new `Client` entity

8. **No Index on Memory Entry Filters**
   - **Table:** `memory_memoryentry`
   - **Fields:** `client_name`, `industry`
   - **Issue:** Filtering fields lack indexes
   - **Impact:** Full table scans when retrieving memories
   - **Fix:** Add indexes on filter fields

### LOW Priority Issues

9. **Founded Year Should Be SmallInt Not Integer**
   - **Table:** `research_researchreport`
   - **Field:** `founded_year`
   - **Issue:** Uses INTEGER (4 bytes) for year value (1000-9999 range)
   - **Impact:** Marginal storage waste
   - **Fix:** Use SmallIntegerField (2 bytes) or PositiveSmallIntegerField

10. **CharField Used for Numeric-Looking Fields**
   - **Tables:** `research_researchreport`, `ideation_usecase`
   - **Fields:** `employee_count`, `annual_revenue`, `estimated_roi`, `time_to_value`
   - **Issue:** Storing ranges as strings (e.g., "1,000-5,000", "$500M-$1B")
   - **Impact:** Cannot perform numeric queries or aggregations
   - **Rationale:** Acceptable design choice - AI generates human-readable ranges, not precise numbers
   - **Alternative:** Could use IntegerField for min/max ranges, but current approach is simpler

11. **PDF Path Stored as CharField**
   - **Tables:** `assets_onepager`, `assets_accountplan`
   - **Field:** `pdf_path`
   - **Issue:** Should use FileField or FilePathField
   - **Impact:** No path validation, no storage backend integration
   - **Fix:** Use FileField with proper storage backend

12. **Prompts App Uses Different PK Strategy**
   - **Table:** `prompts_prompttemplate`
   - **Field:** `id` (BigAutoField vs UUID everywhere else)
   - **Issue:** Inconsistent PK strategy across schema
   - **Impact:** Minimal, but inconsistent
   - **Fix:** Standardize on UUID for consistency (breaking change)

### Data Integrity Concerns

13. **CASCADE Deletes Everywhere**
   - **Impact:** Deleting a ResearchJob cascades to all reports, case studies, personas, etc.
   - **Risk:** Accidental data loss if research job deleted
   - **Mitigation:** Consider PROTECT on critical relationships or soft-delete pattern

14. **No Soft Delete Pattern**
   - **Issue:** Hard deletes throughout, no `deleted_at` timestamp pattern
   - **Impact:** Cannot recover accidentally deleted data, cannot audit deletions
   - **Fix:** Consider adding soft delete fields and manager

15. **No Audit Trail**
   - **Issue:** No change tracking, no `modified_by` fields
   - **Impact:** Cannot track who made changes or when
   - **Fix:** Consider Django Simple History or custom audit log

### JSONField Schema Validation

16. **No JSONField Schema Validation**
   - **Issue:** All JSONFields accept arbitrary JSON, no schema enforcement
   - **Impact:** Inconsistent data structures, runtime errors
   - **Fix:** Add JSONSchema validation in model `clean()` methods or use django-jsonschema-field

### Performance Concerns

17. **Large TEXT Fields**
   - **Fields:** `research_researchjob.result`, `research_researchreport.company_overview`, etc.
   - **Issue:** Large TEXT fields included in every SELECT
   - **Impact:** Increased memory/network overhead
   - **Mitigation:** Use `defer()` or `only()` in queries, or separate into related table

18. **No Pagination Guidance**
   - **Issue:** No explicit pagination strategy documented
   - **Impact:** Large result sets could cause memory issues
   - **Mitigation:** DRF provides pagination, ensure it's configured

### Security Concerns

19. **No Row Level Security**
   - **Issue:** No user/tenant isolation in schema
   - **Impact:** All data visible to all authenticated users
   - **Mitigation:** Implement RLS in PostgreSQL or application-level filtering

20. **No User Tracking**
   - **Issue:** No `created_by` or `owner` fields
   - **Impact:** Cannot attribute data to users, cannot implement ownership-based access control
   - **Fix:** Add User FK fields and filter by user in queries

---

## Recommended Actions

### Immediate (Critical)

1. Add composite indexes for GenericFK fields
2. Add index on `research_researchjob.vertical`
3. Add index on `projects_project.client_name`
4. Investigate and either implement or remove `assets_citation` model

### Short Term (High)

5. Add indexes on ordering fields (relevance_score, priority, impact_score, etc.)
6. Consider normalizing `decision_makers` JSONField to separate table
7. Add indexes on memory entry filter fields

### Medium Term (Improvements)

8. Implement soft delete pattern
9. Add JSONField schema validation
10. Add user tracking fields (created_by, owner)
11. Document and enforce JSONField structures

### Long Term (Architecture)

12. Consider row-level security strategy
13. Evaluate memory module FK relationships
14. Implement audit trail/change tracking
15. Review CASCADE delete strategy for data protection

---

## Conclusion

The Deep Prospecting Engine database schema is well-structured for its AI-powered research workflow, with clear hierarchical relationships and appropriate use of JSONFields for AI-generated content. The primary concerns are:

1. Missing indexes on GenericFK fields and commonly-filtered columns
2. An apparently unused `Citation` model
3. Lack of user tracking and audit trail
4. No schema validation on JSONFields

These issues should be addressed to improve query performance, data integrity, and maintainability as the system scales.

---

**End of Data Dictionary**
