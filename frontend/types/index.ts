// Web source from Google Search grounding
export interface WebSource {
  uri: string;
  title: string;
}

// Decision maker from research report
export interface DecisionMaker {
  name: string;
  title: string;
  background: string;
  linkedin_url?: string;
}

// News item from research report
export interface NewsItem {
  title: string;
  summary: string;
  date: string;
  source: string;
  url?: string;
}

// Structured research report (AGE-10)
export interface ResearchReport {
  id: string;
  company_overview: string;
  founded_year?: number;
  headquarters: string;
  employee_count: string;
  annual_revenue: string;
  website: string;
  recent_news: NewsItem[];
  decision_makers: DecisionMaker[];
  pain_points: string[];
  opportunities: string[];
  digital_maturity: string;
  ai_footprint: string;
  ai_adoption_stage: string;
  strategic_goals: string[];
  key_initiatives: string[];
  talking_points: string[];
  cloud_footprint?: string;
  security_posture?: string;
  data_maturity?: string;
  financial_signals?: string[];
  tech_partnerships?: (string | Record<string, string>)[];
  web_sources?: WebSource[];
  created_at: string;
  updated_at: string;
}

// Competitor case study (AGE-12)
export interface CompetitorCaseStudy {
  id: string;
  competitor_name: string;
  vertical: string;
  case_study_title: string;
  summary: string;
  technologies_used: string[];
  outcomes: string[];
  source_url: string;
  relevance_score: number;
  created_at: string;
}

// Gap analysis (AGE-13)
export interface GapAnalysis {
  id: string;
  technology_gaps: string[];
  capability_gaps: string[];
  process_gaps: string[];
  recommendations: string[];
  priority_areas: string[];
  confidence_score: number;
  analysis_notes: string;
  created_at: string;
  updated_at: string;
}

// Internal Operations Intelligence types (AGE-20)

export interface EmployeeSentiment {
  overall_rating: number;
  work_life_balance: number;
  compensation: number;
  culture: number;
  management: number;
  recommend_pct: number;
  positive_themes: string[];
  negative_themes: string[];
  trend: 'improving' | 'declining' | 'stable';
}

export interface LinkedInPost {
  title: string;
  summary: string;
  date: string;
}

export interface LinkedInPresence {
  follower_count: number;
  engagement_level: 'low' | 'medium' | 'high';
  recent_posts: LinkedInPost[];
  employee_trend: 'growing' | 'shrinking' | 'stable';
  notable_changes: string[];
}

export interface SocialMediaMention {
  platform: 'reddit' | 'twitter' | 'facebook' | string;
  summary: string;
  sentiment: 'positive' | 'negative' | 'neutral' | 'mixed';
  topic: string;
}

export interface JobPostings {
  total_openings: number;
  departments_hiring: Record<string, number>;
  skills_sought: string[];
  seniority_distribution: Record<string, number>;
  urgency_signals: string[];
  insights: string;
}

export interface NewsHeadline {
  title: string;
  source: string;
  date: string;
  sentiment: 'positive' | 'negative' | 'neutral' | 'mixed';
}

export interface NewsSentiment {
  overall_sentiment: 'positive' | 'negative' | 'neutral' | 'mixed';
  coverage_volume: 'low' | 'medium' | 'high';
  topics: string[];
  headlines: NewsHeadline[];
}

export interface GapCorrelation {
  gap_type: 'technology' | 'capability' | 'process';
  description: string;
  evidence: string;
  evidence_type: 'supporting' | 'contradicting' | 'neutral';
  confidence: number;
  sales_implication: string;
}

export interface InternalOpsIntel {
  id: string;
  employee_sentiment: EmployeeSentiment;
  linkedin_presence: LinkedInPresence;
  social_media_mentions: SocialMediaMention[];
  job_postings: JobPostings;
  news_sentiment: NewsSentiment;
  key_insights: string[];
  gap_correlations: GapCorrelation[];
  confidence_score: number;
  data_freshness: string;
  analysis_notes: string;
  created_at: string;
  updated_at: string;
}

// Main research job with nested data
export interface ResearchJob {
  id: string;
  client_name: string;
  sales_history: string;
  prompt: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  current_step?: string;
  result: string;
  error: string;
  vertical?: string;
  report?: ResearchReport;
  competitor_case_studies?: CompetitorCaseStudy[];
  gap_analysis?: GapAnalysis;
  internal_ops?: InternalOpsIntel;
  created_at: string;
  updated_at: string;
}

export interface ResearchFormData {
  client_name: string;
  sales_history: string;
  prompt: string;
}

export interface PromptTemplate {
  id: number;
  name: string;
  content: string;
  is_default: boolean;
  created_at: string;
  updated_at: string;
}

// Project-based iterative workflow types

export type ContextMode = 'accumulate' | 'fresh';
export type IterationStatus = 'pending' | 'running' | 'completed' | 'failed';
export type WorkProductCategory = 'play' | 'persona' | 'insight' | 'one_pager' | 'case_study' | 'use_case' | 'gap' | 'other';

// Project list item (lightweight)
export interface ProjectListItem {
  id: string;
  name: string;
  client_name: string;
  context_mode: ContextMode;
  iteration_count: number;
  latest_iteration_status?: IterationStatus;
  created_at: string;
  updated_at: string;
}

// Iteration list item (lightweight)
export interface IterationListItem {
  id: string;
  sequence: number;
  name?: string;
  status: IterationStatus;
  has_research_job: boolean;
  research_job_id?: string;
  research_job_status?: IterationStatus;
  created_at: string;
}

// Full iteration details
export interface Iteration extends IterationListItem {
  project: string;
  sales_history: string;
  prompt_override: string;
  inherited_context: Record<string, unknown>;
  work_products: WorkProduct[];
}

// Work product content preview
export interface WorkProductPreview {
  title?: string;
  summary?: string;
}

// Work product (starred item)
export interface WorkProduct {
  id: string;
  project: string;
  content_type: string;
  content_type_name: string;
  object_id: string;
  content_preview?: WorkProductPreview;
  source_iteration?: string;
  source_iteration_sequence?: number;
  category: WorkProductCategory;
  starred: boolean;
  notes: string;
  created_at: string;
}

// Annotation (user note)
export interface Annotation {
  id: string;
  project: string;
  content_type: string;
  content_type_name: string;
  object_id: string;
  text: string;
  created_at: string;
  updated_at: string;
}

// Full project details
export interface Project extends ProjectListItem {
  description: string;
  iterations: IterationListItem[];
  work_products_count: number;
  annotations_count: number;
}

// Project creation data
export interface ProjectCreateData {
  name: string;
  client_name: string;
  description?: string;
  context_mode?: ContextMode;
}

// Iteration creation data
export interface IterationCreateData {
  name?: string;
  sales_history?: string;
  prompt_override?: string;
}

// Timeline view data
export interface TimelineData {
  iterations: IterationListItem[];
  work_products_by_iteration: Record<number, WorkProduct[]>;
}

// Iteration comparison data
export interface IterationComparison {
  iteration_a: IterationComparisonData;
  iteration_b: IterationComparisonData;
  differences: {
    pain_points: ListDiff;
    opportunities: ListDiff;
    talking_points: ListDiff;
  };
}

export interface IterationComparisonData {
  id: string;
  sequence: number;
  name: string;
  status: IterationStatus;
  created_at: string;
  research_job?: {
    id: string;
    client_name: string;
    status: string;
    vertical?: string;
  };
  report?: {
    company_overview: string;
    pain_points: string[];
    opportunities: string[];
    digital_maturity: string;
    ai_adoption_stage: string;
    talking_points: string[];
    decision_makers: DecisionMaker[];
    data_maturity?: string;
  };
  gap_analysis?: {
    technology_gaps: string[];
    capability_gaps: string[];
    process_gaps: string[];
    recommendations: string[];
    priority_areas: string[];
  };
  use_cases_count?: number;
  personas_count?: number;
  competitor_case_studies_count?: number;
}

export interface ListDiff {
  added: string[];
  removed: string[];
  unchanged: string[];
}

// Ideation types (AGE-18, AGE-19, AGE-20)

export type UseCasePriority = 'high' | 'medium' | 'low';
export type UseCaseStatus = 'draft' | 'validated' | 'refined' | 'approved' | 'rejected';
export type FeasibilityLevel = 'high' | 'medium' | 'low';
export type PlayStatus = 'draft' | 'reviewed' | 'approved' | 'active' | 'archived';

export interface UseCase {
  id: string;
  research_job: string;
  title: string;
  description: string;
  business_problem: string;
  proposed_solution: string;
  expected_benefits: string[];
  estimated_roi: string;
  time_to_value: string;
  technologies: string[];
  data_requirements: string[];
  integration_points: string[];
  priority: UseCasePriority;
  impact_score: number;
  feasibility_score: number;
  status: UseCaseStatus;
  created_at: string;
  updated_at: string;
}

export interface FeasibilityAssessment {
  id: string;
  use_case: string;
  overall_feasibility: FeasibilityLevel;
  overall_score: number;
  technical_complexity: string;
  data_availability: string;
  integration_complexity: string;
  scalability_considerations: string;
  technical_risks: string[];
  mitigation_strategies: string[];
  prerequisites: string[];
  dependencies: string[];
  recommendations: string[];
  next_steps: string[];
  created_at: string;
}

export interface RefinedPlay {
  id: string;
  use_case: string;
  title: string;
  elevator_pitch: string;
  value_proposition: string;
  key_differentiators: string[];
  target_persona: string;
  target_vertical: string;
  company_size_fit: string;
  discovery_questions: string[];
  objection_handlers: string[];
  proof_points: string[];
  competitive_positioning: string;
  next_steps: string[];
  success_metrics: string[];
  status: PlayStatus;
  created_at: string;
  updated_at: string;
}

// Asset types (AGE-21, AGE-22, AGE-23)

export type AssetStatus = 'draft' | 'reviewed' | 'approved' | 'shared';

export interface Persona {
  id: string;
  research_job: string;
  name: string;
  title: string;
  department: string;
  seniority_level: string;
  background: string;
  goals: string[];
  challenges: string[];
  motivations: string[];
  decision_criteria: string[];
  preferred_communication: string[];
  objections: string[];
  content_preferences: string[];
  key_messages: string[];
  created_at: string;
}

export interface OnePager {
  id: string;
  research_job: string;
  title: string;
  headline: string;
  executive_summary: string;
  challenge_section: string;
  solution_section: string;
  benefits_section: string;
  differentiators: string[];
  call_to_action: string;
  next_steps: string[];
  html_content: string;
  pdf_path: string;
  status: AssetStatus;
  created_at: string;
  updated_at: string;
}

export interface AccountPlan {
  id: string;
  research_job: string;
  title: string;
  executive_summary: string;
  account_overview: string;
  strategic_objectives: string[];
  key_stakeholders: Record<string, unknown>[];
  opportunities: Record<string, unknown>[];
  competitive_landscape: string;
  swot_analysis: {
    strengths: string[];
    weaknesses: string[];
    opportunities: string[];
    threats: string[];
  };
  engagement_strategy: string;
  value_propositions: string[];
  action_plan: Record<string, unknown>[];
  success_metrics: string[];
  milestones: Record<string, unknown>[];
  timeline: Record<string, unknown>;
  html_content: string;
  pdf_path: string;
  status: AssetStatus;
  created_at: string;
  updated_at: string;
}

// Citation types (AGE-24)

export type CitationType =
  | 'news'
  | 'website'
  | 'report'
  | 'social'
  | 'financial'
  | 'press_release'
  | 'other';

export interface Citation {
  id: string;
  research_job: string;
  citation_type: CitationType;
  title: string;
  source: string;
  url: string;
  author: string;
  publication_date: string | null;
  excerpt: string;
  relevance_note: string;
  verified: boolean;
  verification_date: string | null;
  created_at: string;
}

// Memory / Knowledge Base types (AGE-14, AGE-15, AGE-16, AGE-17)

export interface ClientProfile {
  id: string;
  client_name: string;
  industry: string;
  company_size: string;
  region: string;
  key_contacts: Record<string, unknown>[];
  summary: string;
  vector_id: string;
  created_at: string;
  updated_at: string;
}

export type SalesPlayType =
  | 'pitch'
  | 'objection_handler'
  | 'value_proposition'
  | 'case_study'
  | 'competitive_response'
  | 'discovery_question';

export interface SalesPlay {
  id: string;
  title: string;
  play_type: SalesPlayType;
  content: string;
  context: string;
  industry: string;
  vertical: string;
  usage_count: number;
  success_rate: number;
  created_at: string;
  updated_at: string;
}

export type MemoryEntryType =
  | 'research_insight'
  | 'client_interaction'
  | 'deal_outcome'
  | 'best_practice'
  | 'lesson_learned';

export interface MemoryEntry {
  id: string;
  entry_type: MemoryEntryType;
  title: string;
  content: string;
  client_name: string;
  industry: string;
  tags: string[];
  source_type: string;
  source_id: string;
  created_at: string;
  updated_at: string;
}
