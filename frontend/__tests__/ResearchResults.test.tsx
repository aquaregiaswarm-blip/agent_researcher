import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import ResearchResults from '@/components/ResearchResults';
import { ResearchJob, GapAnalysis, InternalOpsIntel, CompetitorCaseStudy, ResearchReport } from '@/types';

vi.mock('@/lib/api', () => ({
  api: {
    downloadResearchPdf: vi.fn(),
  },
}));

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const baseReport: ResearchReport = {
  id: 'r1',
  company_overview: 'A leading tech company.',
  founded_year: 2000,
  headquarters: 'San Francisco, CA',
  employee_count: '5000',
  annual_revenue: '$1B',
  website: 'https://example.com',
  recent_news: [],
  decision_makers: [],
  pain_points: ['Legacy systems', 'Data silos'],
  opportunities: ['Cloud migration', 'AI adoption'],
  digital_maturity: 'maturing',
  ai_footprint: 'Uses **ML models** for demand forecasting.',
  ai_adoption_stage: 'implementing',
  strategic_goals: ['Modernise infrastructure'],
  key_initiatives: ['Cloud-first programme'],
  talking_points: ['Strong ROI from cloud'],
  cloud_footprint: 'AWS heavy with some **Azure** for Office 365.',
  security_posture: 'SOC2 Type II certified.',
  data_maturity: 'Advanced analytics capability.',
  financial_signals: ['Series B funding'],
  tech_partnerships: ['AWS', 'Snowflake'],
  web_sources: [{ uri: 'https://source1.com', title: 'Source 1' }],
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
};

const gapAnalysis: GapAnalysis = {
  id: 'g1',
  technology_gaps: ['**Gap 1:** Missing real-time data pipeline'],
  capability_gaps: ['ML expertise gap'],
  process_gaps: ['Manual reporting'],
  recommendations: ['Implement streaming platform'],
  priority_areas: ['**#1 Priority:** Data infrastructure'],
  confidence_score: 0.82,
  analysis_notes: 'Analysis based on **public data** and job postings.\n\n- Finding one\n- Finding two',
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
};

const internalOps: InternalOpsIntel = {
  id: 'i1',
  employee_sentiment: {
    overall_rating: 3.8,
    work_life_balance: 3.5,
    compensation: 4.0,
    culture: 3.7,
    management: 3.4,
    recommend_pct: 72,
    positive_themes: ['Good benefits'],
    negative_themes: ['Long hours'],
    trend: 'stable',
  },
  linkedin_presence: {
    follower_count: 50000,
    engagement_level: 'medium',
    recent_posts: [],
    employee_trend: 'growing',
    notable_changes: [],
  },
  social_media_mentions: [],
  job_postings: {
    total_openings: 45,
    departments_hiring: { Engineering: 20 },
    skills_sought: ['Python', 'Cloud'],
    seniority_distribution: { Senior: 15 },
    urgency_signals: [],
    insights: 'Heavy focus on **technical hiring** and AI roles.',
  },
  news_sentiment: {
    overall_sentiment: 'positive',
    coverage_volume: 'medium',
    topics: ['Product launch'],
    headlines: [],
  },
  key_insights: [
    '**Strong hiring** suggests growth phase',
    'Culture challenges noted in reviews',
  ],
  gap_correlations: [
    {
      gap_type: 'technology',
      description: 'Missing cloud infrastructure',
      evidence: 'Heavy hiring for **cloud engineers**',
      evidence_type: 'supporting',
      confidence: 0.85,
      sales_implication: 'Strong opportunity for cloud platform sales',
    },
  ],
  confidence_score: 0.75,
  data_freshness: 'last_30_days',
  analysis_notes: 'Data sourced from **Glassdoor**, LinkedIn, and news APIs.',
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
};

const competitor: CompetitorCaseStudy = {
  id: 'c1',
  competitor_name: 'RivalCo',
  vertical: 'technology',
  case_study_title: 'AI-Powered Supply Chain',
  summary: 'Deployed ML to cut costs.',
  technologies_used: ['Python', 'TensorFlow'],
  outcomes: ['20% cost reduction'],
  source_url: 'https://rival.co',
  relevance_score: 0.85,
  created_at: '2024-01-01T00:00:00Z',
};

const completedJob: ResearchJob = {
  id: 'j1',
  client_name: 'TestCo',
  sales_history: '',
  prompt: '',
  status: 'completed',
  result: 'Raw research output',
  error: '',
  vertical: 'technology',
  report: baseReport,
  competitor_case_studies: [competitor],
  gap_analysis: gapAnalysis,
  internal_ops: internalOps,
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
};


// ---------------------------------------------------------------------------
// Tab visibility
// ---------------------------------------------------------------------------

describe('ResearchResults — tab visibility', () => {
  it('renders overview tab always', () => {
    render(<ResearchResults job={completedJob} />);
    expect(screen.getByRole('button', { name: 'Overview' })).toBeInTheDocument();
  });

  it('shows competitors tab when data present', () => {
    render(<ResearchResults job={completedJob} />);
    expect(screen.getByRole('button', { name: 'Competitors' })).toBeInTheDocument();
  });

  it('shows gaps tab when gap_analysis present', () => {
    render(<ResearchResults job={completedJob} />);
    expect(screen.getByRole('button', { name: 'Gap Analysis' })).toBeInTheDocument();
  });

  it('shows intel tab when internal_ops present', () => {
    render(<ResearchResults job={completedJob} />);
    expect(screen.getByRole('button', { name: 'Inside Intel' })).toBeInTheDocument();
  });

  it('shows sources tab when web_sources present', () => {
    render(<ResearchResults job={completedJob} />);
    expect(screen.getByRole('button', { name: 'Sources' })).toBeInTheDocument();
  });

  it('hides competitors tab when no competitor data', () => {
    const job = { ...completedJob, competitor_case_studies: [] };
    render(<ResearchResults job={job} />);
    expect(screen.queryByRole('button', { name: 'Competitors' })).toBeNull();
  });

  it('hides gaps tab when gap_analysis absent', () => {
    const job = { ...completedJob, gap_analysis: undefined };
    render(<ResearchResults job={job} />);
    expect(screen.queryByRole('button', { name: 'Gap Analysis' })).toBeNull();
  });
});


// ---------------------------------------------------------------------------
// Gap Analysis tab rendering
// ---------------------------------------------------------------------------

describe('ResearchResults — Gap Analysis tab', () => {
  function renderGapTab() {
    render(<ResearchResults job={completedJob} />);
    fireEvent.click(screen.getByRole('button', { name: 'Gap Analysis' }));
  }

  it('renders confidence score', () => {
    renderGapTab();
    expect(screen.getByText(/82%/)).toBeInTheDocument();
  });

  it('renders technology gaps section', () => {
    renderGapTab();
    expect(screen.getByText('Technology Gaps')).toBeInTheDocument();
  });

  it('renders capability gaps section', () => {
    renderGapTab();
    expect(screen.getByText('Capability Gaps')).toBeInTheDocument();
  });

  it('renders process gaps section', () => {
    renderGapTab();
    expect(screen.getByText('Process Gaps')).toBeInTheDocument();
  });

  it('renders recommendations section', () => {
    renderGapTab();
    expect(screen.getByText('Recommendations')).toBeInTheDocument();
  });

  it('renders priority areas section', () => {
    renderGapTab();
    expect(screen.getByText('Priority Areas')).toBeInTheDocument();
  });

  it('renders analysis notes section', () => {
    renderGapTab();
    expect(screen.getByText('Analysis Notes')).toBeInTheDocument();
  });

  it('does not show raw asterisks in priority areas (regression — TJX issue)', () => {
    renderGapTab();
    // The markdown should render as formatted text, not raw **...**
    const container = document.querySelector('[data-testid]') || document.body;
    expect(container.innerHTML).not.toMatch(/\*\*#1 Priority\*\*/);
  });

  it('renders bold text in analysis_notes as strong elements (markdown)', () => {
    renderGapTab();
    // analysis_notes contains "**public data**" — should render as <strong>
    const strongs = document.querySelectorAll('strong');
    const strongTexts = Array.from(strongs).map(s => s.textContent);
    expect(strongTexts.some(t => t?.includes('public data'))).toBe(true);
  });
});


// ---------------------------------------------------------------------------
// Inside Intel tab rendering
// ---------------------------------------------------------------------------

describe('ResearchResults — Inside Intel tab', () => {
  function renderIntelTab() {
    render(<ResearchResults job={completedJob} />);
    fireEvent.click(screen.getByRole('button', { name: 'Inside Intel' }));
  }

  it('renders employee sentiment section', () => {
    renderIntelTab();
    expect(screen.getByText('Employee Sentiment Overview')).toBeInTheDocument();
  });

  it('renders overall rating', () => {
    renderIntelTab();
    expect(screen.getByText('3.8/5.0')).toBeInTheDocument();
  });

  it('renders job postings total', () => {
    renderIntelTab();
    expect(screen.getByText('45')).toBeInTheDocument();
  });

  it('renders key insights section', () => {
    renderIntelTab();
    expect(screen.getByText('Key Insights & Recommendations')).toBeInTheDocument();
  });

  it('renders gap correlations section', () => {
    renderIntelTab();
    expect(screen.getByText('Gap Correlation Insights')).toBeInTheDocument();
  });

  it('does not show raw asterisks in key_insights (markdown rendered)', () => {
    renderIntelTab();
    expect(document.body.innerHTML).not.toMatch(/\*\*Strong hiring\*\*/);
  });

  it('renders analysis_notes as markdown not raw text', () => {
    renderIntelTab();
    // analysis_notes has **Glassdoor** — should render as <strong>, not raw **
    expect(document.body.innerHTML).not.toMatch(/\*\*Glassdoor\*\*/);
  });
});


// ---------------------------------------------------------------------------
// Deep Research tab — enriched fields
// ---------------------------------------------------------------------------

describe('ResearchResults — Deep Research tab', () => {
  function renderReportTab() {
    render(<ResearchResults job={completedJob} />);
    fireEvent.click(screen.getByRole('button', { name: 'Deep Research' }));
  }

  it('renders cloud footprint section', () => {
    renderReportTab();
    expect(screen.getByText('Cloud Footprint')).toBeInTheDocument();
  });

  it('renders security posture section', () => {
    renderReportTab();
    expect(screen.getByText('Security Posture')).toBeInTheDocument();
  });

  it('renders data maturity section', () => {
    renderReportTab();
    expect(screen.getByText('Data Maturity')).toBeInTheDocument();
  });

  it('renders financial signals', () => {
    renderReportTab();
    expect(screen.getByText('Financial Signals')).toBeInTheDocument();
  });

  it('renders tech partnerships', () => {
    renderReportTab();
    expect(screen.getByText('Technology Partnerships')).toBeInTheDocument();
  });

  it('renders cloud_footprint markdown (no raw asterisks)', () => {
    renderReportTab();
    expect(document.body.innerHTML).not.toMatch(/\*\*Azure\*\*/);
  });
});


// ---------------------------------------------------------------------------
// Overview tab — null report fallback
// ---------------------------------------------------------------------------

describe('ResearchResults — Overview tab null report fallback', () => {
  it('shows fallback message when report is null', () => {
    const job = { ...completedJob, report: undefined };
    render(<ResearchResults job={job} />);
    expect(screen.getByText('Structured data not available for this research job.')).toBeInTheDocument();
  });

  it('does not crash when report is null', () => {
    const job = { ...completedJob, report: undefined };
    expect(() => render(<ResearchResults job={job} />)).not.toThrow();
  });

  it('does not render a StatCard for data_maturity — prose section used instead', () => {
    render(<ResearchResults job={completedJob} />);
    // Overview tab is active by default
    // data_maturity should appear under a "Data Maturity" section heading, not as a plain stat card value
    expect(screen.getByText('Data Maturity')).toBeInTheDocument();
    // The value should be rendered as prose (via MarkdownText), visible in the DOM
    expect(screen.getByText('Advanced analytics capability.')).toBeInTheDocument();
  });

  it('renders data_maturity as prose text, not truncated in a stat card', () => {
    const longDataMaturity = 'The organisation uses Snowflake for data warehousing, dbt for transformation, and Tableau for visualisation. Governance is managed via a central data team with a defined data mesh strategy.';
    const reportWithLongDM = { ...baseReport, data_maturity: longDataMaturity };
    const job = { ...completedJob, report: reportWithLongDM };
    render(<ResearchResults job={job} />);
    expect(screen.getByText(longDataMaturity)).toBeInTheDocument();
  });
});


// ---------------------------------------------------------------------------
// Gap Analysis tab — parsing failure state
// ---------------------------------------------------------------------------

describe('ResearchResults — Gap Analysis tab parsing failure', () => {
  function renderGapTabWithGaps(gaps: GapAnalysis) {
    const job = { ...completedJob, gap_analysis: gaps };
    render(<ResearchResults job={job} />);
    fireEvent.click(screen.getByRole('button', { name: 'Gap Analysis' }));
  }

  it('shows clean error state when analysis_notes starts with "Analysis parsing failed" and all arrays are empty', () => {
    const corruptGaps: GapAnalysis = {
      ...gapAnalysis,
      analysis_notes: 'Analysis parsing failed. Raw output: ```json { ... }```',
      technology_gaps: [],
      capability_gaps: [],
      process_gaps: [],
      confidence_score: 0.0,
    };
    renderGapTabWithGaps(corruptGaps);
    expect(screen.getByText('Gap analysis could not be parsed for this research job.')).toBeInTheDocument();
    expect(screen.getByText('Please re-run the research to regenerate gap analysis.')).toBeInTheDocument();
  });

  it('does not show error state when arrays have data even if analysis_notes is empty', () => {
    const gapsWithContent: GapAnalysis = { ...gapAnalysis, analysis_notes: '' };
    renderGapTabWithGaps(gapsWithContent);
    expect(screen.queryByText('Gap analysis could not be parsed for this research job.')).toBeNull();
    expect(screen.getByText('Technology Gaps')).toBeInTheDocument();
  });

  it('does not show error state when confidence_score is 0 but gaps have data', () => {
    const lowConfidenceGaps: GapAnalysis = { ...gapAnalysis, confidence_score: 0.0, analysis_notes: '' };
    renderGapTabWithGaps(lowConfidenceGaps);
    expect(screen.queryByText('Gap analysis could not be parsed for this research job.')).toBeNull();
    expect(screen.getByText('Technology Gaps')).toBeInTheDocument();
  });
});


// ---------------------------------------------------------------------------
// Running / failed states
// ---------------------------------------------------------------------------

describe('ResearchResults — loading and error states', () => {
  it('shows spinner when status is running', () => {
    const job: ResearchJob = { ...completedJob, status: 'running' };
    render(<ResearchResults job={job} />);
    expect(screen.getByText('Researching...')).toBeInTheDocument();
  });

  it('shows error message when status is failed', () => {
    const job: ResearchJob = { ...completedJob, status: 'failed', error: 'API quota exceeded' };
    render(<ResearchResults job={job} />);
    expect(screen.getByText('API quota exceeded')).toBeInTheDocument();
  });
});
