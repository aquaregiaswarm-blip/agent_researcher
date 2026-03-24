import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import AccountPlanSection from '@/components/research-results/generate/AccountPlanSection';
import { AccountPlan } from '@/types';

vi.mock('@/lib/api', () => ({ api: {} }));
vi.mock('dompurify', () => ({
  default: { sanitize: (html: string) => html },
}));

const accountPlan: AccountPlan = {
  id: 'ap1',
  research_job: 'j1',
  title: 'Acme Strategic Account Plan',
  executive_summary: 'Acme is a strategic target for cloud solutions.',
  account_overview: 'Acme Corp overview.',
  strategic_objectives: ['Become cloud-first by 2026', 'Reduce IT costs by 30%'],
  key_stakeholders: [],
  opportunities: [],
  competitive_landscape: 'Microsoft and AWS compete.',
  swot_analysis: {
    strengths: ['Strong brand'],
    weaknesses: ['Legacy systems'],
    opportunities: ['Cloud migration'],
    threats: ['Competitors'],
  },
  engagement_strategy: 'Land and expand via IT leadership.',
  value_propositions: ['Cut infrastructure costs by 30%'],
  action_plan: [],
  success_metrics: ['50% workloads in cloud by Q4'],
  milestones: [],
  timeline: {},
  html_content: '',
  pdf_path: '',
  status: 'completed',
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
};

describe('AccountPlanSection', () => {
  it('renders section heading', () => {
    render(<AccountPlanSection accountPlan={null} generating={false} onGenerate={() => {}} />);
    expect(screen.getByText('Account Plan')).toBeTruthy();
  });

  it('shows "Build Account Plan" button when no plan exists', () => {
    render(<AccountPlanSection accountPlan={null} generating={false} onGenerate={() => {}} />);
    expect(screen.getByRole('button', { name: 'Build Account Plan' })).toBeTruthy();
  });

  it('shows "Regenerate" button when plan exists', () => {
    render(<AccountPlanSection accountPlan={accountPlan} generating={false} onGenerate={() => {}} />);
    expect(screen.getByRole('button', { name: 'Regenerate' })).toBeTruthy();
  });

  it('shows "Generating..." when generating is true', () => {
    render(<AccountPlanSection accountPlan={null} generating={true} onGenerate={() => {}} />);
    expect(screen.getByText('Generating...')).toBeTruthy();
  });

  it('calls onGenerate when generate button is clicked', () => {
    const onGenerate = vi.fn();
    render(<AccountPlanSection accountPlan={null} generating={false} onGenerate={onGenerate} />);
    fireEvent.click(screen.getByRole('button', { name: 'Build Account Plan' }));
    expect(onGenerate).toHaveBeenCalled();
  });

  it('shows plan title when plan exists', () => {
    render(<AccountPlanSection accountPlan={accountPlan} generating={false} onGenerate={() => {}} />);
    expect(screen.getByText('Acme Strategic Account Plan')).toBeTruthy();
  });

  it('shows "View Plan" button when plan exists', () => {
    render(<AccountPlanSection accountPlan={accountPlan} generating={false} onGenerate={() => {}} />);
    expect(screen.getByRole('button', { name: 'View Plan' })).toBeTruthy();
  });

  it('shows empty state message when no plan and not generating', () => {
    render(<AccountPlanSection accountPlan={null} generating={false} onGenerate={() => {}} />);
    expect(screen.getByText(/No account plan generated yet/)).toBeTruthy();
  });

  it('opens drawer when View Plan button is clicked', () => {
    render(<AccountPlanSection accountPlan={accountPlan} generating={false} onGenerate={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: 'View Plan' }));
    expect(screen.getByLabelText('Close account plan')).toBeTruthy();
  });

  it('shows strategic objectives count in plan preview', () => {
    render(<AccountPlanSection accountPlan={accountPlan} generating={false} onGenerate={() => {}} />);
    expect(screen.getByText(/2 strategic objectives/)).toBeTruthy();
  });
});
