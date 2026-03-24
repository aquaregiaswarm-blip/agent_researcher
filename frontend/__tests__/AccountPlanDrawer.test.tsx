import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import AccountPlanDrawer from '@/components/research-results/generate/AccountPlanDrawer';
import { AccountPlan } from '@/types';

vi.mock('@/lib/api', () => ({ api: {} }));
vi.mock('dompurify', () => ({
  default: { sanitize: (html: string) => html },
}));

const plan: AccountPlan = {
  id: 'ap1',
  research_job: 'j1',
  title: 'Acme Strategic Account Plan',
  executive_summary: 'Acme is a prime target for cloud expansion.',
  account_overview: 'Global distribution company.',
  strategic_objectives: ['Cloud-first by 2026'],
  key_stakeholders: [],
  opportunities: [],
  competitive_landscape: 'AWS and Azure compete.',
  swot_analysis: {
    strengths: ['Strong brand recognition'],
    weaknesses: ['Legacy ERP'],
    opportunities: ['Cloud migration budget'],
    threats: ['Incumbent vendor'],
  },
  engagement_strategy: 'Land and expand via CIO relationship.',
  value_propositions: ['Reduce TCO by 30%'],
  action_plan: [],
  success_metrics: ['50% workloads migrated by Q4'],
  milestones: [],
  timeline: {},
  html_content: '',
  pdf_path: '',
  status: 'completed',
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
};

describe('AccountPlanDrawer', () => {
  it('renders plan title', () => {
    render(<AccountPlanDrawer plan={plan} onClose={() => {}} />);
    expect(screen.getByText('Acme Strategic Account Plan')).toBeTruthy();
  });

  it('renders executive summary', () => {
    render(<AccountPlanDrawer plan={plan} onClose={() => {}} />);
    expect(screen.getByText('Executive Summary')).toBeTruthy();
    expect(screen.getByText('Acme is a prime target for cloud expansion.')).toBeTruthy();
  });

  it('renders strategic objectives', () => {
    render(<AccountPlanDrawer plan={plan} onClose={() => {}} />);
    expect(screen.getByText('Strategic Objectives')).toBeTruthy();
    expect(screen.getByText('Cloud-first by 2026')).toBeTruthy();
  });

  it('renders SWOT analysis sections', () => {
    render(<AccountPlanDrawer plan={plan} onClose={() => {}} />);
    expect(screen.getByText('SWOT Analysis')).toBeTruthy();
    expect(screen.getByText('Strengths')).toBeTruthy();
    expect(screen.getByText('Weaknesses')).toBeTruthy();
    expect(screen.getByText('Opportunities')).toBeTruthy();
    expect(screen.getByText('Threats')).toBeTruthy();
  });

  it('renders SWOT items', () => {
    render(<AccountPlanDrawer plan={plan} onClose={() => {}} />);
    expect(screen.getByText(/Strong brand recognition/)).toBeTruthy();
    expect(screen.getByText(/Legacy ERP/)).toBeTruthy();
  });

  it('renders engagement strategy', () => {
    render(<AccountPlanDrawer plan={plan} onClose={() => {}} />);
    expect(screen.getByText('Engagement Strategy')).toBeTruthy();
    expect(screen.getByText('Land and expand via CIO relationship.')).toBeTruthy();
  });

  it('renders value propositions', () => {
    render(<AccountPlanDrawer plan={plan} onClose={() => {}} />);
    expect(screen.getByText('Value Propositions')).toBeTruthy();
    expect(screen.getByText('Reduce TCO by 30%')).toBeTruthy();
  });

  it('renders success metrics', () => {
    render(<AccountPlanDrawer plan={plan} onClose={() => {}} />);
    expect(screen.getByText('Success Metrics')).toBeTruthy();
    expect(screen.getByText('50% workloads migrated by Q4')).toBeTruthy();
  });

  it('calls onClose when close button is clicked', () => {
    const onClose = vi.fn();
    render(<AccountPlanDrawer plan={plan} onClose={onClose} />);
    fireEvent.click(screen.getByLabelText('Close account plan'));
    expect(onClose).toHaveBeenCalled();
  });

  it('calls onClose when backdrop is clicked', () => {
    const onClose = vi.fn();
    render(<AccountPlanDrawer plan={plan} onClose={onClose} />);
    // Drawer uses createPortal — query from document.body
    const backdrop = document.body.querySelector('[aria-hidden="true"]');
    fireEvent.click(backdrop!);
    expect(onClose).toHaveBeenCalled();
  });

  it('calls onClose when Escape key is pressed', () => {
    const onClose = vi.fn();
    render(<AccountPlanDrawer plan={plan} onClose={onClose} />);
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(onClose).toHaveBeenCalled();
  });
});
