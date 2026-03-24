import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import UseCaseSection from '@/components/research-results/generate/UseCaseSection';
import { UseCase } from '@/types';

vi.mock('@/lib/api', () => ({ api: {} }));
vi.mock('@/lib/toast', () => ({
  useToast: () => ({ addToast: vi.fn() }),
}));

const useCase: UseCase = {
  id: 'uc1',
  research_job: 'j1',
  title: 'AI Supply Chain Optimisation',
  description: 'Use ML to predict demand and reduce inventory waste.',
  business_problem: 'Excess inventory costs $5M/year.',
  proposed_solution: 'Deploy ML forecasting pipeline on Azure.',
  expected_benefits: ['30% reduction in inventory costs'],
  estimated_roi: '$1.5M over 3 years',
  time_to_value: '6 months',
  technologies: ['Azure ML', 'Power BI'],
  data_requirements: ['ERP transaction data'],
  integration_points: ['SAP ERP', 'Power BI'],
  priority: 'high',
  impact_score: 0.85,
  feasibility_score: 0.75,
  status: 'draft',
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
};

describe('UseCaseSection', () => {
  it('renders section heading', () => {
    render(
      <UseCaseSection researchJobId="j1" useCases={[]} generating={false} onGenerate={() => {}} />
    );
    expect(screen.getByText('Use Cases')).toBeTruthy();
  });

  it('shows "Generate Use Cases" button when no use cases exist', () => {
    render(
      <UseCaseSection researchJobId="j1" useCases={[]} generating={false} onGenerate={() => {}} />
    );
    expect(screen.getByRole('button', { name: 'Generate Use Cases' })).toBeTruthy();
  });

  it('shows "Regenerate" button when use cases exist', () => {
    render(
      <UseCaseSection researchJobId="j1" useCases={[useCase]} generating={false} onGenerate={() => {}} />
    );
    expect(screen.getByRole('button', { name: 'Regenerate' })).toBeTruthy();
  });

  it('shows "Generating..." and disables button while generating', () => {
    render(
      <UseCaseSection researchJobId="j1" useCases={[]} generating={true} onGenerate={() => {}} />
    );
    const btn = screen.getByRole('button', { name: /Generating/ });
    expect((btn as HTMLButtonElement).disabled).toBe(true);
  });

  it('calls onGenerate when button is clicked', () => {
    const onGenerate = vi.fn();
    render(
      <UseCaseSection researchJobId="j1" useCases={[]} generating={false} onGenerate={onGenerate} />
    );
    fireEvent.click(screen.getByRole('button', { name: 'Generate Use Cases' }));
    expect(onGenerate).toHaveBeenCalled();
  });

  it('renders use case title and business problem', () => {
    render(
      <UseCaseSection researchJobId="j1" useCases={[useCase]} generating={false} onGenerate={() => {}} />
    );
    expect(screen.getByText('AI Supply Chain Optimisation')).toBeTruthy();
    expect(screen.getByText('Excess inventory costs $5M/year.')).toBeTruthy();
  });

  it('renders priority badge', () => {
    render(
      <UseCaseSection researchJobId="j1" useCases={[useCase]} generating={false} onGenerate={() => {}} />
    );
    expect(screen.getByText('high')).toBeTruthy();
  });

  it('renders estimated ROI', () => {
    render(
      <UseCaseSection researchJobId="j1" useCases={[useCase]} generating={false} onGenerate={() => {}} />
    );
    expect(screen.getByText(/ROI: \$1\.5M over 3 years/)).toBeTruthy();
  });

  it('shows empty state when no use cases and not generating', () => {
    render(
      <UseCaseSection researchJobId="j1" useCases={[]} generating={false} onGenerate={() => {}} />
    );
    expect(screen.getByText(/No use cases generated yet/)).toBeTruthy();
  });

  it('renders multiple use case cards', () => {
    const uc2 = { ...useCase, id: 'uc2', title: 'Predictive Maintenance', priority: 'medium' as const };
    render(
      <UseCaseSection researchJobId="j1" useCases={[useCase, uc2]} generating={false} onGenerate={() => {}} />
    );
    expect(screen.getByText('AI Supply Chain Optimisation')).toBeTruthy();
    expect(screen.getByText('Predictive Maintenance')).toBeTruthy();
  });
});
