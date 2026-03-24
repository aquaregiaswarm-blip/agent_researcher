import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import OnePagerSection from '@/components/research-results/generate/OnePagerSection';
import { OnePager } from '@/types';

vi.mock('@/lib/api', () => ({ api: {} }));
vi.mock('dompurify', () => ({
  default: { sanitize: (html: string) => html },
}));

const onePager: OnePager = {
  id: 'op1',
  research_job: 'j1',
  title: 'Acme Cloud Migration',
  headline: 'Modernise in 6 months',
  executive_summary: 'Acme needs cloud to scale.',
  challenge_section: 'Legacy costs $10M/year.',
  solution_section: 'Azure migration.',
  benefits_section: '30% cost reduction.',
  html_content: '<p>Full HTML content</p>',
  pdf_path: '',
  status: 'completed',
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
};

describe('OnePagerSection', () => {
  it('renders section heading', () => {
    render(<OnePagerSection onePager={null} generating={false} onGenerate={() => {}} />);
    expect(screen.getByText('One-Pager')).toBeTruthy();
  });

  it('shows "Create One-Pager" button when no one-pager exists', () => {
    render(<OnePagerSection onePager={null} generating={false} onGenerate={() => {}} />);
    expect(screen.getByRole('button', { name: 'Create One-Pager' })).toBeTruthy();
  });

  it('shows "Regenerate" button when one-pager exists', () => {
    render(<OnePagerSection onePager={onePager} generating={false} onGenerate={() => {}} />);
    expect(screen.getByRole('button', { name: 'Regenerate' })).toBeTruthy();
  });

  it('shows "Generating..." when generating is true', () => {
    render(<OnePagerSection onePager={null} generating={true} onGenerate={() => {}} />);
    expect(screen.getByText('Generating...')).toBeTruthy();
  });

  it('disables generate button while generating', () => {
    render(<OnePagerSection onePager={null} generating={true} onGenerate={() => {}} />);
    const btn = screen.getByRole('button', { name: /Generating/ });
    expect((btn as HTMLButtonElement).disabled).toBe(true);
  });

  it('calls onGenerate when button is clicked', () => {
    const onGenerate = vi.fn();
    render(<OnePagerSection onePager={null} generating={false} onGenerate={onGenerate} />);
    fireEvent.click(screen.getByRole('button', { name: 'Create One-Pager' }));
    expect(onGenerate).toHaveBeenCalled();
  });

  it('renders HTML content preview when one-pager has html_content', () => {
    render(<OnePagerSection onePager={onePager} generating={false} onGenerate={() => {}} />);
    expect(screen.getByText('Acme Cloud Migration')).toBeTruthy();
  });

  it('renders Print/PDF and Full Screen buttons when one-pager exists', () => {
    render(<OnePagerSection onePager={onePager} generating={false} onGenerate={() => {}} />);
    expect(screen.getByRole('button', { name: 'Print / PDF' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Full Screen' })).toBeTruthy();
  });

  it('shows empty state message when no one-pager and not generating', () => {
    render(<OnePagerSection onePager={null} generating={false} onGenerate={() => {}} />);
    expect(screen.getByText(/No one-pager generated yet/)).toBeTruthy();
  });

  it('opens full-screen modal when Full Screen is clicked', () => {
    render(<OnePagerSection onePager={onePager} generating={false} onGenerate={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: 'Full Screen' }));
    expect(screen.getByLabelText('Close')).toBeTruthy();
  });

  it('closes full-screen modal when Close is clicked', () => {
    render(<OnePagerSection onePager={onePager} generating={false} onGenerate={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: 'Full Screen' }));
    fireEvent.click(screen.getByLabelText('Close'));
    expect(screen.queryByLabelText('Close')).toBeNull();
  });
});
