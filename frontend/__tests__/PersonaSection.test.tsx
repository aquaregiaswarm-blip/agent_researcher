import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import PersonaSection from '@/components/research-results/generate/PersonaSection';
import { Persona } from '@/types';

vi.mock('@/lib/api', () => ({ api: {} }));

const persona: Persona = {
  id: 'p1',
  research_job: 'j1',
  name: 'Alex Chen',
  title: 'Chief Information Officer',
  department: 'IT',
  seniority_level: 'C-Suite',
  background: 'Former Google engineer with 15 years in enterprise IT.',
  goals: ['Reduce IT costs', 'Enable cloud migration'],
  challenges: ['Legacy systems', 'Budget constraints'],
  motivations: ['Efficiency', 'Innovation'],
  decision_criteria: ['Total cost of ownership', 'Vendor support'],
  preferred_communication: ['Email', 'Briefings'],
  objections: ['Too expensive', 'Complex integration'],
  content_preferences: ['Case studies', 'ROI calculators'],
  key_messages: ['We reduce your TCO by 30%'],
  created_at: '2026-01-01T00:00:00Z',
};

describe('PersonaSection', () => {
  it('renders section heading', () => {
    render(<PersonaSection personas={[]} generating={false} onGenerate={() => {}} />);
    expect(screen.getByText('Buyer Personas')).toBeTruthy();
  });

  it('shows "Build Personas" button when no personas exist', () => {
    render(<PersonaSection personas={[]} generating={false} onGenerate={() => {}} />);
    expect(screen.getByRole('button', { name: 'Build Personas' })).toBeTruthy();
  });

  it('shows "Regenerate" button when personas exist', () => {
    render(<PersonaSection personas={[persona]} generating={false} onGenerate={() => {}} />);
    expect(screen.getByRole('button', { name: 'Regenerate' })).toBeTruthy();
  });

  it('shows "Generating..." when generating is true', () => {
    render(<PersonaSection personas={[]} generating={true} onGenerate={() => {}} />);
    expect(screen.getByText('Generating...')).toBeTruthy();
  });

  it('disables button while generating', () => {
    render(<PersonaSection personas={[]} generating={true} onGenerate={() => {}} />);
    const btn = screen.getByRole('button', { name: /Generating/ });
    expect((btn as HTMLButtonElement).disabled).toBe(true);
  });

  it('calls onGenerate when button is clicked', () => {
    const onGenerate = vi.fn();
    render(<PersonaSection personas={[]} generating={false} onGenerate={onGenerate} />);
    fireEvent.click(screen.getByRole('button', { name: 'Build Personas' }));
    expect(onGenerate).toHaveBeenCalled();
  });

  it('renders persona name and title when personas exist', () => {
    render(<PersonaSection personas={[persona]} generating={false} onGenerate={() => {}} />);
    expect(screen.getByText('Alex Chen')).toBeTruthy();
    expect(screen.getByText(/Chief Information Officer/)).toBeTruthy();
  });

  it('renders persona background', () => {
    render(<PersonaSection personas={[persona]} generating={false} onGenerate={() => {}} />);
    expect(screen.getByText('Former Google engineer with 15 years in enterprise IT.')).toBeTruthy();
  });

  it('renders goals and challenges', () => {
    render(<PersonaSection personas={[persona]} generating={false} onGenerate={() => {}} />);
    expect(screen.getByText('Reduce IT costs')).toBeTruthy();
    expect(screen.getByText('Legacy systems')).toBeTruthy();
  });

  it('shows empty state when no personas and not generating', () => {
    render(<PersonaSection personas={[]} generating={false} onGenerate={() => {}} />);
    expect(screen.getByText(/No personas generated yet/)).toBeTruthy();
  });

  it('renders multiple persona cards', () => {
    const persona2 = { ...persona, id: 'p2', name: 'Sam Lee', title: 'CTO' };
    render(<PersonaSection personas={[persona, persona2]} generating={false} onGenerate={() => {}} />);
    expect(screen.getByText('Alex Chen')).toBeTruthy();
    expect(screen.getByText('Sam Lee')).toBeTruthy();
  });
});
