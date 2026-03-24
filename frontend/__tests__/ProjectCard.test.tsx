import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import ProjectCard from '@/components/projects/ProjectCard';
import { ProjectListItem } from '@/types';

vi.mock('next/navigation', () => ({ usePathname: () => '/' }));
vi.mock('@/lib/api', () => ({ api: {} }));

const baseProject: ProjectListItem = {
  id: 'p1',
  name: 'Acme Deep Dive',
  client_name: 'Acme Corp',
  context_mode: 'accumulate',
  iteration_count: 3,
  latest_iteration_status: 'completed',
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-03-15T00:00:00Z',
};

describe('ProjectCard', () => {
  it('renders project name and client name', () => {
    render(<ProjectCard project={baseProject} />);
    expect(screen.getByText('Acme Deep Dive')).toBeTruthy();
    expect(screen.getByText('Acme Corp')).toBeTruthy();
  });

  it('renders iteration count with correct plural form', () => {
    render(<ProjectCard project={baseProject} />);
    expect(screen.getByText(/3 iterations/)).toBeTruthy();
  });

  it('renders singular "iteration" for count of 1', () => {
    render(<ProjectCard project={{ ...baseProject, iteration_count: 1 }} />);
    // Text is split across SVG + text nodes, so query the container text
    expect(screen.queryByText(/iterations/)).toBeNull();
    const { container } = render(<ProjectCard project={{ ...baseProject, iteration_count: 1 }} />);
    expect(container.textContent).toContain('1 iteration');
    expect(container.textContent).not.toMatch(/1 iterations/);
  });

  it('renders status badge when latest_iteration_status is present', () => {
    render(<ProjectCard project={baseProject} />);
    expect(screen.getByText('completed')).toBeTruthy();
  });

  it('does not render status badge when latest_iteration_status is absent', () => {
    const project = { ...baseProject, latest_iteration_status: undefined };
    render(<ProjectCard project={project} />);
    expect(screen.queryByText('completed')).toBeNull();
  });

  it('renders "Builds context" label for accumulate mode', () => {
    render(<ProjectCard project={baseProject} />);
    expect(screen.getByText('Builds context')).toBeTruthy();
  });

  it('renders "Fresh starts" label for fresh mode', () => {
    render(<ProjectCard project={{ ...baseProject, context_mode: 'fresh' }} />);
    expect(screen.getByText('Fresh starts')).toBeTruthy();
  });

  it('renders formatted updated date', () => {
    render(<ProjectCard project={baseProject} />);
    // Date formatted as "Mar 15, 2026" (or similar locale variant)
    expect(screen.getByText(/Mar/)).toBeTruthy();
  });

  it('wraps in a link to the project page', () => {
    render(<ProjectCard project={baseProject} />);
    const link = screen.getByRole('link');
    expect(link.getAttribute('href')).toBe('/projects/p1');
  });
});
