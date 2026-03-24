import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import Navigation from '@/components/Navigation';

vi.mock('@/lib/api', () => ({ api: {} }));

const mockUsePathname = vi.fn();
vi.mock('next/navigation', () => ({
  usePathname: () => mockUsePathname(),
}));

describe('Navigation', () => {
  it('renders the app title', () => {
    mockUsePathname.mockReturnValue('/');
    render(<Navigation />);
    expect(screen.getByText('Deep Prospecting Engine')).toBeTruthy();
  });

  it('renders all nav items', () => {
    mockUsePathname.mockReturnValue('/');
    render(<Navigation />);
    expect(screen.getByRole('link', { name: 'Quick Research' })).toBeTruthy();
    expect(screen.getByRole('link', { name: 'Research History' })).toBeTruthy();
    expect(screen.getByRole('link', { name: 'Projects' })).toBeTruthy();
    expect(screen.getByRole('link', { name: 'Assets' })).toBeTruthy();
  });

  it('highlights the active route', () => {
    mockUsePathname.mockReturnValue('/projects');
    render(<Navigation />);
    const projectsLink = screen.getByRole('link', { name: 'Projects' });
    expect(projectsLink.className).toContain('bg-blue-50');
  });

  it('does not highlight inactive routes', () => {
    mockUsePathname.mockReturnValue('/projects');
    render(<Navigation />);
    const researchLink = screen.getByRole('link', { name: 'Quick Research' });
    expect(researchLink.className).not.toContain('bg-blue-50');
  });

  it('links to correct hrefs', () => {
    mockUsePathname.mockReturnValue('/');
    render(<Navigation />);
    expect(screen.getByRole('link', { name: 'Research History' }).getAttribute('href')).toBe('/research');
    expect(screen.getByRole('link', { name: 'Projects' }).getAttribute('href')).toBe('/projects');
    expect(screen.getByRole('link', { name: 'Assets' }).getAttribute('href')).toBe('/assets');
  });
});
