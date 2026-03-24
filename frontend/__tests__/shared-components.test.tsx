import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import StatCard from '@/components/research-results/shared/StatCard';
import DetailRow from '@/components/research-results/shared/DetailRow';
import Section from '@/components/research-results/shared/Section';
import GapList from '@/components/research-results/shared/GapList';

vi.mock('@/lib/api', () => ({ api: {} }));

// ---------------------------------------------------------------------------
// StatCard
// ---------------------------------------------------------------------------
describe('StatCard', () => {
  it('renders label and value', () => {
    render(<StatCard label="Founded" value="2001" />);
    expect(screen.getByText('Founded')).toBeTruthy();
    expect(screen.getByText('2001')).toBeTruthy();
  });

  it('renders without optional className', () => {
    const { container } = render(<StatCard label="Revenue" value="$4.5B" />);
    expect(container).toBeTruthy();
  });

  it('applies extra className to value element', () => {
    render(<StatCard label="Status" value="maturing" className="text-green-600" />);
    const value = screen.getByText('maturing');
    expect(value.className).toContain('text-green-600');
  });
});

// ---------------------------------------------------------------------------
// DetailRow
// ---------------------------------------------------------------------------
describe('DetailRow', () => {
  it('renders label and string value', () => {
    render(<DetailRow label="HQ" value="Seattle, WA" />);
    expect(screen.getByText('HQ')).toBeTruthy();
    expect(screen.getByText('Seattle, WA')).toBeTruthy();
  });

  it('renders ReactNode value', () => {
    render(<DetailRow label="Website" value={<a href="https://example.com">example.com</a>} />);
    expect(screen.getByText('Website')).toBeTruthy();
    expect(screen.getByRole('link', { name: 'example.com' })).toBeTruthy();
  });

  it('renders separator line between items', () => {
    const { container } = render(<DetailRow label="Key" value="Val" />);
    const div = container.firstChild as HTMLElement;
    expect(div.className).toContain('border-b');
  });
});

// ---------------------------------------------------------------------------
// Section
// ---------------------------------------------------------------------------
describe('Section', () => {
  it('renders title', () => {
    render(<Section title="Company Overview"><p>Content here</p></Section>);
    expect(screen.getByText('Company Overview')).toBeTruthy();
  });

  it('renders children', () => {
    render(<Section title="Any"><span>Child text</span></Section>);
    expect(screen.getByText('Child text')).toBeTruthy();
  });
});

// ---------------------------------------------------------------------------
// GapList
// ---------------------------------------------------------------------------
describe('GapList', () => {
  it('renders title and all items', () => {
    render(<GapList title="Technology Gaps" items={['No SIEM', 'Legacy ERP']} color="red" />);
    expect(screen.getByText('Technology Gaps')).toBeTruthy();
    expect(screen.getByText('No SIEM')).toBeTruthy();
    expect(screen.getByText('Legacy ERP')).toBeTruthy();
  });

  it('renders orange color variant without crashing', () => {
    const { container } = render(
      <GapList title="Capability Gaps" items={['No DevOps']} color="orange" />
    );
    expect(container).toBeTruthy();
  });

  it('renders purple color variant without crashing', () => {
    const { container } = render(
      <GapList title="Process Gaps" items={['Manual patching']} color="purple" />
    );
    expect(container).toBeTruthy();
  });

  it('renders empty list without crashing', () => {
    const { container } = render(<GapList title="No Gaps" items={[]} color="red" />);
    expect(container).toBeTruthy();
    expect(screen.getByText('No Gaps')).toBeTruthy();
  });
});
