/**
 * Tests for app/memory/page.tsx
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import MemoryPage from '@/app/memory/page';
import { api } from '@/lib/api';

vi.mock('@/lib/api', () => ({
  api: {
    listProfiles: vi.fn(),
    listPlays: vi.fn(),
    listEntries: vi.fn(),
  },
}));

const mockProfiles = [
  {
    id: 'p1',
    client_name: 'Acme Corp',
    industry: 'retail',
    company_size: 'Enterprise',
    region: 'North America',
    summary: 'Leading retail company pursuing cloud migration.',
    key_contacts: [],
    vector_id: '',
    created_at: '2025-01-01T00:00:00Z',
    updated_at: '2025-01-10T00:00:00Z',
  },
];

const mockPlays = [
  {
    id: 'sp1',
    title: 'Cloud Migration Pitch',
    play_type: 'pitch' as const,
    content: 'Reduce infrastructure costs by moving to the cloud.',
    context: 'Use when customer mentions legacy systems.',
    industry: 'retail',
    vertical: 'enterprise',
    usage_count: 5,
    success_rate: 0.8,
    created_at: '2025-01-01T00:00:00Z',
    updated_at: '2025-01-10T00:00:00Z',
  },
];

const mockEntries = [
  {
    id: 'e1',
    entry_type: 'research_insight' as const,
    title: 'Acme Cloud Strategy',
    content: 'Acme is pursuing cloud-first strategy to reduce TCO.',
    client_name: 'Acme Corp',
    industry: 'retail',
    tags: ['cloud', 'strategy'],
    source_type: 'research',
    source_id: 'job-123',
    created_at: '2025-01-05T00:00:00Z',
    updated_at: '2025-01-05T00:00:00Z',
  },
];

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(api.listProfiles).mockResolvedValue(mockProfiles);
  vi.mocked(api.listPlays).mockResolvedValue(mockPlays);
  vi.mocked(api.listEntries).mockResolvedValue(mockEntries);
});

describe('MemoryPage', () => {
  it('renders page heading', async () => {
    render(<MemoryPage />);
    expect(screen.getByText('Memory / Knowledge Base')).toBeTruthy();
  });

  it('shows tab buttons', () => {
    render(<MemoryPage />);
    expect(screen.getByText('Client Profiles')).toBeTruthy();
    expect(screen.getByText('Sales Play Library')).toBeTruthy();
    expect(screen.getByText('Memory Entries')).toBeTruthy();
  });

  it('renders client profiles by default', async () => {
    render(<MemoryPage />);
    await waitFor(() => {
      expect(screen.getByText('Acme Corp')).toBeTruthy();
    });
    expect(screen.getByText('retail')).toBeTruthy();
    expect(screen.getByText('Leading retail company pursuing cloud migration.')).toBeTruthy();
  });

  it('switches to sales plays tab', async () => {
    render(<MemoryPage />);
    await userEvent.click(screen.getByText('Sales Play Library'));
    await waitFor(() => {
      expect(screen.getByText('Cloud Migration Pitch')).toBeTruthy();
    });
    expect(screen.getAllByText('Pitch').length).toBeGreaterThan(0);
  });

  it('switches to memory entries tab', async () => {
    render(<MemoryPage />);
    await userEvent.click(screen.getByText('Memory Entries'));
    await waitFor(() => {
      expect(screen.getByText('Acme Cloud Strategy')).toBeTruthy();
    });
    expect(screen.getByText('Acme is pursuing cloud-first strategy to reduce TCO.')).toBeTruthy();
  });

  it('shows entry tags', async () => {
    render(<MemoryPage />);
    await userEvent.click(screen.getByText('Memory Entries'));
    await waitFor(() => {
      expect(screen.getByText('cloud')).toBeTruthy();
    });
    expect(screen.getByText('strategy')).toBeTruthy();
  });

  it('shows empty state when no profiles', async () => {
    vi.mocked(api.listProfiles).mockResolvedValue([]);
    render(<MemoryPage />);
    await waitFor(() => {
      expect(screen.getByText(/No client profiles captured yet/)).toBeTruthy();
    });
  });

  it('shows empty state when no plays', async () => {
    vi.mocked(api.listPlays).mockResolvedValue([]);
    render(<MemoryPage />);
    await userEvent.click(screen.getByText('Sales Play Library'));
    await waitFor(() => {
      expect(screen.getByText(/No sales plays stored yet/)).toBeTruthy();
    });
  });

  it('filters profiles by search', async () => {
    vi.mocked(api.listProfiles).mockResolvedValue([
      ...mockProfiles,
      {
        id: 'p2',
        client_name: 'TJX Companies',
        industry: 'retail',
        company_size: 'Large',
        region: 'US',
        summary: 'Off-price retailer.',
        key_contacts: [],
        vector_id: '',
        created_at: '2025-01-01T00:00:00Z',
        updated_at: '2025-01-01T00:00:00Z',
      },
    ]);
    render(<MemoryPage />);
    await waitFor(() => screen.getByText('Acme Corp'));
    await userEvent.type(screen.getByPlaceholderText('Search by company or industry...'), 'TJX');
    expect(screen.getByText('TJX Companies')).toBeTruthy();
    expect(screen.queryByText('Acme Corp')).toBeNull();
  });
});
