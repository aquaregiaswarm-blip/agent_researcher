import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { api } from '@/lib/api';

describe('API Client', () => {
  const mockFetch = vi.fn();

  beforeEach(() => {
    global.fetch = mockFetch;
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('createResearch', () => {
    it('sends POST request with correct data', async () => {
      const mockResponse = {
        id: '123',
        client_name: 'Test Corp',
        status: 'pending',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await api.createResearch({
        client_name: 'Test Corp',
        sales_history: 'Previous sales',
        prompt: 'Research prompt',
      });

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/research/',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            client_name: 'Test Corp',
            sales_history: 'Previous sales',
            prompt: 'Research prompt',
          }),
        })
      );

      expect(result).toEqual(mockResponse);
    });

    it('throws error on failed request', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({ detail: 'Bad request' }),
      });

      await expect(
        api.createResearch({
          client_name: '',
          sales_history: '',
          prompt: '',
        })
      ).rejects.toThrow('Bad request');
    });
  });

  describe('getResearch', () => {
    it('sends GET request with correct ID', async () => {
      const mockResponse = {
        id: '123',
        client_name: 'Test Corp',
        status: 'completed',
        result: 'Research results',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await api.getResearch('123');

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/research/123/',
        expect.objectContaining({
          headers: {
            'Content-Type': 'application/json',
          },
        })
      );

      expect(result).toEqual(mockResponse);
    });
  });

  describe('getDefaultPrompt', () => {
    it('fetches default prompt', async () => {
      const mockResponse = {
        id: 1,
        name: 'default',
        content: 'Prompt content',
        is_default: true,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await api.getDefaultPrompt();

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/prompts/default/',
        expect.objectContaining({
          headers: {
            'Content-Type': 'application/json',
          },
        })
      );

      expect(result).toEqual(mockResponse);
    });
  });

  describe('updateDefaultPrompt', () => {
    it('sends PUT request with content', async () => {
      const mockResponse = {
        id: 1,
        name: 'default',
        content: 'Updated content',
        is_default: true,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await api.updateDefaultPrompt('Updated content');

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/prompts/default/',
        expect.objectContaining({
          method: 'PUT',
          body: JSON.stringify({ content: 'Updated content' }),
        })
      );

      expect(result).toEqual(mockResponse);
    });
  });

  describe('listAccountPlans', () => {
    it('sends GET request with research_job filter', async () => {
      const mockPlans = [{ id: 'ap-1', title: 'Plan A' }];

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockPlans),
      });

      const result = await api.listAccountPlans('job-123');

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/assets/account-plans/?research_job=job-123',
        expect.objectContaining({ headers: { 'Content-Type': 'application/json' } })
      );
      expect(result).toEqual(mockPlans);
    });

    it('throws error when request fails', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ detail: 'Server error' }),
      });

      await expect(api.listAccountPlans('job-123')).rejects.toThrow('Server error');
    });
  });

  describe('generateAccountPlan', () => {
    it('sends POST request with research_job_id', async () => {
      const mockPlan = { id: 'ap-1', title: 'Generated Plan' };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockPlan),
      });

      const result = await api.generateAccountPlan('job-456');

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/assets/account-plans/generate/',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ research_job_id: 'job-456' }),
        })
      );
      expect(result).toEqual(mockPlan);
    });
  });

  describe('listPersonas', () => {
    it('sends GET request with research_job filter', async () => {
      const mockPersonas = [{ id: 'p-1', name: 'Alice' }];

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockPersonas),
      });

      const result = await api.listPersonas('job-789');

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/assets/personas/?research_job=job-789',
        expect.objectContaining({ headers: { 'Content-Type': 'application/json' } })
      );
      expect(result).toEqual(mockPersonas);
    });
  });

  describe('listOnePagers', () => {
    it('sends GET request with research_job filter', async () => {
      const mockPagers = [{ id: 'op-1', title: 'One Pager' }];

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockPagers),
      });

      const result = await api.listOnePagers('job-abc');

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/assets/one-pagers/?research_job=job-abc',
        expect.objectContaining({ headers: { 'Content-Type': 'application/json' } })
      );
      expect(result).toEqual(mockPagers);
    });
  });

  describe('captureToMemory', () => {
    it('sends POST request to capture endpoint', async () => {
      const mockResponse = { client_profile_created: true, memory_entries_created: 3 };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await api.captureToMemory('job-mem-111');

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/memory/capture/job-mem-111/',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
        })
      );
      expect(result).toEqual(mockResponse);
    });
  });

  describe('listProfiles', () => {
    it('sends GET request to profiles endpoint', async () => {
      const mockProfiles = [{ id: 'p-1', client_name: 'Acme Corp' }];
      mockFetch.mockResolvedValueOnce({ ok: true, json: () => Promise.resolve(mockProfiles) });

      const result = await api.listProfiles();

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/memory/profiles/',
        expect.objectContaining({ headers: { 'Content-Type': 'application/json' } })
      );
      expect(result).toEqual(mockProfiles);
    });
  });

  describe('listPlays', () => {
    it('sends GET request to plays endpoint', async () => {
      const mockPlays = [{ id: 'sp-1', title: 'Cloud Pitch' }];
      mockFetch.mockResolvedValueOnce({ ok: true, json: () => Promise.resolve(mockPlays) });

      const result = await api.listPlays();

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/memory/plays/',
        expect.objectContaining({ headers: { 'Content-Type': 'application/json' } })
      );
      expect(result).toEqual(mockPlays);
    });
  });

  describe('listEntries', () => {
    it('sends GET request to entries endpoint without filter', async () => {
      const mockEntries = [{ id: 'e-1', title: 'Insight' }];
      mockFetch.mockResolvedValueOnce({ ok: true, json: () => Promise.resolve(mockEntries) });

      const result = await api.listEntries();

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/memory/entries/',
        expect.objectContaining({ headers: { 'Content-Type': 'application/json' } })
      );
      expect(result).toEqual(mockEntries);
    });

    it('includes client_name filter when provided', async () => {
      mockFetch.mockResolvedValueOnce({ ok: true, json: () => Promise.resolve([]) });

      await api.listEntries('Acme Corp');

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/memory/entries/?client_name=Acme%20Corp',
        expect.anything()
      );
    });
  });

  describe('queryContext', () => {
    it('sends POST request with client_name', async () => {
      const mockContext = { client_profiles: [], sales_plays: [], memory_entries: [], relevance_summary: 'None' };
      mockFetch.mockResolvedValueOnce({ ok: true, json: () => Promise.resolve(mockContext) });

      const result = await api.queryContext('Acme Corp');

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/memory/context/',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ client_name: 'Acme Corp' }),
        })
      );
      expect(result).toEqual(mockContext);
    });
  });

  describe('listCitations', () => {
    it('sends GET request with research_job filter', async () => {
      const mockCitations = [{ id: 'c-1', title: 'Acme News' }];
      mockFetch.mockResolvedValueOnce({ ok: true, json: () => Promise.resolve(mockCitations) });

      const result = await api.listCitations('job-cit-1');

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/assets/citations/?research_job=job-cit-1',
        expect.objectContaining({ headers: { 'Content-Type': 'application/json' } })
      );
      expect(result).toEqual(mockCitations);
    });
  });

  describe('verifyCitation', () => {
    it('sends PATCH with verified: true', async () => {
      const mockCitation = { id: 'c-1', verified: true };
      mockFetch.mockResolvedValueOnce({ ok: true, json: () => Promise.resolve(mockCitation) });

      const result = await api.verifyCitation('c-1');

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/assets/citations/c-1/',
        expect.objectContaining({
          method: 'PATCH',
          body: JSON.stringify({ verified: true }),
        })
      );
      expect(result).toEqual(mockCitation);
    });
  });
});
