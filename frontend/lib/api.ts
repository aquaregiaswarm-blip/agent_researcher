import {
  ResearchJob,
  ResearchFormData,
  PromptTemplate,
  ProjectListItem,
  Project,
  ProjectCreateData,
  Iteration,
  IterationCreateData,
  WorkProduct,
  Annotation,
  TimelineData,
  IterationComparison,
  UseCase,
  FeasibilityAssessment,
  RefinedPlay,
  Persona,
  OnePager,
  AccountPlan,
  ClientProfile,
  SalesPlay,
  MemoryEntry,
  Citation,
} from '@/types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.error || error.detail || `Request failed: ${response.status}`);
    }

    return response.json();
  }

  // Research endpoints
  async listResearch(): Promise<ResearchJob[]> {
    return this.request<ResearchJob[]>('/api/research/jobs/');
  }

  async createResearch(data: ResearchFormData): Promise<ResearchJob> {
    const job = await this.request<ResearchJob>('/api/research/', {
      method: 'POST',
      body: JSON.stringify(data),
    });

    // Fire off the execute call (runs synchronously on backend, ~1-5 min)
    // Don't await — the frontend will poll for status
    this.executeResearch(job.id).catch(console.error);

    return job;
  }

  async executeResearch(id: string): Promise<ResearchJob> {
    return this.request<ResearchJob>(`/api/research/${id}/execute/`, {
      method: 'POST',
    });
  }

  async getResearch(id: string): Promise<ResearchJob> {
    return this.request<ResearchJob>(`/api/research/${id}/`);
  }

  async downloadResearchPdf(id: string): Promise<void> {
    const url = `${this.baseUrl}/api/research/${id}/export/pdf/`;
    const response = await fetch(url);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.error || `Failed to download PDF: ${response.status}`);
    }

    // Get filename from Content-Disposition header or use default
    const contentDisposition = response.headers.get('Content-Disposition');
    let filename = 'research_report.pdf';
    if (contentDisposition) {
      const match = contentDisposition.match(/filename="?([^"]+)"?/);
      if (match) {
        filename = match[1];
      }
    }

    // Convert response to blob and trigger download
    const blob = await response.blob();
    const downloadUrl = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(downloadUrl);
  }

  pollResearch(
    id: string,
    onUpdate: (job: ResearchJob) => void,
    intervalMs: number = 2000
  ): { promise: Promise<ResearchJob>; cancel: () => void } {
    let cancelled = false;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    const MAX_POLLS = 150; // 5 minutes at 2s intervals
    let pollCount = 0;

    const promise = new Promise<ResearchJob>((resolve, reject) => {
      const poll = async () => {
        if (cancelled) return;
        pollCount++;
        if (pollCount > MAX_POLLS) {
          reject(new Error('Research polling timed out after 5 minutes'));
          return;
        }
        try {
          const job = await this.getResearch(id);
          if (cancelled) return;
          onUpdate(job);

          if (job.status === 'completed' || job.status === 'failed') {
            resolve(job);
          } else {
            const delay = pollCount > 10 ? Math.min(intervalMs * 2, 8000) : intervalMs;
            timeoutId = setTimeout(poll, delay);
          }
        } catch (error) {
          if (!cancelled) reject(error);
        }
      };
      poll();
    });

    const cancel = () => {
      cancelled = true;
      if (timeoutId !== null) clearTimeout(timeoutId);
    };

    return { promise, cancel };
  }

  // Prompt endpoints
  async getDefaultPrompt(): Promise<PromptTemplate> {
    return this.request<PromptTemplate>('/api/prompts/default/');
  }

  async updateDefaultPrompt(content: string): Promise<PromptTemplate> {
    return this.request<PromptTemplate>('/api/prompts/default/', {
      method: 'PUT',
      body: JSON.stringify({ content }),
    });
  }

  // Project endpoints
  async listProjects(): Promise<ProjectListItem[]> {
    return this.request<ProjectListItem[]>('/api/projects/');
  }

  async getProject(id: string): Promise<Project> {
    return this.request<Project>(`/api/projects/${id}/`);
  }

  async createProject(data: ProjectCreateData): Promise<Project> {
    return this.request<Project>('/api/projects/', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateProject(id: string, data: Partial<ProjectCreateData>): Promise<Project> {
    return this.request<Project>(`/api/projects/${id}/`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    });
  }

  async deleteProject(id: string): Promise<void> {
    await this.request(`/api/projects/${id}/`, {
      method: 'DELETE',
    });
  }

  // Iteration endpoints
  async listIterations(projectId: string): Promise<Iteration[]> {
    return this.request<Iteration[]>(`/api/projects/${projectId}/iterations/`);
  }

  async getIteration(projectId: string, sequence: number): Promise<Iteration> {
    return this.request<Iteration>(`/api/projects/${projectId}/iterations/${sequence}/`);
  }

  async createIteration(projectId: string, data: IterationCreateData): Promise<Iteration> {
    return this.request<Iteration>(`/api/projects/${projectId}/iterations/`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async startIteration(projectId: string, sequence: number): Promise<{ iteration_id: string; research_job_id: string; status: string }> {
    return this.request(`/api/projects/${projectId}/iterations/${sequence}/start/`, {
      method: 'POST',
    });
  }

  pollIteration(
    projectId: string,
    sequence: number,
    onUpdate: (iteration: Iteration) => void,
    intervalMs: number = 2000
  ): { promise: Promise<Iteration>; cancel: () => void } {
    let cancelled = false;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    const MAX_POLLS = 150;
    let pollCount = 0;

    const promise = new Promise<Iteration>((resolve, reject) => {
      const poll = async () => {
        if (cancelled) return;
        pollCount++;
        if (pollCount > MAX_POLLS) {
          reject(new Error('Iteration polling timed out after 5 minutes'));
          return;
        }
        try {
          const iteration = await this.getIteration(projectId, sequence);
          if (cancelled) return;
          onUpdate(iteration);

          if (iteration.status === 'completed' || iteration.status === 'failed') {
            resolve(iteration);
          } else {
            const delay = pollCount > 10 ? Math.min(intervalMs * 2, 8000) : intervalMs;
            timeoutId = setTimeout(poll, delay);
          }
        } catch (error) {
          if (!cancelled) reject(error);
        }
      };
      poll();
    });

    const cancel = () => {
      cancelled = true;
      if (timeoutId !== null) clearTimeout(timeoutId);
    };

    return { promise, cancel };
  }

  // Timeline and comparison
  async getTimeline(projectId: string): Promise<TimelineData> {
    return this.request<TimelineData>(`/api/projects/${projectId}/timeline/`);
  }

  async compareIterations(projectId: string, seqA: number, seqB: number): Promise<IterationComparison> {
    return this.request<IterationComparison>(`/api/projects/${projectId}/compare/?a=${seqA}&b=${seqB}`);
  }

  // Work product endpoints
  async listWorkProducts(projectId: string): Promise<WorkProduct[]> {
    return this.request<WorkProduct[]>(`/api/projects/${projectId}/work-products/`);
  }

  async createWorkProduct(projectId: string, data: {
    content_type: string;
    object_id: string;
    category: string;
    source_iteration_id?: string;
    notes?: string;
  }): Promise<WorkProduct> {
    return this.request<WorkProduct>(`/api/projects/${projectId}/work-products/`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateWorkProduct(projectId: string, id: string, data: Partial<WorkProduct>): Promise<WorkProduct> {
    return this.request<WorkProduct>(`/api/projects/${projectId}/work-products/${id}/`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    });
  }

  async deleteWorkProduct(projectId: string, id: string): Promise<void> {
    await this.request(`/api/projects/${projectId}/work-products/${id}/`, {
      method: 'DELETE',
    });
  }

  // Annotation endpoints
  async listAnnotations(projectId: string): Promise<Annotation[]> {
    return this.request<Annotation[]>(`/api/projects/${projectId}/annotations/`);
  }

  async createAnnotation(projectId: string, data: {
    content_type: string;
    object_id: string;
    text: string;
  }): Promise<Annotation> {
    return this.request<Annotation>(`/api/projects/${projectId}/annotations/`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateAnnotation(projectId: string, id: string, data: { text: string }): Promise<Annotation> {
    return this.request<Annotation>(`/api/projects/${projectId}/annotations/${id}/`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    });
  }

  async deleteAnnotation(projectId: string, id: string): Promise<void> {
    await this.request(`/api/projects/${projectId}/annotations/${id}/`, {
      method: 'DELETE',
    });
  }

  // Ideation endpoints (AGE-18, AGE-19, AGE-20)

  async generateUseCases(researchJobId: string): Promise<UseCase[]> {
    return this.request<UseCase[]>('/api/ideation/use-cases/generate/', {
      method: 'POST',
      body: JSON.stringify({ research_job_id: researchJobId }),
    });
  }

  async listUseCases(researchJobId: string): Promise<UseCase[]> {
    return this.request<UseCase[]>(`/api/ideation/use-cases/?research_job=${researchJobId}`);
  }

  async getUseCase(id: string): Promise<UseCase> {
    return this.request<UseCase>(`/api/ideation/use-cases/${id}/`);
  }

  async assessFeasibility(useCaseId: string): Promise<FeasibilityAssessment> {
    return this.request<FeasibilityAssessment>(`/api/ideation/use-cases/${useCaseId}/assess/`, {
      method: 'POST',
    });
  }

  async refinePlay(useCaseId: string): Promise<RefinedPlay> {
    return this.request<RefinedPlay>(`/api/ideation/use-cases/${useCaseId}/refine/`, {
      method: 'POST',
    });
  }

  async getPlay(id: string): Promise<RefinedPlay> {
    return this.request<RefinedPlay>(`/api/ideation/plays/${id}/`);
  }

  // Asset endpoints (AGE-21, AGE-22, AGE-23)

  async generatePersonas(researchJobId: string): Promise<Persona[]> {
    return this.request<Persona[]>('/api/assets/personas/generate/', {
      method: 'POST',
      body: JSON.stringify({ research_job_id: researchJobId }),
    });
  }

  async listPersonas(researchJobId: string): Promise<Persona[]> {
    return this.request<Persona[]>(`/api/assets/personas/?research_job=${researchJobId}`);
  }

  async getPersona(id: string): Promise<Persona> {
    return this.request<Persona>(`/api/assets/personas/${id}/`);
  }

  async generateOnePager(researchJobId: string, useCaseId?: string): Promise<OnePager> {
    const body: Record<string, string> = { research_job_id: researchJobId };
    if (useCaseId) body.use_case_id = useCaseId;
    return this.request<OnePager>('/api/assets/one-pagers/generate/', {
      method: 'POST',
      body: JSON.stringify(body),
    });
  }

  async listOnePagers(researchJobId: string): Promise<OnePager[]> {
    return this.request<OnePager[]>(`/api/assets/one-pagers/?research_job=${researchJobId}`);
  }

  async getOnePager(id: string): Promise<OnePager> {
    return this.request<OnePager>(`/api/assets/one-pagers/${id}/`);
  }

  async getOnePagerHtml(id: string): Promise<string> {
    const url = `${this.baseUrl}/api/assets/one-pagers/${id}/html/`;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to fetch one-pager HTML: ${response.status}`);
    return response.text();
  }

  async generateAccountPlan(researchJobId: string): Promise<AccountPlan> {
    return this.request<AccountPlan>('/api/assets/account-plans/generate/', {
      method: 'POST',
      body: JSON.stringify({ research_job_id: researchJobId }),
    });
  }

  async listAccountPlans(researchJobId: string): Promise<AccountPlan[]> {
    return this.request<AccountPlan[]>(`/api/assets/account-plans/?research_job=${researchJobId}`);
  }

  async getAccountPlan(id: string): Promise<AccountPlan> {
    return this.request<AccountPlan>(`/api/assets/account-plans/${id}/`);
  }

  async getAccountPlanHtml(id: string): Promise<string> {
    const url = `${this.baseUrl}/api/assets/account-plans/${id}/html/`;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to fetch account plan HTML: ${response.status}`);
    return response.text();
  }

  // Memory endpoints

  async captureToMemory(jobId: string): Promise<{ client_profile_created: boolean; memory_entries_created: number }> {
    return this.request(`/api/memory/capture/${jobId}/`, {
      method: 'POST',
    });
  }

  async listProfiles(): Promise<ClientProfile[]> {
    return this.request<ClientProfile[]>('/api/memory/profiles/');
  }

  async getProfile(id: string): Promise<ClientProfile> {
    return this.request<ClientProfile>(`/api/memory/profiles/${id}/`);
  }

  async listPlays(): Promise<SalesPlay[]> {
    return this.request<SalesPlay[]>('/api/memory/plays/');
  }

  async getMemoryPlay(id: string): Promise<SalesPlay> {
    return this.request<SalesPlay>(`/api/memory/plays/${id}/`);
  }

  async listEntries(clientName?: string): Promise<MemoryEntry[]> {
    const qs = clientName ? `?client_name=${encodeURIComponent(clientName)}` : '';
    return this.request<MemoryEntry[]>(`/api/memory/entries/${qs}`);
  }

  async getEntry(id: string): Promise<MemoryEntry> {
    return this.request<MemoryEntry>(`/api/memory/entries/${id}/`);
  }

  async queryContext(clientName: string): Promise<{ client_profiles: ClientProfile[]; sales_plays: SalesPlay[]; memory_entries: MemoryEntry[]; relevance_summary: string }> {
    return this.request('/api/memory/context/', {
      method: 'POST',
      body: JSON.stringify({ client_name: clientName }),
    });
  }

  // Citation endpoints (AGE-24)

  async listCitations(researchJobId: string): Promise<Citation[]> {
    return this.request<Citation[]>(`/api/assets/citations/?research_job=${researchJobId}`);
  }

  async getCitation(id: string): Promise<Citation> {
    return this.request<Citation>(`/api/assets/citations/${id}/`);
  }

  async verifyCitation(id: string): Promise<Citation> {
    return this.request<Citation>(`/api/assets/citations/${id}/`, {
      method: 'PATCH',
      body: JSON.stringify({ verified: true }),
    });
  }
}

export const api = new ApiClient(API_URL);
