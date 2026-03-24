'use client';

import { useState, useEffect, useCallback } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { Project, Iteration, IterationListItem, WorkProduct, Annotation, IterationComparison, ResearchJob, UseCase, Persona, OnePager, AccountPlan } from '@/types';
import { api } from '@/lib/api';
import {
  ProjectHeader,
  IterationTabs,
  WorkProductPanel,
  AnnotationPanel,
  IterationCompare,
} from '@/components/projects';
import ResearchResults from '@/components/ResearchResults';

export default function ProjectDashboardPage() {
  const params = useParams();
  const projectId = params.id as string;

  const [project, setProject] = useState<Project | null>(null);
  const [selectedIteration, setSelectedIteration] = useState<Iteration | null>(null);
  const [workProducts, setWorkProducts] = useState<WorkProduct[]>([]);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Comparison modal state
  const [showCompare, setShowCompare] = useState(false);
  const [comparison, setComparison] = useState<IterationComparison | null>(null);

  // Sidebar state
  const [sidebarTab, setSidebarTab] = useState<'saved' | 'notes' | 'generated'>('saved');

  // Generated assets state
  const [generatedAssets, setGeneratedAssets] = useState<{
    useCases: UseCase[];
    personas: Persona[];
    onePagers: OnePager[];
    accountPlans: AccountPlan[];
  }>({ useCases: [], personas: [], onePagers: [], accountPlans: [] });
  const [loadingGenerated, setLoadingGenerated] = useState(false);

  const loadIteration = useCallback(async (sequence: number) => {
    try {
      const iteration = await api.getIteration(projectId, sequence);
      setSelectedIteration(iteration);

      // If running, poll for updates
      if (iteration.status === 'running') {
        const poller = api.pollIteration(projectId, sequence, (updated) => {
          setSelectedIteration(updated);
        });
        poller.promise.catch(console.error);
        // Return cancel so callers can clean up (best-effort; useCallback doesn't support cleanup directly)
        return poller.cancel;
      }
      return undefined;
    } catch (err) {
      console.error('Failed to load iteration:', err);
    }
  }, [projectId]);

  const loadProject = useCallback(async () => {
    try {
      setIsLoading(true);
      const data = await api.getProject(projectId);
      setProject(data);

      // Select latest iteration by default
      if (data.iterations.length > 0) {
        const latestSeq = Math.max(...data.iterations.map((i) => i.sequence));
        loadIteration(latestSeq);
      }
    } catch (err) {
      setError('Failed to load project');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  }, [projectId, loadIteration]);

  const loadWorkProducts = useCallback(async () => {
    try {
      const data = await api.listWorkProducts(projectId);
      setWorkProducts(data);
    } catch (err) {
      console.error('Failed to load work products:', err);
    }
  }, [projectId]);

  const loadAnnotations = useCallback(async () => {
    try {
      const data = await api.listAnnotations(projectId);
      setAnnotations(data);
    } catch (err) {
      console.error('Failed to load annotations:', err);
    }
  }, [projectId]);

  const loadGeneratedAssets = useCallback(async (iterations: IterationListItem[]) => {
    const jobIds = iterations
      .map((i) => i.research_job_id)
      .filter((id): id is string => !!id);
    if (jobIds.length === 0) return;

    setLoadingGenerated(true);
    try {
      const [useCases, personas, onePagers, accountPlans] = await Promise.all([
        Promise.all(jobIds.map((id) => api.listUseCases(id))).then((r) => r.flat()),
        Promise.all(jobIds.map((id) => api.listPersonas(id))).then((r) => r.flat()),
        Promise.all(jobIds.map((id) => api.listOnePagers(id))).then((r) => r.flat()),
        Promise.all(jobIds.map((id) => api.listAccountPlans(id))).then((r) => r.flat()),
      ]);
      setGeneratedAssets({ useCases, personas, onePagers, accountPlans });
    } catch (err) {
      console.error('Failed to load generated assets:', err);
    } finally {
      setLoadingGenerated(false);
    }
  }, []);

  const handleStartIteration = async () => {
    if (!selectedIteration) return;

    try {
      await api.startIteration(projectId, selectedIteration.sequence);
      loadIteration(selectedIteration.sequence);
    } catch (err) {
      console.error('Failed to start iteration:', err);
    }
  };

  const handleCompare = async () => {
    if (!project || project.iterations.length < 2) return;

    // Default to comparing last two iterations
    const sequences = project.iterations.map((i) => i.sequence).sort((a, b) => a - b);
    const seqA = sequences[sequences.length - 2];
    const seqB = sequences[sequences.length - 1];

    try {
      const data = await api.compareIterations(projectId, seqA, seqB);
      setComparison(data);
      setShowCompare(true);
    } catch (err) {
      console.error('Failed to compare iterations:', err);
    }
  };

  useEffect(() => {
    loadProject();
    loadWorkProducts();
    loadAnnotations();
  }, [loadProject, loadWorkProducts, loadAnnotations]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
      </div>
    );
  }

  if (error || !project) {
    return (
      <div className="text-center py-12">
        <p className="text-red-500 mb-4">{error || 'Project not found'}</p>
        <Link
          href="/projects"
          className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
        >
          Back to Projects
        </Link>
      </div>
    );
  }

  return (
    <div className="min-h-screen -mx-4 -mt-8">
      {/* Back link */}
      <div className="px-6 py-3 border-b border-gray-200 bg-gray-50">
        <Link
          href="/projects"
          className="inline-flex items-center gap-1 text-sm text-gray-600 hover:text-gray-900"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          All Projects
        </Link>
      </div>

      {/* Project header */}
      <ProjectHeader project={project} onUpdate={setProject} />

      {/* Iteration tabs */}
      <IterationTabs
        iterations={project.iterations}
        selectedSequence={selectedIteration?.sequence}
        onSelect={loadIteration}
        onCompare={project.iterations.length >= 2 ? handleCompare : undefined}
      />

      {/* Main content */}
      <div className="flex">
        {/* Research content */}
        <div className="flex-1 p-6">
          {project.iterations.length === 0 ? (
            <div className="text-center py-16 bg-gray-50 rounded-lg border-2 border-dashed border-gray-200">
              <svg className="w-16 h-16 mx-auto text-gray-300 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
              <h3 className="text-lg font-medium text-gray-900 mb-2">No iterations yet</h3>
              <p className="text-gray-500 mb-4">
                Start your first research iteration for this project
              </p>
              <Link
                href={`/projects/${projectId}/iterate`}
                className="inline-flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Start First Iteration
              </Link>
            </div>
          ) : selectedIteration ? (
            <div>
              {/* Iteration header */}
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-lg font-semibold text-gray-900">
                    {selectedIteration.name || `Iteration ${selectedIteration.sequence}`}
                  </h2>
                  {selectedIteration.inherited_context && Object.keys(selectedIteration.inherited_context).length > 0 && (
                    <p className="text-sm text-purple-600 mt-1">
                      Building on context from previous iterations
                    </p>
                  )}
                </div>

                <div className="flex items-center gap-2">
                  {selectedIteration.status === 'pending' && !selectedIteration.research_job_id && (
                    <button
                      onClick={handleStartIteration}
                      className="flex items-center gap-2 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
                    >
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      Start Research
                    </button>
                  )}

                  <Link
                    href={`/projects/${projectId}/iterate`}
                    className="flex items-center gap-2 px-4 py-2 border border-gray-200 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                    </svg>
                    New Iteration
                  </Link>
                </div>
              </div>

              {/* Next Steps CTA — shown when research is completed */}
              {selectedIteration.research_job_status === 'completed' && (
                <div className="mb-4 p-4 bg-blue-50 border border-blue-100 rounded-lg">
                  <p className="text-sm font-medium text-blue-900 mb-3">Research complete — what&apos;s next?</p>
                  <div className="flex flex-wrap gap-2">
                    <Link
                      href={`/research/${selectedIteration.research_job_id}#generate`}
                      className="px-3 py-1.5 text-xs font-medium text-blue-700 bg-white border border-blue-200 rounded-md hover:bg-blue-50 transition-colors"
                    >
                      Generate Use Cases
                    </Link>
                    <Link
                      href={`/research/${selectedIteration.research_job_id}#generate`}
                      className="px-3 py-1.5 text-xs font-medium text-blue-700 bg-white border border-blue-200 rounded-md hover:bg-blue-50 transition-colors"
                    >
                      Build Personas
                    </Link>
                    <Link
                      href={`/research/${selectedIteration.research_job_id}#generate`}
                      className="px-3 py-1.5 text-xs font-medium text-blue-700 bg-white border border-blue-200 rounded-md hover:bg-blue-50 transition-colors"
                    >
                      Create One-Pager
                    </Link>
                    <Link
                      href={`/research/${selectedIteration.research_job_id}#generate`}
                      className="px-3 py-1.5 text-xs font-medium text-blue-700 bg-white border border-blue-200 rounded-md hover:bg-blue-50 transition-colors"
                    >
                      Account Plan
                    </Link>
                    <Link
                      href={`/projects/${projectId}/iterate`}
                      className="px-3 py-1.5 text-xs font-medium text-gray-600 bg-white border border-gray-200 rounded-md hover:bg-gray-50 transition-colors"
                    >
                      + New Research Round
                    </Link>
                  </div>
                </div>
              )}

              {/* Research results */}
              {selectedIteration.research_job_id ? (
                <ResearchResultsWrapper
                  researchJobId={selectedIteration.research_job_id}
                  status={selectedIteration.research_job_status || selectedIteration.status}
                  projectId={projectId}
                  iterationId={selectedIteration.id}
                />
              ) : (
                <div className="bg-gray-50 rounded-lg p-8 text-center">
                  <p className="text-gray-500">
                    {selectedIteration.status === 'pending'
                      ? 'Click "Start Research" to begin this iteration'
                      : `Status: ${selectedIteration.status}`}
                  </p>
                </div>
              )}
            </div>
          ) : null}
        </div>

        {/* Sidebar */}
        <div className="w-80 border-l border-gray-200 bg-gray-50">
          {/* Sidebar tabs */}
          <div className="flex border-b border-gray-200">
            <button
              onClick={() => setSidebarTab('saved')}
              className={`flex-1 px-3 py-3 text-xs font-medium transition-colors ${
                sidebarTab === 'saved'
                  ? 'bg-white border-b-2 border-blue-500 text-blue-600'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Saved ({workProducts.length})
            </button>
            <button
              onClick={() => setSidebarTab('notes')}
              className={`flex-1 px-3 py-3 text-xs font-medium transition-colors ${
                sidebarTab === 'notes'
                  ? 'bg-white border-b-2 border-blue-500 text-blue-600'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Notes ({annotations.length})
            </button>
            <button
              onClick={() => {
                setSidebarTab('generated');
                if (project) loadGeneratedAssets(project.iterations);
              }}
              className={`flex-1 px-3 py-3 text-xs font-medium transition-colors ${
                sidebarTab === 'generated'
                  ? 'bg-white border-b-2 border-blue-500 text-blue-600'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Generated
            </button>
          </div>

          {/* Sidebar content */}
          <div className="bg-white min-h-[400px]">
            {sidebarTab === 'saved' ? (
              <WorkProductPanel
                projectId={projectId}
                workProducts={workProducts}
                onUpdate={loadWorkProducts}
              />
            ) : sidebarTab === 'notes' ? (
              <AnnotationPanel
                projectId={projectId}
                annotations={annotations}
                onUpdate={loadAnnotations}
              />
            ) : (
              <GeneratedAssetsPanel
                assets={generatedAssets}
                loading={loadingGenerated}
              />
            )}
          </div>
        </div>
      </div>

      {/* Comparison modal */}
      {showCompare && comparison && (
        <IterationCompare
          comparison={comparison}
          onClose={() => {
            setShowCompare(false);
            setComparison(null);
          }}
        />
      )}
    </div>
  );
}

// Wrapper component to fetch and display research results
function ResearchResultsWrapper({
  researchJobId,
  status,
  projectId,
  iterationId,
}: {
  researchJobId: string;
  status: string;
  projectId: string;
  iterationId: string;
}) {
  const [job, setJob] = useState<ResearchJob | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadJob = async () => {
      try {
        const data = await api.getResearch(researchJobId);
        setJob(data);
      } catch (err) {
        console.error('Failed to load research job:', err);
      } finally {
        setIsLoading(false);
      }
    };

    loadJob();

    // Poll if running
    if (status === 'running') {
      const poller = api.pollResearch(researchJobId, (updated) => {
        setJob(updated);
      });
      poller.promise.catch(console.error);
      return () => poller.cancel();
    }
  }, [researchJobId, status]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
      </div>
    );
  }

  if (!job) {
    return (
      <div className="bg-red-50 text-red-700 p-4 rounded-lg">
        Failed to load research results
      </div>
    );
  }

  return <ResearchResults job={job} projectId={projectId} iterationId={iterationId} />;
}

// Panel showing all generated assets across project iterations
function GeneratedAssetsPanel({
  assets,
  loading,
}: {
  assets: { useCases: UseCase[]; personas: Persona[]; onePagers: OnePager[]; accountPlans: AccountPlan[] };
  loading: boolean;
}) {
  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-500" />
      </div>
    );
  }

  const total = assets.useCases.length + assets.personas.length + assets.onePagers.length + assets.accountPlans.length;

  if (total === 0) {
    return (
      <div className="p-4 text-center text-sm text-gray-500">
        <p>No generated assets yet.</p>
        <p className="mt-1 text-xs text-gray-400">Use the Generate tab in research results to create use cases, personas, and more.</p>
      </div>
    );
  }

  return (
    <div className="divide-y divide-gray-100">
      {assets.useCases.length > 0 && (
        <div className="p-3">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Use Cases ({assets.useCases.length})</p>
          <ul className="space-y-1">
            {assets.useCases.map((uc) => (
              <li key={uc.id} className="text-xs text-gray-800 truncate" title={uc.title}>{uc.title}</li>
            ))}
          </ul>
        </div>
      )}
      {assets.personas.length > 0 && (
        <div className="p-3">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Personas ({assets.personas.length})</p>
          <ul className="space-y-1">
            {assets.personas.map((p) => (
              <li key={p.id} className="text-xs text-gray-800 truncate" title={p.title}>{p.title}</li>
            ))}
          </ul>
        </div>
      )}
      {assets.onePagers.length > 0 && (
        <div className="p-3">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">One-Pagers ({assets.onePagers.length})</p>
          <ul className="space-y-1">
            {assets.onePagers.map((op) => (
              <li key={op.id} className="text-xs text-gray-800 truncate" title={op.title}>{op.title}</li>
            ))}
          </ul>
        </div>
      )}
      {assets.accountPlans.length > 0 && (
        <div className="p-3">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Account Plans ({assets.accountPlans.length})</p>
          <ul className="space-y-1">
            {assets.accountPlans.map((ap) => (
              <li key={ap.id} className="text-xs text-gray-800 truncate" title={ap.title}>{ap.title}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
