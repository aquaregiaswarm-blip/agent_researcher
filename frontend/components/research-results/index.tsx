'use client';

import { useState } from 'react';
import { ResearchJob } from '@/types';
import { api } from '@/lib/api';
import { ResearchAnimation } from '@/components/ResearchAnimation';
import OverviewTab from './tabs/OverviewTab';
import ReportTab from './tabs/ReportTab';
import CompetitorsTab from './tabs/CompetitorsTab';
import GapsTab from './tabs/GapsTab';
import InsideIntelTab from './tabs/InsideIntelTab';
import SourcesTab from './tabs/SourcesTab';
import RawTab from './tabs/RawTab';
import GenerateTab from './tabs/GenerateTab';
import ResearchCompletionBanner from '@/components/ResearchCompletionBanner';

interface ResearchResultsProps {
  job: ResearchJob;
  projectId?: string;
  iterationId?: string;
}

type TabId = 'overview' | 'report' | 'competitors' | 'gaps' | 'intel' | 'sources' | 'raw' | 'generate';

export default function ResearchResults({ job, projectId, iterationId }: ResearchResultsProps) {
  const [activeTab, setActiveTab] = useState<TabId>('overview');
  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);

  const handleExportPdf = async () => {
    setIsExporting(true);
    setExportError(null);
    try {
      await api.downloadResearchPdf(job.id);
    } catch (error) {
      setExportError(error instanceof Error ? error.message : 'Failed to export PDF');
    } finally {
      setIsExporting(false);
    }
  };

  const isCompleted = job.status === 'completed';

  const tabs = [
    { id: 'overview' as TabId, label: 'Overview', available: true, cta: false },
    { id: 'report' as TabId, label: 'Full Report', available: !!job.report, cta: false },
    { id: 'competitors' as TabId, label: 'Competitors', available: !!job.competitor_case_studies?.length, cta: false },
    { id: 'gaps' as TabId, label: 'Gap Analysis', available: !!job.gap_analysis, cta: false },
    { id: 'intel' as TabId, label: 'Org Signals', available: !!job.internal_ops, cta: false },
    { id: 'sources' as TabId, label: 'Sources', available: !!(job.report?.web_sources?.length), cta: false },
    { id: 'raw' as TabId, label: 'Debug', available: !!job.result, cta: false },
    { id: 'generate' as TabId, label: 'Generate', available: isCompleted, cta: true },
  ];

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 bg-gray-50 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-medium text-gray-900">{job.client_name}</h3>
            {job.vertical && (
              <span className="text-sm text-gray-500 capitalize">{job.vertical.replace('_', ' ')}</span>
            )}
          </div>
          <div className="flex items-center gap-3">
            {job.status === 'completed' && (
              <button
                onClick={handleExportPdf}
                disabled={isExporting}
                className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-md disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {isExporting ? (
                  <>
                    <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Exporting...
                  </>
                ) : (
                  <>
                    <svg className="h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
                    </svg>
                    Export PDF
                  </>
                )}
              </button>
            )}
            <span className={`px-2 py-1 text-xs font-medium rounded-full ${
              job.status === 'completed' ? 'bg-green-100 text-green-800' :
              job.status === 'failed' ? 'bg-red-100 text-red-800' :
              job.status === 'running' ? 'bg-blue-100 text-blue-800' :
              'bg-gray-100 text-gray-800'
            }`}>
              {job.status}
            </span>
          </div>
        </div>
        {exportError && (
          <div className="mt-2 text-sm text-red-600">
            {exportError}
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 bg-white">
        <nav className="flex -mb-px overflow-x-auto items-center">
          {tabs.filter(t => t.available).map((tab) => (
            tab.cta ? (
              <button
                key={tab.id}
                data-testid={`tab-${tab.id}`}
                onClick={() => setActiveTab(tab.id)}
                className={`ml-auto mr-2 my-1.5 px-3 py-1 text-sm font-medium whitespace-nowrap rounded-full transition-colors ${
                  activeTab === tab.id
                    ? 'bg-blue-600 text-white'
                    : 'bg-blue-50 text-blue-700 hover:bg-blue-100 border border-blue-200'
                }`}
              >
                {tab.label}
              </button>
            ) : (
              <button
                key={tab.id}
                data-testid={`tab-${tab.id}`}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 text-sm font-medium whitespace-nowrap border-b-2 ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.label}
              </button>
            )
          ))}
        </nav>
      </div>

      {/* Completion banner */}
      {isCompleted && activeTab !== 'generate' && (
        <ResearchCompletionBanner
          jobId={job.id}
          onOpenGenerate={() => setActiveTab('generate')}
        />
      )}

      {/* Content */}
      <div className="p-4 bg-white">
        {job.status === 'running' || job.status === 'pending' ? (
          <ResearchAnimation currentStep={job.current_step} clientName={job.client_name} />
        ) : job.status === 'failed' ? (
          <p className="text-red-600">{job.error || 'Research failed'}</p>
        ) : (
          <>
            {activeTab === 'overview' && <OverviewTab job={job} sources={job.report?.web_sources ?? []} />}
            {activeTab === 'report' && job.report && <ReportTab report={job.report} sources={job.report.web_sources ?? []} />}
            {activeTab === 'competitors' && job.competitor_case_studies && (
              <CompetitorsTab
                caseStudies={job.competitor_case_studies}
                projectId={projectId}
                iterationId={iterationId}
                clientName={job.client_name}
              />
            )}
            {activeTab === 'gaps' && job.gap_analysis && (
              <GapsTab
                gaps={job.gap_analysis}
                sources={job.report?.web_sources ?? []}
                projectId={projectId}
                iterationId={iterationId}
                clientName={job.client_name}
              />
            )}
            {activeTab === 'intel' && job.internal_ops && <InsideIntelTab intel={job.internal_ops} sources={job.report?.web_sources ?? []} />}
            {activeTab === 'sources' && job.report?.web_sources && (
              <SourcesTab sources={job.report.web_sources} />
            )}
            {activeTab === 'raw' && <RawTab result={job.result} />}
            {activeTab === 'generate' && (
              <GenerateTab
                researchJobId={job.id}
                clientName={job.client_name}
                projectId={projectId}
                iterationId={iterationId}
              />
            )}
          </>
        )}
      </div>
    </div>
  );
}
