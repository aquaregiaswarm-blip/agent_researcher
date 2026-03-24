'use client';

import { useState } from 'react';
import { UseCase, FeasibilityAssessment, RefinedPlay } from '@/types';
import { api } from '@/lib/api';
import { useToast } from '@/lib/toast';
import StarOrSaveButton from '@/components/research-results/shared/StarOrSaveButton';

const PRIORITY_COLORS: Record<string, string> = {
  high: 'bg-red-100 text-red-800',
  medium: 'bg-yellow-100 text-yellow-800',
  low: 'bg-gray-100 text-gray-700',
};

function FeasibilityPanel({ assessment }: { assessment: FeasibilityAssessment }) {
  const levelColors: Record<string, string> = {
    high: 'text-green-700 bg-green-50 border-green-200',
    medium: 'text-yellow-700 bg-yellow-50 border-yellow-200',
    low: 'text-red-700 bg-red-50 border-red-200',
  };
  const cls = levelColors[assessment.overall_feasibility] ?? levelColors.medium;

  return (
    <div className="mt-3 p-3 bg-gray-50 border border-gray-200 rounded-lg space-y-3">
      <div className="flex items-center gap-2">
        <span className={`px-2 py-0.5 text-xs font-semibold rounded border ${cls} capitalize`}>
          {assessment.overall_feasibility} feasibility
        </span>
        <span className="text-xs text-gray-500">Score: {Math.round(assessment.overall_score * 100)}%</span>
      </div>

      {assessment.technical_risks.length > 0 && (
        <div>
          <p className="text-xs font-medium text-gray-700 mb-1">Technical Risks</p>
          <ul className="space-y-0.5">
            {assessment.technical_risks.map((r, i) => (
              <li key={i} className="text-xs text-gray-800 flex items-start gap-1">
                <span className="text-red-500 mt-0.5">-</span>{r}
              </li>
            ))}
          </ul>
        </div>
      )}

      {assessment.prerequisites.length > 0 && (
        <div>
          <p className="text-xs font-medium text-gray-700 mb-1">Prerequisites</p>
          <ul className="space-y-0.5">
            {assessment.prerequisites.map((p, i) => (
              <li key={i} className="text-xs text-gray-800 flex items-start gap-1">
                <span className="text-blue-500 mt-0.5">-</span>{p}
              </li>
            ))}
          </ul>
        </div>
      )}

      {assessment.recommendations.length > 0 && (
        <div>
          <p className="text-xs font-medium text-gray-700 mb-1">Recommendations</p>
          <ul className="space-y-0.5">
            {assessment.recommendations.map((r, i) => (
              <li key={i} className="text-xs text-gray-800 flex items-start gap-1">
                <span className="text-green-600 mt-0.5">+</span>{r}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function SalesPlayCard({ play, clientName, projectId, iterationId }: { play: RefinedPlay; clientName: string; projectId?: string; iterationId?: string }) {
  return (
    <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold text-blue-800 uppercase tracking-wide">Elevator Pitch</p>
        <StarOrSaveButton
          clientName={clientName}
          projectId={projectId}
          iterationId={iterationId}
          contentType="ideation.refinedplay"
          objectId={play.id}
          category="play"
        />
      </div>
      <p className="text-sm text-blue-900">{play.elevator_pitch}</p>

      {play.discovery_questions.slice(0, 3).length > 0 && (
        <div>
          <p className="text-xs font-semibold text-blue-800 uppercase tracking-wide mb-1">Discovery Questions</p>
          <ol className="space-y-1">
            {play.discovery_questions.slice(0, 3).map((q, i) => (
              <li key={i} className="text-sm text-blue-900">
                <span className="font-medium">{i + 1}.</span> {q}
              </li>
            ))}
          </ol>
        </div>
      )}

      {play.objection_handlers.slice(0, 3).length > 0 && (
        <div>
          <p className="text-xs font-semibold text-blue-800 uppercase tracking-wide mb-1">Objection Handlers</p>
          <ul className="space-y-1">
            {play.objection_handlers.slice(0, 3).map((o, i) => (
              <li key={i} className="text-sm text-blue-900 flex items-start gap-1">
                <span className="text-blue-500 mt-0.5">-</span>{o}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function UseCaseCard({ uc, clientName, projectId, iterationId }: { uc: UseCase; clientName: string; projectId?: string; iterationId?: string }) {
  const { addToast, removeToast } = useToast();
  const [feasibility, setFeasibility] = useState<FeasibilityAssessment | null>(null);
  const [play, setPlay] = useState<RefinedPlay | null>(null);
  const [assessingFeasibility, setAssessingFeasibility] = useState(false);
  const [refiningPlay, setRefiningPlay] = useState(false);

  const handleAssess = async () => {
    setAssessingFeasibility(true);
    const tid = addToast({ type: 'loading', message: 'Assessing feasibility...' });
    try {
      const result = await api.assessFeasibility(uc.id);
      setFeasibility(result);
      removeToast(tid);
      addToast({ type: 'success', message: 'Feasibility assessment complete' });
    } catch {
      removeToast(tid);
      addToast({ type: 'error', message: 'Failed to assess feasibility' });
    } finally {
      setAssessingFeasibility(false);
    }
  };

  const handleRefine = async () => {
    setRefiningPlay(true);
    const tid = addToast({ type: 'loading', message: 'Refining sales play...' });
    try {
      const result = await api.refinePlay(uc.id);
      setPlay(result);
      removeToast(tid);
      addToast({ type: 'success', message: 'Sales play ready' });
    } catch {
      removeToast(tid);
      addToast({ type: 'error', message: 'Failed to refine play' });
    } finally {
      setRefiningPlay(false);
    }
  };

  return (
    <div className="border border-gray-200 rounded-lg p-4 space-y-3">
      <div className="flex items-start justify-between gap-2">
        <h5 className="font-medium text-gray-900 text-sm">{uc.title}</h5>
        <div className="flex items-center gap-1.5 flex-shrink-0">
          <span className={`px-2 py-0.5 text-xs font-medium rounded capitalize ${PRIORITY_COLORS[uc.priority] ?? PRIORITY_COLORS.low}`}>
            {uc.priority}
          </span>
          <StarOrSaveButton
            clientName={clientName}
            projectId={projectId}
            iterationId={iterationId}
            contentType="ideation.usecase"
            objectId={uc.id}
            category="use_case"
          />
        </div>
      </div>

      <p className="text-sm text-gray-700">{uc.business_problem}</p>

      {uc.estimated_roi && (
        <p className="text-xs text-green-700 font-medium">ROI: {uc.estimated_roi}</p>
      )}

      {uc.technologies.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {uc.technologies.map((t, i) => (
            <span key={i} className="px-2 py-0.5 bg-gray-100 text-gray-700 text-xs rounded">{t}</span>
          ))}
        </div>
      )}

      {/* Step 2: Feasibility */}
      {feasibility ? (
        <>
          <FeasibilityPanel assessment={feasibility} />
          {/* Step 3: Refine */}
          {play ? (
            <SalesPlayCard play={play} clientName={clientName} projectId={projectId} iterationId={iterationId} />
          ) : (
            <button
              onClick={handleRefine}
              disabled={refiningPlay}
              className="w-full mt-2 px-3 py-1.5 text-xs font-medium text-blue-700 bg-blue-50 hover:bg-blue-100 border border-blue-200 rounded-md disabled:opacity-50 transition-colors flex items-center justify-center gap-1.5"
            >
              {refiningPlay && (
                <svg className="animate-spin h-3 w-3" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
              )}
              {refiningPlay ? 'Refining play...' : 'Refine into Sales Play'}
            </button>
          )}
        </>
      ) : (
        <button
          onClick={handleAssess}
          disabled={assessingFeasibility}
          className="w-full px-3 py-1.5 text-xs font-medium text-gray-700 bg-gray-50 hover:bg-gray-100 border border-gray-200 rounded-md disabled:opacity-50 transition-colors flex items-center justify-center gap-1.5"
        >
          {assessingFeasibility && (
            <svg className="animate-spin h-3 w-3" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
          )}
          {assessingFeasibility ? 'Assessing...' : 'Assess Feasibility'}
        </button>
      )}
    </div>
  );
}

interface UseCaseSectionProps {
  researchJobId: string;
  clientName: string;
  projectId?: string;
  iterationId?: string;
  useCases: UseCase[];
  generating: boolean;
  onGenerate: () => void;
}

export default function UseCaseSection({ clientName, projectId, iterationId, useCases, generating, onGenerate }: UseCaseSectionProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <h4 className="font-medium text-gray-900">Use Cases</h4>
          <p className="text-xs text-gray-500 mt-0.5">AI-generated use cases based on identified gaps and pain points</p>
        </div>
        <button
          onClick={onGenerate}
          disabled={generating}
          className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-md disabled:opacity-50 transition-colors"
        >
          {generating && (
            <svg className="animate-spin h-3.5 w-3.5" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
          )}
          {generating ? 'Generating...' : useCases.length > 0 ? 'Regenerate' : 'Generate Use Cases'}
        </button>
      </div>

      {useCases.length > 0 ? (
        <div className="space-y-3">
          {useCases.map((uc) => (
            <UseCaseCard key={uc.id} uc={uc} clientName={clientName} projectId={projectId} iterationId={iterationId} />
          ))}
        </div>
      ) : (
        !generating && (
          <div className="p-6 bg-gray-50 rounded-lg text-center text-sm text-gray-500">
            No use cases generated yet. Click &ldquo;Generate Use Cases&rdquo; to start.
          </div>
        )
      )}
    </div>
  );
}
