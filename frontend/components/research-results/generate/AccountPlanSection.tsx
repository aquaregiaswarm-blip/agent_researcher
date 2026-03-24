'use client';

import { useState } from 'react';
import { AccountPlan } from '@/types';
import AccountPlanDrawer from './AccountPlanDrawer';
import StarOrSaveButton from '@/components/research-results/shared/StarOrSaveButton';

interface AccountPlanSectionProps {
  clientName: string;
  projectId?: string;
  iterationId?: string;
  accountPlan: AccountPlan | null;
  generating: boolean;
  onGenerate: () => void;
}

export default function AccountPlanSection({ clientName, projectId, iterationId, accountPlan, generating, onGenerate }: AccountPlanSectionProps) {
  const [drawerOpen, setDrawerOpen] = useState(false);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <h4 className="font-medium text-gray-900">Account Plan</h4>
          <p className="text-xs text-gray-500 mt-0.5">Strategic account plan with SWOT, engagement strategy, and milestones</p>
        </div>
        <div className="flex items-center gap-2">
          {accountPlan && (
            <>
              <StarOrSaveButton
                clientName={clientName}
                projectId={projectId}
                iterationId={iterationId}
                contentType="assets.accountplan"
                objectId={accountPlan.id}
                category="other"
              />
              <button
                onClick={() => setDrawerOpen(true)}
                className="px-3 py-1.5 text-sm font-medium text-blue-700 bg-blue-50 hover:bg-blue-100 border border-blue-200 rounded-md transition-colors"
              >
                View Plan
              </button>
            </>
          )}
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
            {generating ? 'Generating...' : accountPlan ? 'Regenerate' : 'Build Account Plan'}
          </button>
        </div>
      </div>

      {accountPlan ? (
        <div
          className="p-4 bg-gray-50 border border-gray-200 rounded-lg cursor-pointer hover:border-blue-300 hover:bg-blue-50 transition-colors"
          onClick={() => setDrawerOpen(true)}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === 'Enter' && setDrawerOpen(true)}
        >
          <p className="font-medium text-gray-900 text-sm">{accountPlan.title}</p>
          {accountPlan.executive_summary && (
            <p className="text-sm text-gray-600 mt-1 line-clamp-2">{accountPlan.executive_summary}</p>
          )}
          {accountPlan.strategic_objectives.length > 0 && (
            <p className="text-xs text-gray-500 mt-2">
              {accountPlan.strategic_objectives.length} strategic objective{accountPlan.strategic_objectives.length !== 1 ? 's' : ''} &middot; Click to view full plan
            </p>
          )}
        </div>
      ) : (
        !generating && (
          <div className="p-6 bg-gray-50 rounded-lg text-center text-sm text-gray-500">
            No account plan generated yet. Click &ldquo;Build Account Plan&rdquo; to start.
          </div>
        )
      )}

      {drawerOpen && accountPlan && (
        <AccountPlanDrawer plan={accountPlan} onClose={() => setDrawerOpen(false)} />
      )}
    </div>
  );
}
