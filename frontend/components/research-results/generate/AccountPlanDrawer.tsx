'use client';

import { useEffect } from 'react';
import { createPortal } from 'react-dom';
import DOMPurify from 'dompurify';
import { AccountPlan } from '@/types';

interface AccountPlanDrawerProps {
  plan: AccountPlan;
  onClose: () => void;
}

function SwotGrid({ swot }: { swot: AccountPlan['swot_analysis'] }) {
  const cells = [
    { label: 'Strengths', items: swot.strengths, color: 'bg-green-50 border-green-200' },
    { label: 'Weaknesses', items: swot.weaknesses, color: 'bg-red-50 border-red-200' },
    { label: 'Opportunities', items: swot.opportunities, color: 'bg-blue-50 border-blue-200' },
    { label: 'Threats', items: swot.threats, color: 'bg-orange-50 border-orange-200' },
  ];
  return (
    <div className="grid grid-cols-2 gap-3">
      {cells.map(({ label, items, color }) => (
        <div key={label} className={`p-3 rounded-lg border ${color}`}>
          <p className="text-xs font-semibold text-gray-700 mb-2">{label}</p>
          <ul className="space-y-1">
            {items.map((item, i) => (
              <li key={i} className="text-xs text-gray-800">- {item}</li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

function DrawerContent({ plan, onClose }: AccountPlanDrawerProps) {
  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [onClose]);

  const handlePrint = () => {
    if (!plan.html_content) return;
    const win = window.open('', '_blank');
    if (!win) return;
    // Scope DOMPurify to the new window's document to avoid cross-document bypass
    const purify = DOMPurify(win as unknown as typeof window);
    win.document.write(purify.sanitize(plan.html_content));
    win.document.close();
    win.focus();
    win.print();
  };

  return (
    <div className="fixed inset-0 z-50 flex">
      {/* Backdrop */}
      <div
        className="flex-1 bg-black/50"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Drawer */}
      <div className="w-full max-w-2xl bg-white h-full flex flex-col shadow-2xl">
        {/* Sticky header */}
        <div className="sticky top-0 bg-white z-10 px-5 py-4 border-b border-gray-200 flex items-center justify-between">
          <h2 className="font-semibold text-gray-900 truncate pr-4">{plan.title || 'Account Plan'}</h2>
          <div className="flex items-center gap-2 flex-shrink-0">
            {plan.html_content && (
              <button
                onClick={handlePrint}
                className="px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-100 border border-gray-200 rounded-md transition-colors"
              >
                Print / PDF
              </button>
            )}
            <button
              onClick={onClose}
              className="p-1.5 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded"
              aria-label="Close account plan"
            >
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Scrollable content */}
        <div className="flex-1 overflow-y-auto p-5 space-y-6">
          {plan.html_content ? (
            <div dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(plan.html_content) }} className="prose prose-sm max-w-none" />
          ) : (
            <>
              {plan.executive_summary && (
                <section>
                  <h3 className="text-sm font-semibold text-gray-900 mb-2">Executive Summary</h3>
                  <p className="text-sm text-gray-700">{plan.executive_summary}</p>
                </section>
              )}

              {plan.account_overview && (
                <section>
                  <h3 className="text-sm font-semibold text-gray-900 mb-2">Account Overview</h3>
                  <p className="text-sm text-gray-700">{plan.account_overview}</p>
                </section>
              )}

              {plan.strategic_objectives.length > 0 && (
                <section>
                  <h3 className="text-sm font-semibold text-gray-900 mb-2">Strategic Objectives</h3>
                  <ol className="space-y-1">
                    {plan.strategic_objectives.map((obj, i) => (
                      <li key={i} className="text-sm text-gray-800 flex items-start gap-2">
                        <span className="text-blue-600 font-medium">{i + 1}.</span>{obj}
                      </li>
                    ))}
                  </ol>
                </section>
              )}

              {plan.swot_analysis && (
                <section>
                  <h3 className="text-sm font-semibold text-gray-900 mb-2">SWOT Analysis</h3>
                  <SwotGrid swot={plan.swot_analysis} />
                </section>
              )}

              {plan.engagement_strategy && (
                <section>
                  <h3 className="text-sm font-semibold text-gray-900 mb-2">Engagement Strategy</h3>
                  <p className="text-sm text-gray-700">{plan.engagement_strategy}</p>
                </section>
              )}

              {plan.value_propositions.length > 0 && (
                <section>
                  <h3 className="text-sm font-semibold text-gray-900 mb-2">Value Propositions</h3>
                  <ul className="space-y-1">
                    {plan.value_propositions.map((vp, i) => (
                      <li key={i} className="text-sm text-gray-800 flex items-start gap-2">
                        <span className="text-green-600">+</span>{vp}
                      </li>
                    ))}
                  </ul>
                </section>
              )}

              {plan.success_metrics.length > 0 && (
                <section>
                  <h3 className="text-sm font-semibold text-gray-900 mb-2">Success Metrics</h3>
                  <ul className="space-y-1">
                    {plan.success_metrics.map((m, i) => (
                      <li key={i} className="text-sm text-gray-800 flex items-start gap-2">
                        <span className="text-blue-500">-</span>{m}
                      </li>
                    ))}
                  </ul>
                </section>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default function AccountPlanDrawer({ plan, onClose }: AccountPlanDrawerProps) {
  if (typeof document === 'undefined') return null;
  return createPortal(<DrawerContent plan={plan} onClose={onClose} />, document.body);
}
