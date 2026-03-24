'use client';

import { useState } from 'react';
import DOMPurify from 'dompurify';
import { OnePager } from '@/types';

interface OnePagerSectionProps {
  onePager: OnePager | null;
  generating: boolean;
  onGenerate: () => void;
}

export default function OnePagerSection({ onePager, generating, onGenerate }: OnePagerSectionProps) {
  const [fullScreen, setFullScreen] = useState(false);

  const handlePrint = () => {
    if (!onePager?.html_content) return;
    const win = window.open('', '_blank');
    if (!win) return;
    // Scope DOMPurify to the new window's document to avoid cross-document bypass
    const purify = DOMPurify(win as unknown as typeof window);
    win.document.write(purify.sanitize(onePager.html_content));
    win.document.close();
    win.focus();
    win.print();
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <h4 className="font-medium text-gray-900">One-Pager</h4>
          <p className="text-xs text-gray-500 mt-0.5">Executive summary document for the opportunity</p>
        </div>
        <div className="flex items-center gap-2">
          {onePager && (
            <>
              <button
                onClick={handlePrint}
                className="px-3 py-1.5 text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 border border-gray-200 rounded-md transition-colors"
              >
                Print / PDF
              </button>
              <button
                onClick={() => setFullScreen(true)}
                className="px-3 py-1.5 text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 border border-gray-200 rounded-md transition-colors"
              >
                Full Screen
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
            {generating ? 'Generating...' : onePager ? 'Regenerate' : 'Create One-Pager'}
          </button>
        </div>
      </div>

      {onePager ? (
        onePager.html_content ? (
          <div className="border border-gray-200 rounded-lg overflow-hidden">
            <div className="bg-gray-50 px-3 py-2 border-b border-gray-200 text-xs text-gray-600">
              {onePager.title || 'One-Pager Preview'}
            </div>
            <div
              className="p-4 max-h-[500px] overflow-auto text-sm"
              dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(onePager.html_content) }}
            />
          </div>
        ) : (
          <div className="p-4 bg-gray-50 rounded-lg space-y-2 text-sm">
            <p className="font-medium text-gray-900">{onePager.title}</p>
            {onePager.headline && <p className="text-gray-700 italic">{onePager.headline}</p>}
            {onePager.executive_summary && <p className="text-gray-800">{onePager.executive_summary}</p>}
          </div>
        )
      ) : (
        !generating && (
          <div className="p-6 bg-gray-50 rounded-lg text-center text-sm text-gray-500">
            No one-pager generated yet. Click &ldquo;Create One-Pager&rdquo; to start.
          </div>
        )
      )}

      {/* Full-screen modal */}
      {fullScreen && onePager?.html_content && (
        <div className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] flex flex-col">
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200">
              <span className="font-medium text-gray-900">{onePager.title || 'One-Pager'}</span>
              <div className="flex items-center gap-2">
                <button
                  onClick={handlePrint}
                  className="px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-100 rounded"
                >
                  Print / PDF
                </button>
                <button
                  onClick={() => setFullScreen(false)}
                  className="p-1.5 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded"
                  aria-label="Close"
                >
                  <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>
            <div
              className="flex-1 overflow-auto p-6"
              dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(onePager.html_content) }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
