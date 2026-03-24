import { Persona } from '@/types';
import StarOrSaveButton from '@/components/research-results/shared/StarOrSaveButton';

function PersonaCard({ persona, clientName, projectId, iterationId }: { persona: Persona; clientName: string; projectId?: string; iterationId?: string }) {
  const initial = persona.name.charAt(0).toUpperCase();

  return (
    <div className="border border-gray-200 rounded-lg p-4 space-y-3">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-full bg-indigo-100 text-indigo-700 flex items-center justify-center font-semibold text-base flex-shrink-0">
          {initial}
        </div>
        <div className="flex-1 min-w-0">
          <p className="font-medium text-gray-900 text-sm">{persona.name}</p>
          <p className="text-xs text-gray-500">{persona.title} &middot; {persona.department}</p>
          {persona.seniority_level && (
            <span className="text-xs text-gray-400">{persona.seniority_level}</span>
          )}
        </div>
        <StarOrSaveButton
          clientName={clientName}
          projectId={projectId}
          iterationId={iterationId}
          contentType="assets.persona"
          objectId={persona.id}
          category="persona"
        />
      </div>

      {persona.background && (
        <p className="text-sm text-gray-700">{persona.background}</p>
      )}

      <div className="grid md:grid-cols-2 gap-3">
        {persona.goals.length > 0 && (
          <div>
            <p className="text-xs font-semibold text-gray-600 uppercase tracking-wide mb-1">Goals</p>
            <ul className="space-y-0.5">
              {persona.goals.slice(0, 3).map((g, i) => (
                <li key={i} className="text-xs text-gray-800 flex items-start gap-1">
                  <span className="text-green-500 mt-0.5">+</span>{g}
                </li>
              ))}
            </ul>
          </div>
        )}

        {persona.challenges.length > 0 && (
          <div>
            <p className="text-xs font-semibold text-gray-600 uppercase tracking-wide mb-1">Challenges</p>
            <ul className="space-y-0.5">
              {persona.challenges.slice(0, 3).map((c, i) => (
                <li key={i} className="text-xs text-gray-800 flex items-start gap-1">
                  <span className="text-red-400 mt-0.5">-</span>{c}
                </li>
              ))}
            </ul>
          </div>
        )}

        {persona.decision_criteria.length > 0 && (
          <div>
            <p className="text-xs font-semibold text-gray-600 uppercase tracking-wide mb-1">Decision Criteria</p>
            <ul className="space-y-0.5">
              {persona.decision_criteria.slice(0, 3).map((d, i) => (
                <li key={i} className="text-xs text-gray-800 flex items-start gap-1">
                  <span className="text-blue-400 mt-0.5">-</span>{d}
                </li>
              ))}
            </ul>
          </div>
        )}

        {persona.objections.length > 0 && (
          <div>
            <p className="text-xs font-semibold text-gray-600 uppercase tracking-wide mb-1">Likely Objections</p>
            <ul className="space-y-0.5">
              {persona.objections.slice(0, 3).map((o, i) => (
                <li key={i} className="text-xs text-gray-800 flex items-start gap-1">
                  <span className="text-orange-400 mt-0.5">!</span>{o}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {persona.key_messages.length > 0 && (
        <div className="pt-1 border-t border-gray-100">
          <p className="text-xs font-semibold text-gray-600 uppercase tracking-wide mb-1">Key Messages</p>
          <div className="flex flex-wrap gap-1">
            {persona.key_messages.map((m, i) => (
              <span key={i} className="px-2 py-0.5 bg-indigo-50 text-indigo-700 text-xs rounded">{m}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

interface PersonaSectionProps {
  clientName: string;
  projectId?: string;
  iterationId?: string;
  personas: Persona[];
  generating: boolean;
  onGenerate: () => void;
}

export default function PersonaSection({ clientName, projectId, iterationId, personas, generating, onGenerate }: PersonaSectionProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <h4 className="font-medium text-gray-900">Buyer Personas</h4>
          <p className="text-xs text-gray-500 mt-0.5">Target personas with goals, challenges, and objection handlers</p>
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
          {generating ? 'Generating...' : personas.length > 0 ? 'Regenerate' : 'Build Personas'}
        </button>
      </div>

      {personas.length > 0 ? (
        <div className="grid md:grid-cols-2 gap-4">
          {personas.map((p) => (
            <PersonaCard key={p.id} persona={p} clientName={clientName} projectId={projectId} iterationId={iterationId} />
          ))}
        </div>
      ) : (
        !generating && (
          <div className="p-6 bg-gray-50 rounded-lg text-center text-sm text-gray-500">
            No personas generated yet. Click &ldquo;Build Personas&rdquo; to start.
          </div>
        )
      )}
    </div>
  );
}
