import { ResearchReport, WebSource } from '@/types';
import MarkdownText from '../shared/MarkdownText';
import Section from '../shared/Section';
import DetailRow from '../shared/DetailRow';

/** Gemini may return tech_partnerships as objects {vendor, relationship_type, competitive_threat}
 *  instead of flat strings. Normalize either shape to a display string. */
function normalizePartner(partner: unknown): string {
  if (typeof partner === 'string') return partner;
  if (partner && typeof partner === 'object') {
    const p = partner as Record<string, string>;
    const parts = [p.vendor, p.relationship_type].filter(Boolean);
    if (parts.length) return parts.join(' — ');
    return JSON.stringify(partner);
  }
  return String(partner);
}

export default function ReportTab({ report, sources = [] }: { report: ResearchReport; sources?: WebSource[] }) {
  return (
    <div className="space-y-6">
      {/* Company Details */}
      <Section title="Company Details">
        <div className="grid md:grid-cols-2 gap-x-8">
          {report.headquarters && <DetailRow label="Headquarters" value={report.headquarters} />}
          {report.website && (
            <DetailRow label="Website" value={
              <a href={report.website} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                {report.website}
              </a>
            } />
          )}
          {report.founded_year && <DetailRow label="Founded" value={String(report.founded_year)} />}
          {report.employee_count && <DetailRow label="Employees" value={report.employee_count} />}
          {report.annual_revenue && <DetailRow label="Revenue" value={report.annual_revenue} />}
        </div>
      </Section>

      {/* AI Assessment */}
      <Section title="Digital & AI Assessment">
        <div className="grid md:grid-cols-3 gap-4 mb-4">
          {report.digital_maturity && (
            <div className="p-3 bg-gray-50 rounded-lg text-center border border-gray-200">
              <div className="text-sm font-medium text-gray-500 mb-1">Digital Maturity</div>
              <div className="text-lg font-semibold text-gray-900 capitalize">{report.digital_maturity}</div>
            </div>
          )}
          {report.ai_adoption_stage && (
            <div className="p-3 bg-gray-50 rounded-lg text-center border border-gray-200">
              <div className="text-sm font-medium text-gray-500 mb-1">AI Adoption</div>
              <div className="text-lg font-semibold text-gray-900 capitalize">{report.ai_adoption_stage}</div>
            </div>
          )}
        </div>
        {report.ai_footprint && (
          <MarkdownText content={report.ai_footprint} className="text-gray-900" sources={sources} />
        )}
      </Section>

      {/* Infrastructure & Security */}
      {(report.cloud_footprint || report.security_posture || report.data_maturity) && (
        <Section title="Infrastructure & Security">
          <div className="space-y-4">
            {report.cloud_footprint && (
              <div>
                <div className="text-sm font-medium text-gray-700 mb-1">Cloud Footprint</div>
                <MarkdownText content={report.cloud_footprint} className="text-gray-900" sources={sources} />
              </div>
            )}
            {report.security_posture && (
              <div>
                <div className="text-sm font-medium text-gray-700 mb-1">Security Posture</div>
                <MarkdownText content={report.security_posture} className="text-gray-900" sources={sources} />
              </div>
            )}
            {report.data_maturity && (
              <div>
                <div className="text-sm font-medium text-gray-700 mb-1">Data Maturity</div>
                <MarkdownText content={report.data_maturity} className="text-gray-900" sources={sources} />
              </div>
            )}
          </div>
        </Section>
      )}

      {/* Recent News */}
      {report.recent_news && report.recent_news.length > 0 && (
        <Section title="Recent News">
          <div className="space-y-3">
            {report.recent_news.map((news, i) => (
              <div key={i} className="p-3 bg-gray-50 rounded-lg">
                <div className="font-medium text-gray-900">{news.title}</div>
                <div className="text-sm text-gray-800 mt-1">{news.summary}</div>
                <div className="text-xs text-gray-600 mt-2">
                  {news.date && <span>{news.date}</span>}
                  {news.source && <span> - {news.source}</span>}
                </div>
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Strategic Goals */}
      {report.strategic_goals && report.strategic_goals.length > 0 && (
        <Section title="Strategic Goals">
          <ul className="space-y-2">
            {report.strategic_goals.map((goal, i) => (
              <li key={i} className="flex items-start">
                <span className="text-blue-600 mr-2">{i + 1}.</span>
                <span className="text-gray-900">{goal}</span>
              </li>
            ))}
          </ul>
        </Section>
      )}

      {/* Key Initiatives */}
      {report.key_initiatives && report.key_initiatives.length > 0 && (
        <Section title="Key Initiatives">
          <ul className="space-y-2">
            {report.key_initiatives.map((init, i) => (
              <li key={i} className="p-3 bg-gray-50 rounded-lg text-gray-900">
                {init}
              </li>
            ))}
          </ul>
        </Section>
      )}

      {/* Strategic Intelligence */}
      {((report.financial_signals && report.financial_signals.length > 0) ||
        (report.tech_partnerships && report.tech_partnerships.length > 0)) && (
        <Section title="Strategic Intelligence">
          <div className="space-y-4">
            {report.financial_signals && report.financial_signals.length > 0 && (
              <div>
                <div className="text-sm font-medium text-gray-700 mb-2">Financial Signals</div>
                <ul className="space-y-2">
                  {report.financial_signals.map((signal, i) => (
                    <li key={i} className="flex items-start">
                      <span className="text-blue-600 mr-2">{i + 1}.</span>
                      <span className="text-gray-900">{signal}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {report.tech_partnerships && report.tech_partnerships.length > 0 && (
              <div>
                <div className="text-sm font-medium text-gray-700 mb-2">Technology Partnerships</div>
                <div className="flex flex-wrap gap-1">
                  {report.tech_partnerships.map((partner, i) => (
                    <span key={i} className="px-2 py-0.5 bg-gray-100 text-gray-800 text-xs rounded">
                      {normalizePartner(partner)}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </Section>
      )}
    </div>
  );
}
