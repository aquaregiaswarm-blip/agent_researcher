'use client';

import { useState } from 'react';
import { ResearchJob, ResearchReport, CompetitorCaseStudy, GapAnalysis, InternalOpsIntel, GapCorrelation, WebSource } from '@/types';

interface ResearchResultsProps {
  job: ResearchJob;
}

type TabId = 'overview' | 'report' | 'competitors' | 'gaps' | 'intel' | 'sources' | 'raw';

export default function ResearchResults({ job }: ResearchResultsProps) {
  const [activeTab, setActiveTab] = useState<TabId>('overview');

  const tabs = [
    { id: 'overview' as TabId, label: 'Overview', available: true },
    { id: 'report' as TabId, label: 'Deep Research', available: !!job.report },
    { id: 'competitors' as TabId, label: 'Competitors', available: !!job.competitor_case_studies?.length },
    { id: 'gaps' as TabId, label: 'Gap Analysis', available: !!job.gap_analysis },
    { id: 'intel' as TabId, label: 'Inside Intel', available: !!job.internal_ops },
    { id: 'sources' as TabId, label: 'Sources', available: !!(job.report?.web_sources?.length) },
    { id: 'raw' as TabId, label: 'Raw Output', available: !!job.result },
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

      {/* Tabs */}
      <div className="border-b border-gray-200 bg-white">
        <nav className="flex -mb-px overflow-x-auto">
          {tabs.filter(t => t.available).map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-2 text-sm font-medium whitespace-nowrap border-b-2 ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Content */}
      <div className="p-4 bg-white">
        {job.status === 'running' || job.status === 'pending' ? (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <span className="ml-3 text-gray-600">Researching...</span>
          </div>
        ) : job.status === 'failed' ? (
          <p className="text-red-600">{job.error || 'Research failed'}</p>
        ) : (
          <>
            {activeTab === 'overview' && <OverviewTab job={job} />}
            {activeTab === 'report' && job.report && <ReportTab report={job.report} />}
            {activeTab === 'competitors' && job.competitor_case_studies && (
              <CompetitorsTab caseStudies={job.competitor_case_studies} />
            )}
            {activeTab === 'gaps' && job.gap_analysis && <GapsTab gaps={job.gap_analysis} />}
            {activeTab === 'intel' && job.internal_ops && <InsideIntelTab intel={job.internal_ops} />}
            {activeTab === 'sources' && job.report?.web_sources && (
              <SourcesTab sources={job.report.web_sources} />
            )}
            {activeTab === 'raw' && <RawTab result={job.result} />}
          </>
        )}
      </div>
    </div>
  );
}

function OverviewTab({ job }: { job: ResearchJob }) {
  const report = job.report;

  return (
    <div className="space-y-6">
      {/* Quick Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {report?.founded_year && (
          <StatCard label="Founded" value={String(report.founded_year)} />
        )}
        {report?.employee_count && (
          <StatCard label="Employees" value={report.employee_count} />
        )}
        {report?.annual_revenue && (
          <StatCard label="Revenue" value={report.annual_revenue} />
        )}
        {report?.digital_maturity && (
          <StatCard label="Digital Maturity" value={report.digital_maturity} className="capitalize" />
        )}
      </div>

      {/* Company Overview */}
      {report?.company_overview && (
        <Section title="Company Overview">
          <p className="text-gray-900">{report.company_overview}</p>
        </Section>
      )}

      {/* Key Decision Makers */}
      {report?.decision_makers && report.decision_makers.length > 0 && (
        <Section title="Key Decision Makers">
          <div className="grid gap-3">
            {report.decision_makers.map((dm, i) => (
              <div key={i} className="p-3 bg-gray-50 rounded-lg">
                <div className="font-medium text-gray-900">{dm.name}</div>
                <div className="text-sm text-gray-800">{dm.title}</div>
                {dm.background && (
                  <div className="text-sm text-gray-700 mt-1">{dm.background}</div>
                )}
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Pain Points & Opportunities */}
      <div className="grid md:grid-cols-2 gap-6">
        {report?.pain_points && report.pain_points.length > 0 && (
          <Section title="Pain Points">
            <ul className="space-y-2">
              {report.pain_points.map((point, i) => (
                <li key={i} className="flex items-start">
                  <span className="text-red-600 mr-2">-</span>
                  <span className="text-gray-900">{point}</span>
                </li>
              ))}
            </ul>
          </Section>
        )}
        {report?.opportunities && report.opportunities.length > 0 && (
          <Section title="Opportunities">
            <ul className="space-y-2">
              {report.opportunities.map((opp, i) => (
                <li key={i} className="flex items-start">
                  <span className="text-green-600 mr-2">+</span>
                  <span className="text-gray-900">{opp}</span>
                </li>
              ))}
            </ul>
          </Section>
        )}
      </div>

      {/* Talking Points */}
      {report?.talking_points && report.talking_points.length > 0 && (
        <Section title="Recommended Talking Points">
          <ul className="space-y-2">
            {report.talking_points.map((point, i) => (
              <li key={i} className="p-3 bg-blue-50 rounded-lg text-gray-900">
                {point}
              </li>
            ))}
          </ul>
        </Section>
      )}
    </div>
  );
}

function ReportTab({ report }: { report: ResearchReport }) {
  return (
    <div className="space-y-6">
      {/* Company Details */}
      <Section title="Company Details">
        <div className="grid md:grid-cols-2 gap-4">
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
            <div className="p-3 bg-gray-50 rounded-lg text-center">
              <div className="text-sm text-gray-500">Digital Maturity</div>
              <div className="font-medium capitalize">{report.digital_maturity}</div>
            </div>
          )}
          {report.ai_adoption_stage && (
            <div className="p-3 bg-gray-50 rounded-lg text-center">
              <div className="text-sm text-gray-500">AI Adoption</div>
              <div className="font-medium capitalize">{report.ai_adoption_stage}</div>
            </div>
          )}
        </div>
        {report.ai_footprint && (
          <p className="text-gray-900">{report.ai_footprint}</p>
        )}
      </Section>

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
    </div>
  );
}

function CompetitorsTab({ caseStudies }: { caseStudies: CompetitorCaseStudy[] }) {
  return (
    <div className="space-y-4">
      <p className="text-gray-800 mb-4">
        Found {caseStudies.length} relevant competitor case studies
      </p>
      {caseStudies.map((cs, i) => (
        <div key={i} className="border border-gray-200 rounded-lg p-4">
          <div className="flex justify-between items-start mb-2">
            <div>
              <h4 className="font-medium text-gray-900">{cs.competitor_name}</h4>
              <p className="text-sm text-gray-700 capitalize">{cs.vertical?.replace('_', ' ')}</p>
            </div>
            <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs font-medium rounded">
              {Math.round(cs.relevance_score * 100)}% match
            </span>
          </div>
          <h5 className="font-medium text-gray-800 mt-3">{cs.case_study_title}</h5>
          <p className="text-gray-900 text-sm mt-1">{cs.summary}</p>

          {cs.technologies_used && cs.technologies_used.length > 0 && (
            <div className="mt-3">
              <span className="text-xs font-medium text-gray-700">Technologies:</span>
              <div className="flex flex-wrap gap-1 mt-1">
                {cs.technologies_used.map((tech, j) => (
                  <span key={j} className="px-2 py-0.5 bg-gray-100 text-gray-800 text-xs rounded">
                    {tech}
                  </span>
                ))}
              </div>
            </div>
          )}

          {cs.outcomes && cs.outcomes.length > 0 && (
            <div className="mt-3">
              <span className="text-xs font-medium text-gray-700">Outcomes:</span>
              <ul className="mt-1 space-y-1">
                {cs.outcomes.map((outcome, j) => (
                  <li key={j} className="text-sm text-gray-900 flex items-start">
                    <span className="text-green-600 mr-1">+</span>
                    {outcome}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function GapsTab({ gaps }: { gaps: GapAnalysis }) {
  return (
    <div className="space-y-6">
      {/* Confidence Score */}
      <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
        <span className="text-gray-800">Analysis Confidence</span>
        <span className={`font-medium ${
          gaps.confidence_score >= 0.7 ? 'text-green-600' :
          gaps.confidence_score >= 0.4 ? 'text-yellow-600' :
          'text-red-600'
        }`}>
          {Math.round(gaps.confidence_score * 100)}%
        </span>
      </div>

      {/* Priority Areas */}
      {gaps.priority_areas && gaps.priority_areas.length > 0 && (
        <Section title="Priority Areas">
          <div className="space-y-2">
            {gaps.priority_areas.map((area, i) => (
              <div key={i} className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg text-gray-900">
                <span className="font-medium">#{i + 1}</span> {area}
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Gaps Grid */}
      <div className="grid md:grid-cols-3 gap-4">
        {gaps.technology_gaps && gaps.technology_gaps.length > 0 && (
          <GapList title="Technology Gaps" items={gaps.technology_gaps} color="red" />
        )}
        {gaps.capability_gaps && gaps.capability_gaps.length > 0 && (
          <GapList title="Capability Gaps" items={gaps.capability_gaps} color="orange" />
        )}
        {gaps.process_gaps && gaps.process_gaps.length > 0 && (
          <GapList title="Process Gaps" items={gaps.process_gaps} color="purple" />
        )}
      </div>

      {/* Recommendations */}
      {gaps.recommendations && gaps.recommendations.length > 0 && (
        <Section title="Recommendations">
          <ul className="space-y-2">
            {gaps.recommendations.map((rec, i) => (
              <li key={i} className="p-3 bg-green-50 rounded-lg text-gray-900">
                {rec}
              </li>
            ))}
          </ul>
        </Section>
      )}

      {/* Analysis Notes */}
      {gaps.analysis_notes && (
        <Section title="Analysis Notes">
          <p className="text-gray-800 text-sm">{gaps.analysis_notes}</p>
        </Section>
      )}
    </div>
  );
}

function InsideIntelTab({ intel }: { intel: InternalOpsIntel }) {
  return (
    <div className="space-y-6">
      {/* Section 1: Employee Sentiment Overview */}
      {intel.employee_sentiment && (
        <Section title="Employee Sentiment Overview">
          <div className="space-y-4">
            {/* Overall Rating */}
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div className="flex items-center gap-3">
                <span className="text-gray-800 font-medium">Overall Rating</span>
                <RatingStars rating={intel.employee_sentiment.overall_rating} />
              </div>
              <span className="text-lg font-bold text-gray-900">
                {intel.employee_sentiment.overall_rating.toFixed(1)}/5.0
              </span>
            </div>

            {/* Category Ratings */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <RatingCard label="Work-Life Balance" value={intel.employee_sentiment.work_life_balance} />
              <RatingCard label="Compensation" value={intel.employee_sentiment.compensation} />
              <RatingCard label="Culture" value={intel.employee_sentiment.culture} />
              <RatingCard label="Management" value={intel.employee_sentiment.management} />
            </div>

            {/* Recommend & Trend */}
            <div className="grid grid-cols-2 gap-4">
              <div className="p-3 bg-blue-50 rounded-lg text-center">
                <div className="text-2xl font-bold text-blue-600">{intel.employee_sentiment.recommend_pct}%</div>
                <div className="text-sm text-gray-700">Would Recommend</div>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg text-center">
                <SentimentTrend trend={intel.employee_sentiment.trend} />
              </div>
            </div>

            {/* Themes */}
            <div className="grid md:grid-cols-2 gap-4">
              {intel.employee_sentiment.positive_themes.length > 0 && (
                <div>
                  <h5 className="text-sm font-medium text-gray-700 mb-2">Positive Themes</h5>
                  <div className="flex flex-wrap gap-2">
                    {intel.employee_sentiment.positive_themes.map((theme, i) => (
                      <span key={i} className="px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full">
                        {theme}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {intel.employee_sentiment.negative_themes.length > 0 && (
                <div>
                  <h5 className="text-sm font-medium text-gray-700 mb-2">Negative Themes</h5>
                  <div className="flex flex-wrap gap-2">
                    {intel.employee_sentiment.negative_themes.map((theme, i) => (
                      <span key={i} className="px-2 py-1 bg-red-100 text-red-800 text-xs rounded-full">
                        {theme}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </Section>
      )}

      {/* Section 2: Talent & Hiring Intelligence */}
      {intel.job_postings && (
        <Section title="Talent & Hiring Intelligence">
          <div className="space-y-4">
            {/* Total Openings */}
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <span className="text-gray-800 font-medium">Total Open Positions</span>
              <span className="text-2xl font-bold text-gray-900">{intel.job_postings.total_openings}</span>
            </div>

            {/* Departments Hiring */}
            {Object.keys(intel.job_postings.departments_hiring).length > 0 && (
              <div>
                <h5 className="text-sm font-medium text-gray-700 mb-2">Departments Hiring</h5>
                <div className="space-y-2">
                  {Object.entries(intel.job_postings.departments_hiring)
                    .sort(([, a], [, b]) => b - a)
                    .map(([dept, count]) => (
                      <div key={dept} className="flex items-center justify-between">
                        <span className="text-gray-800">{dept}</span>
                        <div className="flex items-center gap-2">
                          <div className="w-32 h-2 bg-gray-200 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-blue-500 rounded-full"
                              style={{
                                width: `${(count / Math.max(...Object.values(intel.job_postings.departments_hiring))) * 100}%`
                              }}
                            />
                          </div>
                          <span className="text-gray-900 font-medium w-8 text-right">{count}</span>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            )}

            {/* Skills Sought */}
            {intel.job_postings.skills_sought.length > 0 && (
              <div>
                <h5 className="text-sm font-medium text-gray-700 mb-2">Key Skills Sought</h5>
                <div className="flex flex-wrap gap-2">
                  {intel.job_postings.skills_sought.map((skill, i) => (
                    <span key={i} className="px-3 py-1 bg-purple-100 text-purple-800 text-sm rounded-full">
                      {skill}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Seniority Distribution */}
            {Object.keys(intel.job_postings.seniority_distribution).length > 0 && (
              <div>
                <h5 className="text-sm font-medium text-gray-700 mb-2">Seniority Distribution</h5>
                <div className="flex gap-3">
                  {Object.entries(intel.job_postings.seniority_distribution).map(([level, count]) => (
                    <div key={level} className="text-center p-2 bg-gray-50 rounded-lg flex-1">
                      <div className="font-bold text-gray-900">{count}</div>
                      <div className="text-xs text-gray-600">{level}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Urgency Signals */}
            {intel.job_postings.urgency_signals.length > 0 && (
              <div className="p-3 bg-orange-50 border border-orange-200 rounded-lg">
                <h5 className="text-sm font-medium text-orange-800 mb-2">Urgency Signals</h5>
                <ul className="space-y-1">
                  {intel.job_postings.urgency_signals.map((signal, i) => (
                    <li key={i} className="text-sm text-orange-900 flex items-start">
                      <span className="mr-2">-</span>
                      {signal}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Hiring Insights */}
            {intel.job_postings.insights && (
              <div className="p-3 bg-blue-50 rounded-lg">
                <p className="text-gray-900 text-sm">{intel.job_postings.insights}</p>
              </div>
            )}
          </div>
        </Section>
      )}

      {/* Section 3: Digital & Social Presence */}
      {intel.linkedin_presence && (
        <Section title="Digital & Social Presence">
          <div className="space-y-4">
            {/* LinkedIn Stats */}
            <div className="grid grid-cols-3 gap-3">
              <div className="p-3 bg-blue-50 rounded-lg text-center">
                <div className="text-xl font-bold text-blue-600">
                  {intel.linkedin_presence.follower_count.toLocaleString()}
                </div>
                <div className="text-xs text-gray-700">LinkedIn Followers</div>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg text-center">
                <div className="text-lg font-medium text-gray-900 capitalize">
                  {intel.linkedin_presence.engagement_level}
                </div>
                <div className="text-xs text-gray-700">Engagement Level</div>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg text-center">
                <EmployeeTrend trend={intel.linkedin_presence.employee_trend} />
              </div>
            </div>

            {/* Recent Posts */}
            {intel.linkedin_presence.recent_posts.length > 0 && (
              <div>
                <h5 className="text-sm font-medium text-gray-700 mb-2">Recent Company Posts</h5>
                <div className="space-y-2">
                  {intel.linkedin_presence.recent_posts.slice(0, 3).map((post, i) => (
                    <div key={i} className="p-3 bg-gray-50 rounded-lg">
                      <div className="font-medium text-gray-900 text-sm">{post.title}</div>
                      <div className="text-xs text-gray-700 mt-1">{post.summary}</div>
                      {post.date && <div className="text-xs text-gray-500 mt-1">{post.date}</div>}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Notable Changes */}
            {intel.linkedin_presence.notable_changes.length > 0 && (
              <div>
                <h5 className="text-sm font-medium text-gray-700 mb-2">Notable Changes</h5>
                <ul className="space-y-1">
                  {intel.linkedin_presence.notable_changes.map((change, i) => (
                    <li key={i} className="text-sm text-gray-900 flex items-start">
                      <span className="text-blue-600 mr-2">-</span>
                      {change}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </Section>
      )}

      {/* Section 4: Public Sentiment Analysis */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Social Media Mentions */}
        {intel.social_media_mentions && intel.social_media_mentions.length > 0 && (
          <Section title="Social Media Mentions">
            <div className="space-y-3">
              {intel.social_media_mentions.map((mention, i) => (
                <div key={i} className="p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium text-gray-900 capitalize">{mention.platform}</span>
                    <SentimentBadge sentiment={mention.sentiment} />
                  </div>
                  <p className="text-sm text-gray-800">{mention.summary}</p>
                  {mention.topic && (
                    <span className="inline-block mt-2 px-2 py-0.5 bg-gray-200 text-gray-700 text-xs rounded">
                      {mention.topic}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </Section>
        )}

        {/* News Sentiment */}
        {intel.news_sentiment && (
          <Section title="News Coverage">
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <span className="text-gray-700 text-sm">Overall Sentiment</span>
                  <SentimentBadge sentiment={intel.news_sentiment.overall_sentiment} className="ml-2" />
                </div>
                <div className="text-right">
                  <span className="text-xs text-gray-500">Coverage: </span>
                  <span className="text-sm font-medium text-gray-900 capitalize">
                    {intel.news_sentiment.coverage_volume}
                  </span>
                </div>
              </div>

              {intel.news_sentiment.topics.length > 0 && (
                <div>
                  <h5 className="text-sm font-medium text-gray-700 mb-2">Key Topics</h5>
                  <div className="flex flex-wrap gap-2">
                    {intel.news_sentiment.topics.map((topic, i) => (
                      <span key={i} className="px-2 py-1 bg-gray-100 text-gray-800 text-xs rounded">
                        {topic}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {intel.news_sentiment.headlines.length > 0 && (
                <div>
                  <h5 className="text-sm font-medium text-gray-700 mb-2">Recent Headlines</h5>
                  <div className="space-y-2">
                    {intel.news_sentiment.headlines.slice(0, 5).map((headline, i) => (
                      <div key={i} className="p-2 bg-gray-50 rounded flex items-start justify-between gap-2">
                        <div>
                          <div className="text-sm text-gray-900">{headline.title}</div>
                          <div className="text-xs text-gray-500">
                            {headline.source} {headline.date && `- ${headline.date}`}
                          </div>
                        </div>
                        <SentimentBadge sentiment={headline.sentiment} size="sm" />
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </Section>
        )}
      </div>

      {/* Section 5: Gap Correlation Insights */}
      {intel.gap_correlations && intel.gap_correlations.length > 0 && (
        <Section title="Gap Correlation Insights">
          <p className="text-sm text-gray-700 mb-4">
            Cross-referencing identified gaps with internal operations evidence
          </p>
          <div className="space-y-3">
            {intel.gap_correlations.map((corr, i) => (
              <GapCorrelationCard key={i} correlation={corr} />
            ))}
          </div>
        </Section>
      )}

      {/* Section 6: Key Insights & Recommendations */}
      {intel.key_insights && intel.key_insights.length > 0 && (
        <Section title="Key Insights & Recommendations">
          <div className="space-y-2">
            {intel.key_insights.map((insight, i) => (
              <div key={i} className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                <p className="text-gray-900">{insight}</p>
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Metadata Footer */}
      <div className="p-3 bg-gray-50 rounded-lg flex items-center justify-between text-sm">
        <div className="flex items-center gap-4">
          <span className="text-gray-600">
            Confidence: <span className={`font-medium ${
              intel.confidence_score >= 0.7 ? 'text-green-600' :
              intel.confidence_score >= 0.4 ? 'text-yellow-600' :
              'text-red-600'
            }`}>{Math.round(intel.confidence_score * 100)}%</span>
          </span>
          {intel.data_freshness && (
            <span className="text-gray-600">
              Data: <span className="text-gray-900">{intel.data_freshness.replace('_', ' ')}</span>
            </span>
          )}
        </div>
      </div>

      {/* Analysis Notes */}
      {intel.analysis_notes && (
        <p className="text-xs text-gray-500 italic">{intel.analysis_notes}</p>
      )}
    </div>
  );
}

function SourcesTab({ sources }: { sources: WebSource[] }) {
  return (
    <div className="space-y-4">
      <p className="text-gray-800 mb-4">
        Research grounded with {sources.length} web source{sources.length !== 1 ? 's' : ''}
      </p>
      <div className="space-y-3">
        {sources.map((source, i) => (
          <div key={i} className="p-3 bg-gray-50 rounded-lg border border-gray-200 hover:border-gray-300 transition-colors">
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center">
                <span className="text-xs text-blue-600 font-medium">{i + 1}</span>
              </div>
              <div className="flex-1 min-w-0">
                <h5 className="font-medium text-gray-900 truncate">
                  {source.title || 'Untitled Source'}
                </h5>
                {source.uri && (
                  <a
                    href={source.uri}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-blue-600 hover:text-blue-800 hover:underline truncate block mt-1"
                  >
                    {source.uri}
                  </a>
                )}
              </div>
              {source.uri && (
                <a
                  href={source.uri}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex-shrink-0 px-3 py-1 text-xs font-medium text-blue-600 bg-blue-50 rounded-full hover:bg-blue-100 transition-colors"
                >
                  Open
                </a>
              )}
            </div>
          </div>
        ))}
      </div>
      <p className="text-xs text-gray-500 mt-4">
        Sources are collected via Google Search grounding to provide real-time, verified information.
      </p>
    </div>
  );
}

function RawTab({ result }: { result: string }) {
  return (
    <pre className="whitespace-pre-wrap text-sm text-gray-800 bg-gray-50 p-4 rounded-lg overflow-auto max-h-96">
      {result}
    </pre>
  );
}

// Helper Components
function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h4 className="font-medium text-gray-900 mb-3">{title}</h4>
      {children}
    </div>
  );
}

function StatCard({ label, value, className = '' }: { label: string; value: string; className?: string }) {
  return (
    <div className="p-3 bg-gray-50 rounded-lg text-center">
      <div className="text-xs text-gray-500">{label}</div>
      <div className={`font-medium text-gray-900 ${className}`}>{value}</div>
    </div>
  );
}

function DetailRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between py-2 border-b border-gray-100">
      <span className="text-gray-700">{label}</span>
      <span className="text-gray-900">{value}</span>
    </div>
  );
}

function GapList({ title, items, color }: { title: string; items: string[]; color: 'red' | 'orange' | 'purple' }) {
  const colorClasses = {
    red: 'bg-red-50 border-red-200',
    orange: 'bg-orange-50 border-orange-200',
    purple: 'bg-purple-50 border-purple-200',
  };

  return (
    <div>
      <h5 className="text-sm font-medium text-gray-700 mb-2">{title}</h5>
      <ul className="space-y-2">
        {items.map((item, i) => (
          <li key={i} className={`p-2 text-sm rounded border ${colorClasses[color]}`}>
            {item}
          </li>
        ))}
      </ul>
    </div>
  );
}

// Inside Intel Helper Components

function RatingStars({ rating }: { rating: number }) {
  const fullStars = Math.floor(rating);
  const hasHalfStar = rating % 1 >= 0.5;
  const emptyStars = 5 - fullStars - (hasHalfStar ? 1 : 0);

  return (
    <div className="flex items-center">
      {[...Array(fullStars)].map((_, i) => (
        <span key={`full-${i}`} className="text-yellow-400">★</span>
      ))}
      {hasHalfStar && <span className="text-yellow-400">★</span>}
      {[...Array(emptyStars)].map((_, i) => (
        <span key={`empty-${i}`} className="text-gray-300">★</span>
      ))}
    </div>
  );
}

function RatingCard({ label, value }: { label: string; value: number }) {
  const getColor = (val: number) => {
    if (val >= 4) return 'text-green-600';
    if (val >= 3) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="p-3 bg-gray-50 rounded-lg text-center">
      <div className={`text-lg font-bold ${getColor(value)}`}>{value.toFixed(1)}</div>
      <div className="text-xs text-gray-600">{label}</div>
    </div>
  );
}

function SentimentTrend({ trend }: { trend: 'improving' | 'declining' | 'stable' }) {
  const config = {
    improving: { icon: '↑', color: 'text-green-600', label: 'Improving' },
    declining: { icon: '↓', color: 'text-red-600', label: 'Declining' },
    stable: { icon: '→', color: 'text-gray-600', label: 'Stable' },
  };
  const { icon, color, label } = config[trend] || config.stable;

  return (
    <div>
      <div className={`text-2xl font-bold ${color}`}>{icon}</div>
      <div className="text-sm text-gray-700">Trend: {label}</div>
    </div>
  );
}

function EmployeeTrend({ trend }: { trend: 'growing' | 'shrinking' | 'stable' }) {
  const config = {
    growing: { icon: '↑', color: 'text-green-600', label: 'Growing' },
    shrinking: { icon: '↓', color: 'text-red-600', label: 'Shrinking' },
    stable: { icon: '→', color: 'text-gray-600', label: 'Stable' },
  };
  const { icon, color, label } = config[trend] || config.stable;

  return (
    <div>
      <div className={`text-xl font-bold ${color}`}>{icon}</div>
      <div className="text-xs text-gray-700">Employee Trend: {label}</div>
    </div>
  );
}

function SentimentBadge({
  sentiment,
  className = '',
  size = 'md'
}: {
  sentiment: 'positive' | 'negative' | 'neutral' | 'mixed' | string;
  className?: string;
  size?: 'sm' | 'md';
}) {
  const config: Record<string, { bg: string; text: string }> = {
    positive: { bg: 'bg-green-100', text: 'text-green-800' },
    negative: { bg: 'bg-red-100', text: 'text-red-800' },
    neutral: { bg: 'bg-gray-100', text: 'text-gray-800' },
    mixed: { bg: 'bg-yellow-100', text: 'text-yellow-800' },
  };
  const { bg, text } = config[sentiment] || config.neutral;
  const sizeClass = size === 'sm' ? 'px-1.5 py-0.5 text-xs' : 'px-2 py-1 text-xs';

  return (
    <span className={`${sizeClass} ${bg} ${text} rounded capitalize ${className}`}>
      {sentiment}
    </span>
  );
}

function GapCorrelationCard({ correlation }: { correlation: GapCorrelation }) {
  const typeColors = {
    technology: 'border-l-red-400',
    capability: 'border-l-orange-400',
    process: 'border-l-purple-400',
  };
  const evidenceColors = {
    supporting: 'bg-green-50 text-green-800',
    contradicting: 'bg-red-50 text-red-800',
    neutral: 'bg-gray-50 text-gray-800',
  };

  return (
    <div className={`p-3 bg-white border border-gray-200 rounded-lg border-l-4 ${typeColors[correlation.gap_type] || 'border-l-gray-400'}`}>
      <div className="flex items-start justify-between mb-2">
        <div>
          <span className="text-xs font-medium text-gray-500 uppercase">{correlation.gap_type} gap</span>
          <p className="text-sm font-medium text-gray-900 mt-0.5">{correlation.description}</p>
        </div>
        <div className="flex items-center gap-2">
          <span className={`px-2 py-0.5 text-xs rounded ${evidenceColors[correlation.evidence_type]}`}>
            {correlation.evidence_type}
          </span>
          <span className={`text-xs font-medium ${
            correlation.confidence >= 0.7 ? 'text-green-600' :
            correlation.confidence >= 0.4 ? 'text-yellow-600' :
            'text-gray-500'
          }`}>
            {Math.round(correlation.confidence * 100)}%
          </span>
        </div>
      </div>
      <div className="text-sm text-gray-700 mb-2">
        <span className="font-medium">Evidence:</span> {correlation.evidence}
      </div>
      {correlation.sales_implication && (
        <div className="text-sm text-blue-700 bg-blue-50 p-2 rounded">
          <span className="font-medium">Sales Implication:</span> {correlation.sales_implication}
        </div>
      )}
    </div>
  );
}
