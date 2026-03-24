'use client';

import { useState, useEffect } from 'react';
import { api } from '@/lib/api';
import { ClientProfile, SalesPlay, MemoryEntry } from '@/types';

type Tab = 'profiles' | 'plays' | 'entries';

const ENTRY_TYPE_LABELS: Record<string, string> = {
  research_insight: 'Research Insight',
  client_interaction: 'Client Interaction',
  deal_outcome: 'Deal Outcome',
  best_practice: 'Best Practice',
  lesson_learned: 'Lesson Learned',
};

const PLAY_TYPE_LABELS: Record<string, string> = {
  pitch: 'Pitch',
  objection_handler: 'Objection Handler',
  value_proposition: 'Value Proposition',
  case_study: 'Case Study',
  competitive_response: 'Competitive Response',
  discovery_question: 'Discovery Question',
};

function EmptyState({ message }: { message: string }) {
  return (
    <div className="text-center py-16 text-gray-500">
      <svg className="mx-auto h-10 w-10 text-gray-300 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
      <p className="text-sm">{message}</p>
    </div>
  );
}

function ProfilesTab() {
  const [profiles, setProfiles] = useState<ClientProfile[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');

  useEffect(() => {
    api.listProfiles().then(setProfiles).finally(() => setLoading(false));
  }, []);

  const filtered = profiles.filter(p =>
    p.client_name.toLowerCase().includes(search.toLowerCase()) ||
    p.industry.toLowerCase().includes(search.toLowerCase())
  );

  if (loading) {
    return <div className="text-center py-12 text-gray-400 text-sm">Loading profiles...</div>;
  }

  return (
    <div className="space-y-4">
      <input
        type="text"
        placeholder="Search by company or industry..."
        value={search}
        onChange={e => setSearch(e.target.value)}
        className="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
      />
      {filtered.length === 0 ? (
        <EmptyState message={profiles.length === 0 ? 'No client profiles captured yet. Run a research job and capture to memory to build your knowledge base.' : 'No profiles match your search.'} />
      ) : (
        <div className="grid gap-4 sm:grid-cols-2">
          {filtered.map(profile => (
            <div key={profile.id} className="border border-gray-200 rounded-lg p-4 space-y-2 hover:border-gray-300 transition-colors">
              <div className="flex items-start justify-between gap-2">
                <h3 className="font-medium text-gray-900">{profile.client_name}</h3>
                {profile.industry && (
                  <span className="flex-shrink-0 px-2 py-0.5 bg-blue-50 text-blue-700 text-xs rounded-full">{profile.industry}</span>
                )}
              </div>
              {profile.company_size && (
                <p className="text-xs text-gray-500">Size: {profile.company_size}</p>
              )}
              {profile.region && (
                <p className="text-xs text-gray-500">Region: {profile.region}</p>
              )}
              {profile.summary && (
                <p className="text-sm text-gray-700 line-clamp-3">{profile.summary}</p>
              )}
              <p className="text-xs text-gray-400">
                Updated {new Date(profile.updated_at).toLocaleDateString()}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function PlaysTab() {
  const [plays, setPlays] = useState<SalesPlay[]>([]);
  const [loading, setLoading] = useState(true);
  const [typeFilter, setTypeFilter] = useState<string>('');

  useEffect(() => {
    api.listPlays().then(setPlays).finally(() => setLoading(false));
  }, []);

  const filtered = typeFilter ? plays.filter(p => p.play_type === typeFilter) : plays;
  const types = Array.from(new Set(plays.map(p => p.play_type)));

  if (loading) {
    return <div className="text-center py-12 text-gray-400 text-sm">Loading plays...</div>;
  }

  return (
    <div className="space-y-4">
      {types.length > 0 && (
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setTypeFilter('')}
            className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${!typeFilter ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
          >
            All
          </button>
          {types.map(t => (
            <button
              key={t}
              onClick={() => setTypeFilter(t)}
              className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${typeFilter === t ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
            >
              {PLAY_TYPE_LABELS[t] ?? t}
            </button>
          ))}
        </div>
      )}

      {filtered.length === 0 ? (
        <EmptyState message={plays.length === 0 ? 'No sales plays stored yet. Refine use cases to generate plays and add them to the library.' : 'No plays match the selected filter.'} />
      ) : (
        <div className="space-y-3">
          {filtered.map(play => (
            <div key={play.id} className="border border-gray-200 rounded-lg p-4 space-y-2 hover:border-gray-300 transition-colors">
              <div className="flex items-start justify-between gap-2">
                <h3 className="font-medium text-gray-900">{play.title}</h3>
                <span className="flex-shrink-0 px-2 py-0.5 bg-purple-50 text-purple-700 text-xs rounded-full">
                  {PLAY_TYPE_LABELS[play.play_type] ?? play.play_type}
                </span>
              </div>
              {play.context && (
                <p className="text-xs text-gray-500 italic">{play.context}</p>
              )}
              <p className="text-sm text-gray-700 line-clamp-3">{play.content}</p>
              <div className="flex items-center gap-4 text-xs text-gray-400">
                {play.vertical && <span>Vertical: {play.vertical}</span>}
                <span>Used {play.usage_count}×</span>
                {play.success_rate > 0 && <span>Success: {Math.round(play.success_rate * 100)}%</span>}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function EntriesTab() {
  const [entries, setEntries] = useState<MemoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [typeFilter, setTypeFilter] = useState<string>('');

  useEffect(() => {
    api.listEntries().then(setEntries).finally(() => setLoading(false));
  }, []);

  const types = Array.from(new Set(entries.map(e => e.entry_type)));

  const filtered = entries.filter(e => {
    const matchesType = !typeFilter || e.entry_type === typeFilter;
    const matchesSearch = !search ||
      e.title.toLowerCase().includes(search.toLowerCase()) ||
      e.content.toLowerCase().includes(search.toLowerCase()) ||
      e.client_name.toLowerCase().includes(search.toLowerCase());
    return matchesType && matchesSearch;
  });

  if (loading) {
    return <div className="text-center py-12 text-gray-400 text-sm">Loading entries...</div>;
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row gap-2">
        <input
          type="text"
          placeholder="Search entries..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="flex-1 px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      {types.length > 0 && (
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setTypeFilter('')}
            className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${!typeFilter ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
          >
            All
          </button>
          {types.map(t => (
            <button
              key={t}
              onClick={() => setTypeFilter(t)}
              className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${typeFilter === t ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
            >
              {ENTRY_TYPE_LABELS[t] ?? t}
            </button>
          ))}
        </div>
      )}

      {filtered.length === 0 ? (
        <EmptyState message={entries.length === 0 ? 'No memory entries yet. Complete a research job and use "Capture to Memory" to store insights.' : 'No entries match your search.'} />
      ) : (
        <div className="space-y-3">
          {filtered.map(entry => (
            <div key={entry.id} className="border border-gray-200 rounded-lg p-4 space-y-2 hover:border-gray-300 transition-colors">
              <div className="flex items-start justify-between gap-2">
                <h3 className="font-medium text-gray-900 text-sm">{entry.title}</h3>
                <span className="flex-shrink-0 px-2 py-0.5 bg-gray-100 text-gray-600 text-xs rounded-full">
                  {ENTRY_TYPE_LABELS[entry.entry_type] ?? entry.entry_type}
                </span>
              </div>
              {entry.client_name && (
                <p className="text-xs text-blue-600 font-medium">{entry.client_name}</p>
              )}
              <p className="text-sm text-gray-700 line-clamp-4">{entry.content}</p>
              {entry.tags.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {entry.tags.map((tag, i) => (
                    <span key={i} className="px-2 py-0.5 bg-gray-50 text-gray-500 text-xs rounded border border-gray-200">
                      {tag}
                    </span>
                  ))}
                </div>
              )}
              <p className="text-xs text-gray-400">
                {new Date(entry.created_at).toLocaleDateString()}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const TABS: { id: Tab; label: string }[] = [
  { id: 'profiles', label: 'Client Profiles' },
  { id: 'plays', label: 'Sales Play Library' },
  { id: 'entries', label: 'Memory Entries' },
];

export default function MemoryPage() {
  const [activeTab, setActiveTab] = useState<Tab>('profiles');

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900">Memory / Knowledge Base</h2>
        <p className="text-gray-500 mt-1 text-sm">
          Captured insights, client profiles, and reusable sales plays from past research.
        </p>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200 mb-6">
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors ${
              activeTab === tab.id
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === 'profiles' && <ProfilesTab />}
      {activeTab === 'plays' && <PlaysTab />}
      {activeTab === 'entries' && <EntriesTab />}
    </div>
  );
}
