'use client';

import { useState, useEffect, useCallback } from 'react';
import { UseCase, Persona, OnePager, AccountPlan } from '@/types';
import { api } from '@/lib/api';
import { useToast } from '@/lib/toast';
import UseCaseSection from '../generate/UseCaseSection';
import PersonaSection from '../generate/PersonaSection';
import OnePagerSection from '../generate/OnePagerSection';
import AccountPlanSection from '../generate/AccountPlanSection';

interface GenerateTabProps {
  researchJobId: string;
  clientName: string;
  projectId?: string;
  iterationId?: string;
}

export default function GenerateTab({ researchJobId, clientName, projectId, iterationId }: GenerateTabProps) {
  const { addToast, removeToast } = useToast();

  const [useCases, setUseCases] = useState<UseCase[]>([]);
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [onePager, setOnePager] = useState<OnePager | null>(null);
  const [accountPlan, setAccountPlan] = useState<AccountPlan | null>(null);

  const [generatingUseCases, setGeneratingUseCases] = useState(false);
  const [generatingPersonas, setGeneratingPersonas] = useState(false);
  const [generatingOnePager, setGeneratingOnePager] = useState(false);
  const [generatingAccountPlan, setGeneratingAccountPlan] = useState(false);

  const [loading, setLoading] = useState(true);

  // Load existing assets on mount
  const loadExisting = useCallback(async () => {
    setLoading(true);
    try {
      const [ucs, ps, ops, aps] = await Promise.allSettled([
        api.listUseCases(researchJobId),
        api.listPersonas(researchJobId),
        api.listOnePagers(researchJobId),
        api.listAccountPlans(researchJobId),
      ]);
      if (ucs.status === 'fulfilled') setUseCases(ucs.value);
      if (ps.status === 'fulfilled') setPersonas(ps.value);
      if (ops.status === 'fulfilled' && ops.value.length > 0) setOnePager(ops.value[0]);
      if (aps.status === 'fulfilled' && aps.value.length > 0) setAccountPlan(aps.value[0]);
    } finally {
      setLoading(false);
    }
  }, [researchJobId]);

  useEffect(() => { loadExisting(); }, [loadExisting]);

  const handleGenerateUseCases = async () => {
    setGeneratingUseCases(true);
    const tid = addToast({ type: 'loading', message: 'Generating use cases...' });
    try {
      const result = await api.generateUseCases(researchJobId);
      setUseCases(result);
      removeToast(tid);
      addToast({ type: 'success', message: `${result.length} use case${result.length !== 1 ? 's' : ''} generated` });
    } catch {
      removeToast(tid);
      addToast({ type: 'error', message: 'Failed to generate use cases' });
    } finally {
      setGeneratingUseCases(false);
    }
  };

  const handleGeneratePersonas = async () => {
    setGeneratingPersonas(true);
    const tid = addToast({ type: 'loading', message: 'Building buyer personas...' });
    try {
      const result = await api.generatePersonas(researchJobId);
      setPersonas(result);
      removeToast(tid);
      addToast({ type: 'success', message: `${result.length} persona${result.length !== 1 ? 's' : ''} generated` });
    } catch {
      removeToast(tid);
      addToast({ type: 'error', message: 'Failed to generate personas' });
    } finally {
      setGeneratingPersonas(false);
    }
  };

  const handleGenerateOnePager = async () => {
    setGeneratingOnePager(true);
    const tid = addToast({ type: 'loading', message: 'Creating one-pager...' });
    try {
      const result = await api.generateOnePager(researchJobId);
      setOnePager(result);
      removeToast(tid);
      addToast({ type: 'success', message: 'One-pager ready' });
    } catch {
      removeToast(tid);
      addToast({ type: 'error', message: 'Failed to create one-pager' });
    } finally {
      setGeneratingOnePager(false);
    }
  };

  const handleGenerateAccountPlan = async () => {
    setGeneratingAccountPlan(true);
    const tid = addToast({ type: 'loading', message: 'Building account plan...' });
    try {
      const result = await api.generateAccountPlan(researchJobId);
      setAccountPlan(result);
      removeToast(tid);
      addToast({ type: 'success', message: 'Account plan ready' });
    } catch {
      removeToast(tid);
      addToast({ type: 'error', message: 'Failed to build account plan' });
    } finally {
      setGeneratingAccountPlan(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <svg className="animate-spin h-6 w-6 text-blue-500" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg text-sm text-blue-800">
        Generate sales assets from your research. Each asset builds on the research data above.
      </div>

      <UseCaseSection
        researchJobId={researchJobId}
        clientName={clientName}
        projectId={projectId}
        iterationId={iterationId}
        useCases={useCases}
        generating={generatingUseCases}
        onGenerate={handleGenerateUseCases}
      />

      <div className="border-t border-gray-100" />

      <PersonaSection
        clientName={clientName}
        projectId={projectId}
        iterationId={iterationId}
        personas={personas}
        generating={generatingPersonas}
        onGenerate={handleGeneratePersonas}
      />

      <div className="border-t border-gray-100" />

      <OnePagerSection
        clientName={clientName}
        projectId={projectId}
        iterationId={iterationId}
        onePager={onePager}
        generating={generatingOnePager}
        onGenerate={handleGenerateOnePager}
      />

      <div className="border-t border-gray-100" />

      <AccountPlanSection
        clientName={clientName}
        projectId={projectId}
        iterationId={iterationId}
        accountPlan={accountPlan}
        generating={generatingAccountPlan}
        onGenerate={handleGenerateAccountPlan}
      />
    </div>
  );
}
