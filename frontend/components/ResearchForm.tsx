'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useForm } from 'react-hook-form';
import { api } from '@/lib/api';
import { ResearchFormData, ResearchJob } from '@/types';
import ResearchResults from './ResearchResults';

export default function ResearchForm() {
  const router = useRouter();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [currentJob, setCurrentJob] = useState<ResearchJob | null>(null);
  const [error, setError] = useState<string | null>(null);

  const { register, handleSubmit, formState: { errors } } = useForm<ResearchFormData>();

  const onSubmit = async (data: ResearchFormData) => {
    setIsSubmitting(true);
    setError(null);
    setCurrentJob(null);

    try {
      const job = await api.createResearch(data);

      // Redirect to the results page immediately
      router.push(`/research/${job.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="space-y-8">
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
        <div>
          <label htmlFor="client_name" className="block text-sm font-medium text-gray-700 mb-1">
            Client Name <span className="text-red-500">*</span>
          </label>
          <input
            id="client_name"
            type="text"
            {...register('client_name', { required: 'Client name is required' })}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            placeholder="Enter client or company name"
          />
          {errors.client_name && (
            <p className="mt-1 text-sm text-red-500">{errors.client_name.message}</p>
          )}
        </div>

        <div>
          <label htmlFor="sales_history" className="block text-sm font-medium text-gray-700 mb-1">
            Past Sales History
          </label>
          <textarea
            id="sales_history"
            {...register('sales_history')}
            rows={4}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            placeholder="Enter any relevant past sales history, interactions, or notes about this client..."
          />
        </div>

        <button
          type="submit"
          disabled={isSubmitting}
          className="w-full py-3 px-4 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed transition-colors"
        >
          {isSubmitting ? 'Running Research...' : 'Run Research'}
        </button>
      </form>

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-700">{error}</p>
        </div>
      )}

      {currentJob && <ResearchResults job={currentJob} />}
    </div>
  );
}
