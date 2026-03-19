import { WebSource } from '@/types';

/**
 * Pre-processes markdown content: converts [N] citation markers to
 * markdown links [[N]](citation:N) for the custom ReactMarkdown link renderer.
 * Only converts N within range 1..sources.length; leaves others as-is.
 */
export function preprocessCitations(content: string, sources: WebSource[]): string {
  if (!sources.length) return content;
  return content.replace(/\[(\d+)\]/g, (match, n) => {
    const idx = parseInt(n, 10) - 1;
    if (idx >= 0 && idx < sources.length) {
      return `[${n}](citation:${n})`;
    }
    return match;
  });
}
