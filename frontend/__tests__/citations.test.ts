import { describe, it, expect } from 'vitest';
import { preprocessCitations } from '@/lib/citations';
import { WebSource } from '@/types';

const sources: WebSource[] = [
  { uri: 'https://source1.com', title: 'Source One' },
  { uri: 'https://source2.com', title: 'Source Two' },
  { uri: 'https://source3.com', title: 'Source Three' },
];

describe('preprocessCitations', () => {
  it('returns content unchanged when no [N] patterns present', () => {
    const content = 'Plain text with no citations.';
    expect(preprocessCitations(content, sources)).toBe(content);
  });

  it('returns content unchanged when sources array is empty', () => {
    const content = 'Text with [1] citation.';
    expect(preprocessCitations(content, [])).toBe(content);
  });

  it('converts [1] to citation link when 1 source exists', () => {
    const content = 'A fact [1] about the company.';
    const result = preprocessCitations(content, [sources[0]]);
    expect(result).toBe('A fact [1](citation:1) about the company.');
  });

  it('converts [1] to citation link when multiple sources exist', () => {
    const content = 'First fact [1].';
    const result = preprocessCitations(content, sources);
    expect(result).toBe('First fact [1](citation:1).');
  });

  it('leaves [5] unchanged when only 3 sources exist', () => {
    const content = 'Out of range [5] citation.';
    const result = preprocessCitations(content, sources);
    expect(result).toBe('Out of range [5] citation.');
  });

  it('converts multiple valid citations in one string', () => {
    const content = 'Fact [1] and another [3].';
    const result = preprocessCitations(content, sources);
    expect(result).toBe('Fact [1](citation:1) and another [3](citation:3).');
  });

  it('converts valid citations and leaves invalid ones unchanged', () => {
    const content = 'Valid [2] and invalid [99].';
    const result = preprocessCitations(content, sources);
    expect(result).toBe('Valid [2](citation:2) and invalid [99].');
  });

  it('converts [0] as invalid (1-indexed, so 0 is out of range)', () => {
    const content = 'Zero-indexed [0] citation.';
    const result = preprocessCitations(content, sources);
    expect(result).toBe('Zero-indexed [0] citation.');
  });

  it('handles citation at the exact boundary (3 sources, [3] valid)', () => {
    const content = 'Boundary [3].';
    const result = preprocessCitations(content, sources);
    expect(result).toBe('Boundary [3](citation:3).');
  });

  it('handles citation just beyond boundary (3 sources, [4] invalid)', () => {
    const content = 'Beyond [4] citation.';
    const result = preprocessCitations(content, sources);
    expect(result).toBe('Beyond [4] citation.');
  });
});
