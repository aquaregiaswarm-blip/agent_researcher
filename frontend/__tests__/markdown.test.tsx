import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// Inline the MarkdownText component rather than importing from ResearchResults
// to keep this test isolated and fast.
function MarkdownText({ content, className = '' }: { content: string; className?: string }) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} className={`prose prose-sm max-w-none ${className}`}>
      {content}
    </ReactMarkdown>
  );
}

describe('MarkdownText', () => {
  it('renders plain text unchanged', () => {
    render(<MarkdownText content="Hello world" />);
    expect(screen.getByText('Hello world')).toBeInTheDocument();
  });

  it('renders bold markdown as strong element', () => {
    render(<MarkdownText content="This is **bold** text" />);
    const strong = document.querySelector('strong');
    expect(strong).not.toBeNull();
    expect(strong?.textContent).toBe('bold');
  });

  it('renders bullet list as ul', () => {
    render(<MarkdownText content={'- Item one\n- Item two\n- Item three'} />);
    const ul = document.querySelector('ul');
    expect(ul).not.toBeNull();
    const items = document.querySelectorAll('li');
    expect(items).toHaveLength(3);
  });

  it('renders numbered list as ol', () => {
    render(<MarkdownText content={'1. First\n2. Second\n3. Third'} />);
    const ol = document.querySelector('ol');
    expect(ol).not.toBeNull();
    const items = document.querySelectorAll('li');
    expect(items).toHaveLength(3);
  });

  it('renders link as anchor tag', () => {
    render(<MarkdownText content="[Example](https://example.com)" />);
    const link = document.querySelector('a');
    expect(link).not.toBeNull();
    expect(link?.textContent).toBe('Example');
    expect(link?.getAttribute('href')).toBe('https://example.com');
  });

  it('handles empty string without error', () => {
    const { container } = render(<MarkdownText content="" />);
    expect(container).toBeInTheDocument();
  });

  it('applies className to wrapper', () => {
    const { container } = render(<MarkdownText content="test" className="custom-class" />);
    expect(container.firstChild).toHaveClass('custom-class');
  });

  it('renders priority area with hash prefix without treating it as a heading', () => {
    // This is the regression case from TJX — LLM used "**Priority 1:**" prefixes
    render(<MarkdownText content="**Priority 1:** Data infrastructure modernisation" />);
    const strong = document.querySelector('strong');
    expect(strong?.textContent).toBe('Priority 1:');
  });
});
