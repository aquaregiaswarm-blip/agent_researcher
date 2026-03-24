import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import PromptEditor from '@/components/PromptEditor';

const mockGetDefaultPrompt = vi.fn();
const mockUpdateDefaultPrompt = vi.fn();

vi.mock('@/lib/api', () => ({
  api: {
    getDefaultPrompt: () => mockGetDefaultPrompt(),
    updateDefaultPrompt: (v: string) => mockUpdateDefaultPrompt(v),
  },
}));

beforeEach(() => {
  mockGetDefaultPrompt.mockResolvedValue({ content: 'Default prompt text' });
  mockUpdateDefaultPrompt.mockResolvedValue({});
});

describe('PromptEditor', () => {
  it('renders the "Research Prompt" toggle button', () => {
    render(<PromptEditor value="hello" onChange={() => {}} />);
    expect(screen.getByText('Research Prompt')).toBeTruthy();
  });

  it('does not show textarea when collapsed', () => {
    render(<PromptEditor value="hello" onChange={() => {}} />);
    expect(screen.queryByRole('textbox')).toBeNull();
  });

  it('shows textarea after clicking the toggle button', async () => {
    render(<PromptEditor value="My prompt" onChange={() => {}} />);
    fireEvent.click(screen.getByText('Research Prompt'));
    await waitFor(() => {
      expect(screen.getByRole('textbox')).toBeTruthy();
    });
  });

  it('displays current value in textarea when expanded', async () => {
    render(<PromptEditor value="My custom prompt" onChange={() => {}} />);
    fireEvent.click(screen.getByText('Research Prompt'));
    await waitFor(() => {
      const textarea = screen.getByRole('textbox') as HTMLTextAreaElement;
      expect(textarea.value).toBe('My custom prompt');
    });
  });

  it('calls onChange when textarea is edited', async () => {
    const onChange = vi.fn();
    render(<PromptEditor value="Initial" onChange={onChange} />);
    fireEvent.click(screen.getByText('Research Prompt'));
    await waitFor(() => screen.getByRole('textbox'));
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Updated prompt' } });
    expect(onChange).toHaveBeenCalledWith('Updated prompt');
  });

  it('shows "Save as Default" button when expanded', async () => {
    render(<PromptEditor value="hello" onChange={() => {}} />);
    fireEvent.click(screen.getByText('Research Prompt'));
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Save as Default' })).toBeTruthy();
    });
  });

  it('calls updateDefaultPrompt and shows success message on save', async () => {
    render(<PromptEditor value="My prompt" onChange={() => {}} />);
    fireEvent.click(screen.getByText('Research Prompt'));
    await waitFor(() => screen.getByRole('button', { name: 'Save as Default' }));
    fireEvent.click(screen.getByRole('button', { name: 'Save as Default' }));
    await waitFor(() => {
      expect(mockUpdateDefaultPrompt).toHaveBeenCalledWith('My prompt');
      expect(screen.getByText('Saved as default!')).toBeTruthy();
    });
  });

  it('loads default prompt via API when value is empty', async () => {
    const onChange = vi.fn();
    render(<PromptEditor value="" onChange={onChange} />);
    await waitFor(() => {
      expect(mockGetDefaultPrompt).toHaveBeenCalled();
      expect(onChange).toHaveBeenCalledWith('Default prompt text');
    });
  });
});
