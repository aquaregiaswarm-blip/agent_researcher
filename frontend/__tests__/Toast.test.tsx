import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { ToastContainer } from '@/components/ui/Toast';
import { Toast } from '@/lib/toast';

vi.mock('@/lib/api', () => ({ api: {} }));

const makeToast = (overrides: Partial<Toast> = {}): Toast => ({
  id: 't1',
  type: 'success',
  message: 'Saved successfully!',
  duration: 0, // persistent so it doesn't auto-dismiss in tests
  ...overrides,
});

describe('ToastContainer', () => {
  it('renders nothing when toasts array is empty', () => {
    const { container } = render(<ToastContainer toasts={[]} onRemove={() => {}} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders a toast message', async () => {
    await act(async () => {
      render(<ToastContainer toasts={[makeToast()]} onRemove={() => {}} />);
    });
    expect(screen.getByText('Saved successfully!')).toBeTruthy();
  });

  it('renders error toast', async () => {
    await act(async () => {
      render(<ToastContainer toasts={[makeToast({ type: 'error', message: 'Something went wrong' })]} onRemove={() => {}} />);
    });
    expect(screen.getByText('Something went wrong')).toBeTruthy();
  });

  it('renders info toast', async () => {
    await act(async () => {
      render(<ToastContainer toasts={[makeToast({ type: 'info', message: 'Research started' })]} onRemove={() => {}} />);
    });
    expect(screen.getByText('Research started')).toBeTruthy();
  });

  it('calls onRemove when dismiss button is clicked', async () => {
    const onRemove = vi.fn();
    await act(async () => {
      render(<ToastContainer toasts={[makeToast({ id: 'toast-123' })]} onRemove={onRemove} />);
    });
    fireEvent.click(screen.getByLabelText('Dismiss notification'));
    expect(onRemove).toHaveBeenCalledWith('toast-123');
  });

  it('renders multiple toasts', async () => {
    const toasts = [
      makeToast({ id: 't1', message: 'First message' }),
      makeToast({ id: 't2', message: 'Second message', type: 'error' }),
    ];
    await act(async () => {
      render(<ToastContainer toasts={toasts} onRemove={() => {}} />);
    });
    expect(screen.getByText('First message')).toBeTruthy();
    expect(screen.getByText('Second message')).toBeTruthy();
  });

  it('renders action button when toast has action', async () => {
    const action = { label: 'Undo', onClick: vi.fn() };
    await act(async () => {
      render(
        <ToastContainer
          toasts={[makeToast({ action })]}
          onRemove={() => {}}
        />
      );
    });
    expect(screen.getByRole('button', { name: 'Undo' })).toBeTruthy();
  });

  it('calls action.onClick when action button is clicked', async () => {
    const onClick = vi.fn();
    await act(async () => {
      render(
        <ToastContainer
          toasts={[makeToast({ action: { label: 'Retry', onClick } })]}
          onRemove={() => {}}
        />
      );
    });
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(onClick).toHaveBeenCalled();
  });

  it('has aria-live region for accessibility', async () => {
    await act(async () => {
      render(<ToastContainer toasts={[makeToast()]} onRemove={() => {}} />);
    });
    const region = document.querySelector('[aria-live="polite"]');
    expect(region).toBeTruthy();
  });
});
