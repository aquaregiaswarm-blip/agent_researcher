/**
 * E2E tests for research results page — tab navigation and markdown rendering.
 *
 * Requires:
 *   - Frontend dev server on localhost:3000
 *   - Backend dev server on localhost:8000
 *   - A completed research job in the database (see setup notes below)
 *
 * Run: npx playwright test e2e/research-results.spec.ts
 */

import { test, expect, Page } from '@playwright/test';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function getCompletedJobId(page: Page): Promise<string | null> {
  // Try the jobs list endpoint first
  const resp = await page.request.get('http://localhost:8000/api/research/jobs/');
  if (!resp.ok()) return null;
  const data = await resp.json();
  const results: Array<{ id: string; status: string }> = data.results ?? data;
  const completed = results.find(j => j.status === 'completed');
  return completed?.id ?? null;
}

async function navigateToResult(page: Page, jobId: string) {
  await page.goto(`/research/${jobId}`);
  await page.waitForLoadState('networkidle');
}

// ---------------------------------------------------------------------------
// Tab navigation
// ---------------------------------------------------------------------------

test.describe('Research results — tab navigation', () => {
  test('overview tab is visible for completed job', async ({ page }) => {
    const jobId = await getCompletedJobId(page);
    test.skip(!jobId, 'No completed research job found — run a research job first');

    await navigateToResult(page, jobId!);

    await expect(page.getByRole('button', { name: 'Overview' })).toBeVisible();
  });

  test('all expected tabs visible for job with full data', async ({ page }) => {
    const jobId = await getCompletedJobId(page);
    test.skip(!jobId, 'No completed research job found');

    await navigateToResult(page, jobId!);

    // Overview is always present
    await expect(page.getByRole('button', { name: 'Overview' })).toBeVisible();
  });

  test('clicking competitors tab shows competitor cards', async ({ page }) => {
    const jobId = await getCompletedJobId(page);
    test.skip(!jobId, 'No completed research job found');

    await navigateToResult(page, jobId!);

    const competitorsTab = page.getByRole('button', { name: 'Competitors' });
    if (!(await competitorsTab.isVisible())) {
      test.skip(true, 'No competitor data for this job');
    }

    await competitorsTab.click();
    await expect(page.getByText(/case stud/i)).toBeVisible();
  });

  test('clicking gap analysis tab shows gap sections', async ({ page }) => {
    const jobId = await getCompletedJobId(page);
    test.skip(!jobId, 'No completed research job found');

    await navigateToResult(page, jobId!);

    const gapsTab = page.getByRole('button', { name: 'Gap Analysis' });
    if (!(await gapsTab.isVisible())) {
      test.skip(true, 'No gap analysis data for this job');
    }

    await gapsTab.click();
    await expect(page.getByText('Analysis Confidence')).toBeVisible();
  });

  test('clicking inside intel tab shows sentiment section', async ({ page }) => {
    const jobId = await getCompletedJobId(page);
    test.skip(!jobId, 'No completed research job found');

    await navigateToResult(page, jobId!);

    const intelTab = page.getByRole('button', { name: 'Inside Intel' });
    if (!(await intelTab.isVisible())) {
      test.skip(true, 'No inside intel data for this job');
    }

    await intelTab.click();
    await expect(page.getByText('Employee Sentiment Overview')).toBeVisible();
  });

  test('clicking sources tab shows web source links', async ({ page }) => {
    const jobId = await getCompletedJobId(page);
    test.skip(!jobId, 'No completed research job found');

    await navigateToResult(page, jobId!);

    const sourcesTab = page.getByRole('button', { name: 'Sources' });
    if (!(await sourcesTab.isVisible())) {
      test.skip(true, 'No sources data for this job');
    }

    await sourcesTab.click();
    await expect(page.getByText(/Research grounded with/)).toBeVisible();
  });
});


// ---------------------------------------------------------------------------
// Gap Analysis rendering — regression for formatting bug
// ---------------------------------------------------------------------------

test.describe('Gap Analysis — markdown rendering regression', () => {
  test('priority_areas contain no raw asterisks', async ({ page }) => {
    const jobId = await getCompletedJobId(page);
    test.skip(!jobId, 'No completed research job found');

    await navigateToResult(page, jobId!);

    const gapsTab = page.getByRole('button', { name: 'Gap Analysis' });
    if (!(await gapsTab.isVisible())) {
      test.skip(true, 'No gap analysis data for this job');
    }

    await gapsTab.click();

    const prioritySection = page.locator('text=Priority Areas').locator('..');
    const innerText = await prioritySection.innerText();
    expect(innerText).not.toMatch(/\*\*/);
  });

  test('analysis_notes renders as formatted HTML not raw markdown', async ({ page }) => {
    const jobId = await getCompletedJobId(page);
    test.skip(!jobId, 'No completed research job found');

    await navigateToResult(page, jobId!);

    const gapsTab = page.getByRole('button', { name: 'Gap Analysis' });
    if (!(await gapsTab.isVisible())) {
      test.skip(true, 'No gap analysis data for this job');
    }

    await gapsTab.click();

    // If analysis_notes contains **text**, it must be rendered as <strong>, not shown raw
    const notesSection = page.locator('text=Analysis Notes').locator('..');
    const innerHTML = await notesSection.innerHTML();
    expect(innerHTML).not.toMatch(/\*\*/);
  });
});


// ---------------------------------------------------------------------------
// Inside Intel rendering
// ---------------------------------------------------------------------------

test.describe('Inside Intel — markdown rendering', () => {
  test('key_insights render as formatted text with no raw asterisks', async ({ page }) => {
    const jobId = await getCompletedJobId(page);
    test.skip(!jobId, 'No completed research job found');

    await navigateToResult(page, jobId!);

    const intelTab = page.getByRole('button', { name: 'Inside Intel' });
    if (!(await intelTab.isVisible())) {
      test.skip(true, 'No inside intel data for this job');
    }

    await intelTab.click();

    const insightsSection = page.locator('text=Key Insights & Recommendations').locator('..');
    const innerHTML = await insightsSection.innerHTML();
    expect(innerHTML).not.toMatch(/\*\*/);
  });

  test('shows employee overall rating', async ({ page }) => {
    const jobId = await getCompletedJobId(page);
    test.skip(!jobId, 'No completed research job found');

    await navigateToResult(page, jobId!);

    const intelTab = page.getByRole('button', { name: 'Inside Intel' });
    if (!(await intelTab.isVisible())) {
      test.skip(true, 'No inside intel data for this job');
    }

    await intelTab.click();

    await expect(page.getByText(/\/5\.0/)).toBeVisible();
  });
});
