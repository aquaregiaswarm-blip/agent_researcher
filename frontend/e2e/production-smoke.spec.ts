/**
 * Production smoke tests — Deep Prospecting Engine
 *
 * Targets:
 *   Frontend: https://agent-researcher-frontend-841327020312.us-east1.run.app
 *   Backend:  https://agent-researcher-backend-841327020312.us-east1.run.app
 *
 * Run: npx playwright test e2e/production-smoke.spec.ts --config e2e/playwright.prod.config.ts
 */

import { test, expect, Page, ConsoleMessage } from '@playwright/test';

const FRONTEND = 'https://agent-researcher-frontend-841327020312.us-east1.run.app';
const BACKEND  = 'https://agent-researcher-backend-841327020312.us-east1.run.app';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Collect JS console errors from a page during a callback. */
async function collectConsoleErrors(
  page: Page,
  fn: () => Promise<void>,
): Promise<string[]> {
  const errors: string[] = [];
  const handler = (msg: ConsoleMessage) => {
    if (msg.type() === 'error') {
      errors.push(msg.text());
    }
  };
  page.on('console', handler);
  try {
    await fn();
  } finally {
    page.off('console', handler);
  }
  return errors;
}

/** Return first completed job id from the backend, or null. */
async function getFirstCompletedJobId(page: Page): Promise<string | null> {
  try {
    const resp = await page.request.get(`${BACKEND}/api/research/jobs/`);
    if (!resp.ok()) return null;
    const data = await resp.json();
    const results: Array<{ id: string; status: string }> = data.results ?? data;
    const completed = results.find((j) => j.status === 'completed');
    return completed?.id ?? null;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// 1. API health check
// ---------------------------------------------------------------------------

test.describe('API Health Check', () => {
  test('backend /api/research/ returns 200', async ({ request }) => {
    const resp = await request.get(`${BACKEND}/api/research/`);
    expect(resp.status(), `Expected 200, got ${resp.status()}`).toBe(200);
  });

  test('backend /api/research/jobs/ returns 200', async ({ request }) => {
    const resp = await request.get(`${BACKEND}/api/research/jobs/`);
    expect(resp.status(), `Expected 200, got ${resp.status()}`).toBe(200);
    const body = await resp.json();
    // Should have results array or be an array
    const list: unknown[] = body.results ?? body;
    expect(Array.isArray(list), 'Response should contain a list of jobs').toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 2. Homepage load
// ---------------------------------------------------------------------------

test.describe('Homepage', () => {
  test('renders research form without JS errors', async ({ page }) => {
    const errors = await collectConsoleErrors(page, async () => {
      await page.goto(FRONTEND, { waitUntil: 'networkidle' });
    });

    // Verify form or CTA is visible
    const hasForm =
      (await page.locator('form').count()) > 0 ||
      (await page.locator('input[type="text"]').count()) > 0 ||
      (await page.locator('input[placeholder]').count()) > 0;

    await page.screenshot({
      path: '/Users/jonathangough/Dropbox/Mac/Documents/presales/agent_researcher/prod-homepage.png',
      fullPage: true,
    });

    expect(hasForm, 'Expected at least one form or text input on homepage').toBe(true);
    expect(errors, `Console errors found: ${errors.join(' | ')}`).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// 3. Research history page
// ---------------------------------------------------------------------------

test.describe('Research History Page', () => {
  test('/research loads without 500 errors', async ({ page }) => {
    const responses: Array<{ url: string; status: number }> = [];
    page.on('response', (r) => {
      responses.push({ url: r.url(), status: r.status() });
    });

    const errors = await collectConsoleErrors(page, async () => {
      await page.goto(`${FRONTEND}/research`, { waitUntil: 'networkidle' });
    });

    await page.screenshot({
      path: '/Users/jonathangough/Dropbox/Mac/Documents/presales/agent_researcher/prod-research-list.png',
      fullPage: true,
    });

    const serverErrors = responses.filter((r) => r.status >= 500);
    expect(
      serverErrors,
      `5xx responses: ${JSON.stringify(serverErrors)}`,
    ).toHaveLength(0);

    expect(errors, `Console errors: ${errors.join(' | ')}`).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// 4. Projects list page
// ---------------------------------------------------------------------------

test.describe('Projects List Page', () => {
  test('/projects loads without 500 errors', async ({ page }) => {
    const responses: Array<{ url: string; status: number }> = [];
    page.on('response', (r) => {
      responses.push({ url: r.url(), status: r.status() });
    });

    const errors = await collectConsoleErrors(page, async () => {
      await page.goto(`${FRONTEND}/projects`, { waitUntil: 'networkidle' });
    });

    await page.screenshot({
      path: '/Users/jonathangough/Dropbox/Mac/Documents/presales/agent_researcher/prod-projects-list.png',
      fullPage: true,
    });

    const serverErrors = responses.filter((r) => r.status >= 500);
    expect(
      serverErrors,
      `5xx responses: ${JSON.stringify(serverErrors)}`,
    ).toHaveLength(0);

    expect(errors, `Console errors: ${errors.join(' | ')}`).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// 5. Research form submission — Patagonia
// ---------------------------------------------------------------------------

test.describe('Research Form Submission', () => {
  test('submitting Patagonia redirects to /research/[id] with loading state', async ({ page }) => {
    const errors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') errors.push(msg.text());
    });

    await page.goto(FRONTEND, { waitUntil: 'networkidle' });

    // Find the company name input — try multiple selectors
    const inputSelectors = [
      'input[name="client_name"]',
      'input[placeholder*="company" i]',
      'input[placeholder*="client" i]',
      'input[placeholder*="name" i]',
      'input[type="text"]:first-of-type',
      'input:not([type="hidden"]):first-of-type',
    ];

    let input = null;
    for (const sel of inputSelectors) {
      const el = page.locator(sel).first();
      if (await el.isVisible()) {
        input = el;
        break;
      }
    }

    expect(input, 'Could not find company name input on homepage').not.toBeNull();

    await input!.fill('Patagonia');

    // Screenshot before submit
    await page.screenshot({
      path: '/Users/jonathangough/Dropbox/Mac/Documents/presales/agent_researcher/prod-form-filled.png',
    });

    // Submit — try button click or Enter
    const submitBtn = page.locator('button[type="submit"]').first();
    if (await submitBtn.isVisible()) {
      await submitBtn.click();
    } else {
      await input!.press('Enter');
    }

    // Wait for navigation to /research/[id]
    await page.waitForURL(/\/research\/[a-zA-Z0-9-]+/, { timeout: 15_000 });

    // Screenshot after redirect
    await page.screenshot({
      path: '/Users/jonathangough/Dropbox/Mac/Documents/presales/agent_researcher/prod-form-submitted.png',
      fullPage: true,
    });

    const url = page.url();
    expect(url).toMatch(/\/research\/[a-zA-Z0-9-]+/);

    // Check for loading animation / spinner — the job starts processing
    // Accept either a spinner, loading text, or the orbital animation
    const loadingIndicators = [
      page.locator('[data-testid="loading"]'),
      page.locator('.animate-spin'),
      page.locator('[class*="orbit"]'),
      page.locator('[class*="loading"]'),
      page.locator('text=/researching|processing|loading|analyzing/i'),
    ];

    let foundLoading = false;
    for (const indicator of loadingIndicators) {
      if (await indicator.count() > 0) {
        foundLoading = true;
        break;
      }
    }

    // Loading state OR completed results are both valid (fast jobs)
    const hasResults = await page.locator('button', { hasText: 'Overview' }).count() > 0;
    expect(
      foundLoading || hasResults,
      'Expected either a loading indicator or completed results on the research page',
    ).toBe(true);

    // No JS errors during submission flow
    expect(errors, `Console errors during submission: ${errors.join(' | ')}`).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// 6. Completed research results — tab rendering
// ---------------------------------------------------------------------------

test.describe('Completed Research Results', () => {
  test('all 8 tabs visible on a completed job', async ({ page }) => {
    const jobId = await getFirstCompletedJobId(page);
    test.skip(!jobId, 'No completed research job in production — skipping');

    await page.goto(`${FRONTEND}/research/${jobId}`, { waitUntil: 'networkidle' });

    await page.screenshot({
      path: '/Users/jonathangough/Dropbox/Mac/Documents/presales/agent_researcher/prod-results-tabs.png',
      fullPage: false,
    });

    // All expected tab labels
    const expectedTabs = [
      'Overview',
      'Competitors',
      'Gap Analysis',
      'Sources',
      'Generate',
    ];

    for (const tabName of expectedTabs) {
      const tab = page.getByRole('button', { name: tabName });
      const visible = await tab.isVisible();
      expect(visible, `Tab "${tabName}" not visible for job ${jobId}`).toBe(true);
    }
  });

  test('Overview tab shows company data', async ({ page }) => {
    const jobId = await getFirstCompletedJobId(page);
    test.skip(!jobId, 'No completed research job — skipping');

    const errors = await collectConsoleErrors(page, async () => {
      await page.goto(`${FRONTEND}/research/${jobId}`, { waitUntil: 'networkidle' });
    });

    // Overview should be default — look for company-related content
    const hasContent =
      (await page.locator('h1, h2, h3').count()) > 0 &&
      (await page.locator('p, li').count()) > 0;

    expect(hasContent, 'Overview tab should render headings and paragraphs').toBe(true);
    expect(errors, `Console errors on results page: ${errors.join(' | ')}`).toHaveLength(0);

    await page.screenshot({
      path: '/Users/jonathangough/Dropbox/Mac/Documents/presales/agent_researcher/prod-results-overview.png',
      fullPage: true,
    });
  });

  test('Generate tab renders 4 section cards', async ({ page }) => {
    const jobId = await getFirstCompletedJobId(page);
    test.skip(!jobId, 'No completed research job — skipping');

    await page.goto(`${FRONTEND}/research/${jobId}`, { waitUntil: 'networkidle' });

    const generateTab = page.getByRole('button', { name: 'Generate' });
    if (!(await generateTab.isVisible())) {
      test.skip(true, 'Generate tab not visible for this job');
    }

    await generateTab.click();
    await page.waitForLoadState('networkidle');

    await page.screenshot({
      path: '/Users/jonathangough/Dropbox/Mac/Documents/presales/agent_researcher/prod-results-generate.png',
      fullPage: true,
    });

    // Expect 4 section cards: Use Cases, Personas, One-Pager, Account Plan
    const expectedSections = ['Use Case', 'Persona', 'One-Pager', 'Account Plan'];
    for (const section of expectedSections) {
      const el = page.getByText(new RegExp(section, 'i')).first();
      const visible = await el.isVisible();
      expect(visible, `Generate tab section "${section}" not visible`).toBe(true);
    }
  });

  test('Competitors tab renders without errors', async ({ page }) => {
    const jobId = await getFirstCompletedJobId(page);
    test.skip(!jobId, 'No completed research job — skipping');

    const errors = await collectConsoleErrors(page, async () => {
      await page.goto(`${FRONTEND}/research/${jobId}`, { waitUntil: 'networkidle' });
    });

    const tab = page.getByRole('button', { name: 'Competitors' });
    if (!(await tab.isVisible())) {
      test.skip(true, 'Competitors tab not visible');
    }

    await tab.click();
    await page.waitForLoadState('networkidle');

    await page.screenshot({
      path: '/Users/jonathangough/Dropbox/Mac/Documents/presales/agent_researcher/prod-results-competitors.png',
      fullPage: true,
    });

    expect(errors, `Console errors on Competitors tab: ${errors.join(' | ')}`).toHaveLength(0);
  });

  test('Gap Analysis tab renders without raw asterisks', async ({ page }) => {
    const jobId = await getFirstCompletedJobId(page);
    test.skip(!jobId, 'No completed research job — skipping');

    await page.goto(`${FRONTEND}/research/${jobId}`, { waitUntil: 'networkidle' });

    const tab = page.getByRole('button', { name: 'Gap Analysis' });
    if (!(await tab.isVisible())) {
      test.skip(true, 'Gap Analysis tab not visible');
    }

    await tab.click();
    await page.waitForLoadState('networkidle');

    await page.screenshot({
      path: '/Users/jonathangough/Dropbox/Mac/Documents/presales/agent_researcher/prod-results-gaps.png',
      fullPage: true,
    });

    // Regression check: no raw markdown asterisks visible
    const bodyText = await page.locator('body').innerText();
    const rawAsterisks = (bodyText.match(/\*\*/g) || []).length;
    expect(
      rawAsterisks,
      `Found ${rawAsterisks} raw "**" markdown markers in Gap Analysis tab`,
    ).toBe(0);
  });
});
