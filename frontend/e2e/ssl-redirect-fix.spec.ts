/**
 * SSL redirect fix verification — Deep Prospecting Engine (Production)
 *
 * Verifies that the SECURE_PROXY_SSL_HEADER patch eliminates the 301
 * redirect loop that Cloud Run was causing. Before the fix, the backend
 * would see every request as HTTP (Cloud Run terminates TLS) and issue
 * a 301 → the browser would loop forever.
 *
 * After the fix, /api/research/jobs/ must return 200 with no redirect
 * when called directly by the browser (frontend proxies via NEXT_PUBLIC_API_URL
 * or the frontend itself makes server-side fetches from the same VPC).
 *
 * Run:
 *   npx playwright test e2e/ssl-redirect-fix.spec.ts \
 *     --config e2e/playwright.prod.config.ts
 */

import { test, expect } from '@playwright/test';

const BACKEND = 'https://agent-researcher-backend-841327020312.us-east1.run.app';
const FRONTEND = 'https://agent-researcher-frontend-841327020312.us-east1.run.app';
const SCREENSHOTS = '/Users/jonathangough/Dropbox/Mac/Documents/presales/agent_researcher';

// ---------------------------------------------------------------------------
// SSL redirect fix: direct browser navigation to backend API endpoint
// ---------------------------------------------------------------------------

test.describe('SSL Redirect Fix — SECURE_PROXY_SSL_HEADER patch', () => {

  test('backend /api/research/jobs/ returns 200 — no 301 redirect loop', async ({ page }) => {
    const redirects: Array<{ url: string; status: number }> = [];
    const responses: Array<{ url: string; status: number }> = [];

    page.on('response', (r) => {
      const status = r.status();
      responses.push({ url: r.url(), status });
      if (status >= 300 && status < 400) {
        redirects.push({ url: r.url(), status });
      }
    });

    // Navigate directly to the backend JSON API — a browser GET with proper headers
    await page.goto(`${BACKEND}/api/research/jobs/`, {
      waitUntil: 'domcontentloaded',
      timeout: 30_000,
    });

    await page.screenshot({
      path: `${SCREENSHOTS}/ssl-fix-backend-jobs.png`,
      fullPage: false,
    });

    // The final URL should still be on the backend (no redirect away)
    const finalUrl = page.url();
    expect(
      finalUrl,
      `Expected to stay on backend URL, ended up at: ${finalUrl}`,
    ).toContain('agent-researcher-backend');

    // Verify the last backend response was 200 (or 401 auth, never 301)
    const backendResponses = responses.filter((r) =>
      r.url.includes('agent-researcher-backend') && r.url.includes('/api/'),
    );

    if (backendResponses.length > 0) {
      const lastStatus = backendResponses[backendResponses.length - 1].status;
      expect(
        lastStatus,
        `Final backend response was ${lastStatus} — expected 200 (not 301 redirect loop). Redirects seen: ${JSON.stringify(redirects)}`,
      ).not.toBe(301);
      expect(
        lastStatus,
        `Final backend response was ${lastStatus} — expected 200 (not any 3xx). All responses: ${JSON.stringify(backendResponses)}`,
      ).toBeLessThan(400);
    }

    // Hard assertion: no redirect loops (should be 0 or at most 1 canonical redirect)
    expect(
      redirects.length,
      `Found ${redirects.length} redirect(s) — potential loop: ${JSON.stringify(redirects)}`,
    ).toBeLessThanOrEqual(1);
  });

  test('frontend /research page fetches jobs list — backend returns non-301', async ({ page }) => {
    let jobsApiStatus: number | null = null;
    let jobsApiUrl = '';
    const redirectsSeen: Array<{ url: string; status: number }> = [];

    page.on('response', (r) => {
      const url = r.url();
      const status = r.status();
      if (url.includes('/api/research')) {
        if (status >= 300 && status < 400) {
          redirectsSeen.push({ url, status });
        }
        if (status === 200) {
          jobsApiStatus = status;
          jobsApiUrl = url;
        }
      }
    });

    await page.goto(`${FRONTEND}/research`, { waitUntil: 'networkidle' });

    await page.screenshot({
      path: `${SCREENSHOTS}/ssl-fix-frontend-research.png`,
      fullPage: true,
    });

    // No 301s on any /api/research call
    expect(
      redirectsSeen,
      `Backend issued 301 redirect(s) on /api/research calls — SSL fix may not be deployed: ${JSON.stringify(redirectsSeen)}`,
    ).toHaveLength(0);

    // The page content should not be an error page
    const title = await page.title();
    expect(title).not.toMatch(/500|error|redirect/i);
  });

  test('research history page shows actual job list entries — not empty', async ({ page }) => {
    await page.goto(`${FRONTEND}/research`, { waitUntil: 'networkidle' });

    // Look for job list items — links to /research/[id]
    const jobLinks = page.locator('a[href*="/research/"]');
    const count = await jobLinks.count();

    // Take screenshot regardless
    await page.screenshot({
      path: `${SCREENSHOTS}/ssl-fix-job-list.png`,
      fullPage: true,
    });

    expect(
      count,
      `Expected at least 1 job in the history list, found ${count}. The research history may be empty or failing to load.`,
    ).toBeGreaterThan(0);
  });
});
