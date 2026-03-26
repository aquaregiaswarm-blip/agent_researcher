import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: '.',
  testMatch: ['production-smoke.spec.ts', 'bugfix-verification.spec.ts', 'ssl-redirect-fix.spec.ts', 'infra-fix-verification.spec.ts'],
  timeout: 60_000,
  retries: 1,
  reporter: [
    ['list'],
    ['html', { outputFolder: 'test-results/prod-report', open: 'never' }],
  ],
  use: {
    baseURL: 'https://agent-researcher-frontend-841327020312.us-east1.run.app',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'off',
    actionTimeout: 30_000,
    navigationTimeout: 30_000,
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  // No webServer — targeting live production
});
