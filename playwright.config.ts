// TECH-CI-001 e2e runner. Serves the static dist/ build (astro preview)
// and runs against it, matching TECH-001's "no permanent Node server" rule
// — this server only exists for the duration of the test run.
import { defineConfig, devices } from "@playwright/test";

const configuredBase = process.env.BASE_PATH?.replace(/\/$/, "") ?? "";
const previewOrigin = "http://localhost:4321";
const previewHealthUrl = `${previewOrigin}${configuredBase}/`;

export default defineConfig({
  testDir: "./tests",
  timeout: 30_000,
  fullyParallel: true,
  // Constrained local sandboxes can crash multiple concurrent Chromium
  // workers; CI runners with more resources may raise this.
  workers: process.env.CI ? undefined : 2,
  reporter: [["list"]],
  use: {
    baseURL: previewOrigin,
    trace: "retain-on-failure"
  },
  webServer: {
    command: "npx astro preview --port 4321",
    url: previewHealthUrl,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } }
  ]
});
