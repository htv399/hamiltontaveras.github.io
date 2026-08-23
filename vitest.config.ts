import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // tests/e2e and tests/visual use @playwright/test's own runner
    // (`npm run test:e2e` / `test:a11y` / `test:responsive` / `test:visual`);
    // vitest only owns tests/unit.
    include: ["tests/unit/**/*.test.ts"]
  }
});
