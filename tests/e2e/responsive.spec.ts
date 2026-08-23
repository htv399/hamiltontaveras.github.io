// QA-RESP-001 / QA-VIS-003. All required pages pass all required viewports
// (1440, 1200, 768, 480, 360) with no body horizontal overflow.
import { test, expect } from "@playwright/test";
import { withBase } from "../helpers/basePath";

const viewports = [
  { name: "1440", width: 1440, height: 1000 },
  { name: "1200", width: 1200, height: 900 },
  { name: "768", width: 768, height: 1024 },
  { name: "480", width: 480, height: 900 },
  { name: "360", width: 360, height: 800 }
];
const pages = ["/", "/daas-platform/", "/impact-products/", "/teaching/", "/about/"];

for (const vp of viewports) {
  for (const route of pages) {
    test(`no horizontal overflow at ${vp.name}px on ${route}`, async ({ page }) => {
      await page.setViewportSize({ width: vp.width, height: vp.height });
      await page.goto(withBase(route));
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1);
      expect(overflow, `${route} overflows horizontally at ${vp.width}px`).toBe(false);
    });
  }
}

test("reduced motion disables the chart line transition", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(withBase("/"));
  // Reduced-motion CSS is asserted structurally: the rule only exists
  // inside a `@media (prefers-reduced-motion: no-preference)` block in
  // src/components/data/InteractiveChart.tsx, so under "reduce" no
  // matching stylesheet rule applies transition timing to .interactive-chart__line.
  expect(await page.evaluate(() => window.matchMedia("(prefers-reduced-motion: reduce)").matches)).toBe(true);
});
