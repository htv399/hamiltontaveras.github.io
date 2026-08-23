// REP-TEST-003 / QA-VIS-003. Screenshot baselines for Home, Research,
// Project and Week at the required viewports. First run creates the
// baseline; subsequent runs diff against it.
import { test, expect } from "@playwright/test";
import { withBase } from "../helpers/basePath";

const pages = [
  { name: "home", route: "/" },
  { name: "research", route: "/research/" },
  { name: "project-detail", route: "/work/demo-project/" },
  { name: "week-detail", route: "/teaching/econometrics-i-demo/week-01-demo/" }
];
const viewports = [
  { name: "1440", width: 1440, height: 1000 },
  { name: "768", width: 768, height: 1024 },
  { name: "360", width: 360, height: 800 }
];

for (const p of pages) {
  for (const vp of viewports) {
    test(`visual: ${p.name} @ ${vp.name}`, async ({ page }) => {
      await page.setViewportSize({ width: vp.width, height: vp.height });
      await page.goto(withBase(p.route));
      await expect(page).toHaveScreenshot(`${p.name}-${vp.name}.png`, { fullPage: true, maxDiffPixelRatio: 0.02 });
    });
  }
}
