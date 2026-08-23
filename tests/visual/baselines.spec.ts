// REP-TEST-003 / QA-VIS-003. Screenshot baselines for the seven modified
// destinations on desktop and mobile. First run creates the
// baseline; subsequent runs diff against it.
import { test, expect } from "@playwright/test";
import { withBase } from "../helpers/basePath";

const pages = [
  { name: "home", route: "/" },
  { name: "daas-platform", route: "/daas-platform/" },
  { name: "impact-products", route: "/impact-products/" },
  { name: "teaching", route: "/teaching/" },
  { name: "econometrics-i", route: "/teaching/econometrics-i/" },
  { name: "about", route: "/about/" },
  { name: "cv", route: "/cv/" }
];
const viewports = [
  { name: "1440", width: 1440, height: 1000 },
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
