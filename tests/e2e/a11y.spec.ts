// QA-A11Y-001. Zero serious/critical automated violations on every page
// template (Home, a listing, a detail page, a utility page).
import { test, expect } from "@playwright/test";
import { withBase } from "../helpers/basePath";
import { AxeBuilder } from "@axe-core/playwright";

const templates = [
  "/", "/daas-platform/", "/impact-products/", "/teaching/", "/teaching/econometrics-i/", "/teaching/econometrics-i/week-01/", "/teaching/econometrics-i/week-02/",
  "/work/", "/research/", "/notes/",
  "/about/", "/cv/", "/contact/", "/resources/", "/search/",
  "/es/", "/es/daas-platform/", "/es/impact-products/", "/es/teaching/", "/es/teaching/econometrics-i/", "/es/teaching/econometrics-i/week-01/", "/es/teaching/econometrics-i/week-02/", "/es/about/", "/es/cv/",
  "/404-check-page-that-does-not-exist/"
];

for (const route of templates) {
  test(`a11y: ${route}`, async ({ page }) => {
    await page.goto(withBase(route));
    const results = await new AxeBuilder({ page }).analyze();
    const serious = results.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
    expect(serious, JSON.stringify(serious, null, 2)).toEqual([]);
  });
}
