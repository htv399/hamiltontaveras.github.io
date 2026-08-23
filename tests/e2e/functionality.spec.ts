// QA-FUNC-001 / QA-FUNC-002. Search returns results and announces a count;
// Work presents the three available fixture kinds without a redundant filter.
import { test, expect } from "@playwright/test";
import { withBase } from "../helpers/basePath";

test("search finds a known page and updates the URL query (QA-FUNC-002)", async ({ page }) => {
  await page.goto(withBase("/search/"));
  await page.getByRole("searchbox").fill("econometrics");
  await expect(page.getByRole("status")).toContainText(/result/i, { timeout: 10_000 });
  await expect(page).toHaveURL(/[?&]q=econometrics/);
  await expect(page.locator(".search__results li").first()).toBeVisible();
});

test("search announces zero results distinctly (QA-FUNC-002)", async ({ page }) => {
  await page.goto(withBase("/search/"));
  await page.getByRole("searchbox").fill("zzzznoresultsxyz");
  await expect(page.getByRole("status")).toContainText(/no results/i, { timeout: 10_000 });
});

test("Work distinguishes the three demonstration formats (QA-FUNC-001)", async ({ page }) => {
  await page.goto(withBase("/work/"));
  await expect(page.locator(".work-split article")).toHaveCount(3);
  await expect(page.locator(".work-visual--flow")).toBeVisible();
  await expect(page.locator(".work-visual--coverage")).toBeVisible();
  await expect(page.locator(".work-visual--sensitivity")).toBeVisible();
  await expect(page.getByRole("link", { name: /Demostración de proyecto/ })).toBeVisible();
  await expect(page.getByRole("link", { name: /Producto de datos demostrativo/ })).toBeVisible();
  await expect(page.getByRole("link", { name: /Valoración demostrativa/ })).toBeVisible();
});
