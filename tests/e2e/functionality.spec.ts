// QA-FUNC-001 / QA-FUNC-002. Search returns results and announces a count;
// filters narrow a listing, reflect state in the URL and survive reload.
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

test("Work filters narrow the listing and survive reload (QA-FUNC-001)", async ({ page }) => {
  await page.goto(withBase("/work/"));
  const list = page.locator(".content-filters__results li");
  const before = await list.count();
  expect(before).toBeGreaterThan(0);

  await page.getByRole("group", { name: "Kind" }).getByRole("checkbox", { name: "valuation" }).check();
  await expect(list).toHaveCount(1);
  await expect(page).toHaveURL(/kind=valuation/);

  await page.reload();
  await expect(list).toHaveCount(1);

  await page.getByRole("button", { name: /clear/i }).click();
  await expect(list).toHaveCount(before);
});
