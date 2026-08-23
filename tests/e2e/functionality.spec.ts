// QA-FUNC-001 / QA-FUNC-002. Search returns results and announces a count;
// the public platform does not expose private demonstration records.
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

test("DaaS Platform does not fabricate unpublished catalog records", async ({ page }) => {
  await page.goto(withBase("/daas-platform/"));
  await expect(page.getByRole("heading", { level: 1, name: "DaaS Platform" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Data catalog" })).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "Dashboards and monitors" })).toHaveCount(0);
});

test("Teaching exposes the approved Econometrics I course", async ({ page }) => {
  await page.goto(withBase("/teaching/"));
  await expect(page.getByRole("link", { name: /Econometrics I/i })).toBeVisible();
});

test("Econometrics I renders source content, mathematics and responsive navigation", async ({ page }) => {
  await page.goto(withBase("/teaching/econometrics-i/"));
  await expect(page.getByRole("heading", { level: 2, name: /Introducción a la econometría/i })).toBeVisible();
  await expect(page.locator(".katex-display").first()).toBeVisible();
  await expect(page.locator(".toc-desktop nav")).toBeVisible();
  await page.setViewportSize({ width: 480, height: 900 });
  await expect(page.locator(".toc-desktop")).toBeHidden();
  await expect(page.locator(".toc-mobile")).toBeVisible();
});

test("CV is structured HTML with download, print and responsive index", async ({ page }) => {
  await page.goto(withBase("/cv/"));
  await expect(page.getByRole("heading", { level: 2, name: "Professional experience" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Download PDF" })).toHaveAttribute("href", /hamilton-taveras-cv\.pdf$/);
  await page.emulateMedia({ media: "print" });
  await expect(page.getByRole("button", { name: "Print" })).toBeHidden();
  await expect(page.locator(".cv-content")).toBeVisible();
});
