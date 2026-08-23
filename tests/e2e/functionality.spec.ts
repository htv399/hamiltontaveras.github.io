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
  await expect(page.locator(".platform-intro .text-body-l")).toHaveText(/improve productivity.*ETL, cloud and AI engineering\./);
  await expect(page.getByText(/microdata.*selected international data/i)).toBeVisible();
  await expect(page.getByRole("heading", { name: "Data engineering" })).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "Data catalog" })).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "Dashboards and monitors" })).toHaveCount(0);
});

test("Home keeps the professional identity on two typographic levels", async ({ page }) => {
  await page.goto(withBase("/"));
  const descriptor = page.locator(".platform-hero__descriptor");
  const metadata = page.locator(".platform-hero__professional-meta");
  await expect(descriptor).toBeVisible();
  await expect(metadata).toBeVisible();
  const styles = await page.evaluate(() => {
    const descriptor = document.querySelector<HTMLElement>(".platform-hero__descriptor")!;
    const metadata = document.querySelector<HTMLElement>(".platform-hero__professional-meta")!;
    return {
      descriptorSize: Number.parseFloat(getComputedStyle(descriptor).fontSize),
      metadataSize: Number.parseFloat(getComputedStyle(metadata).fontSize),
      metadataStyle: getComputedStyle(metadata).fontStyle,
      descriptorBottom: descriptor.getBoundingClientRect().bottom,
      metadataTop: metadata.getBoundingClientRect().top
    };
  });
  expect(styles.descriptorSize).toBeGreaterThan(styles.metadataSize);
  expect(styles.metadataStyle).toBe("italic");
  expect(styles.metadataTop).toBeGreaterThanOrEqual(styles.descriptorBottom);
});

test("Impact Products restores the restrained gold rule", async ({ page }) => {
  await page.goto(withBase("/impact-products/"));
  const rule = await page.locator(".concept .text-body-l").evaluate((element) => {
    const style = getComputedStyle(element, "::after");
    return { display: style.display, width: Number.parseFloat(style.width), border: style.borderTopStyle };
  });
  expect(rule).toEqual({ display: "block", width: 96, border: "solid" });
});

test("Teaching exposes the approved Econometrics I course", async ({ page }) => {
  await page.goto(withBase("/teaching/"));
  await expect(page.getByRole("link", { name: /Econometrics I/i })).toBeVisible();
});

test("Econometrics I landing lists both published classes", async ({ page }) => {
  await page.goto(withBase("/teaching/econometrics-i/"));
  await expect(page.getByRole("link", { name: /Read class: Introducción a la econometría/i })).toBeVisible();
  await expect(page.getByRole("link", { name: /Read class: Probabilidad aplicada/i })).toBeVisible();
});

test("legacy Econometrics I section links continue at the Week 01 route", async ({ page }) => {
  await page.goto(withBase("/teaching/econometrics-i/#qué-hace-la-econometría"));
  await expect(page).toHaveURL(/\/teaching\/econometrics-i\/week-01\//);
  expect(await page.evaluate(() => decodeURIComponent(location.hash))).toBe("#qué-hace-la-econometría");
});

for (const route of ["/teaching/econometrics-i/week-01/", "/teaching/econometrics-i/week-02/"]) {
  test(`${route} renders source content, mathematics and responsive navigation`, async ({ page }) => {
    await page.goto(withBase(route));
    const title = route.endsWith("week-02/") ? /Probabilidad aplicada/i : /Introducción a la econometría/i;
    await expect(page.getByRole("heading", { level: 2, name: title })).toBeVisible();
    await expect(page.locator(".katex-display").first()).toBeVisible();
    await expect(page.locator(".toc-desktop nav")).toBeVisible();
    await expect(page.locator(".academic-highlight").first()).toBeVisible();
    await page.setViewportSize({ width: 480, height: 900 });
    await expect(page.locator(".toc-desktop")).toBeHidden();
    await expect(page.locator(".toc-mobile")).toBeVisible();
  });
}

test("Week 02 boxed equations retain their full horizontal layout", async ({ page }) => {
  await page.goto(withBase("/teaching/econometrics-i/week-02/"));
  const widths = await page.locator(".katex .fbox").evaluateAll((boxes) =>
    boxes.map((box) => box.getBoundingClientRect().width)
  );
  expect(widths.length).toBeGreaterThan(0);
  expect(Math.min(...widths)).toBeGreaterThan(20);
  await expect(page.locator(".academic-content ul")).toHaveCount(2);
  await expect(page.locator(".academic-highlight")).toHaveCount(2);
});

test("CV is structured HTML with download, print and responsive index", async ({ page }) => {
  await page.goto(withBase("/cv/"));
  await expect(page.getByRole("heading", { level: 2, name: "Professional experience" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Download PDF" })).toHaveAttribute("href", /hamilton-taveras-cv\.pdf$/);
  await page.emulateMedia({ media: "print" });
  await expect(page.getByRole("button", { name: "Print" })).toBeHidden();
  await expect(page.locator(".cv-content")).toBeVisible();
});
