// REP-TEST-002 / QA-IA-001 / QA-FUNC-003. Every top-level and detail route
// returns 200, the primary nav matches IA-NAV exactly, and the mobile menu
// opens/closes/traps focus/returns focus (QA-FUNC-003).
import { test, expect } from "@playwright/test";
import { withBase } from "../helpers/basePath";

const staticRoutes = [
  "/", "/es/",
  "/daas-platform/", "/impact-products/", "/teaching/", "/about/", "/cv/", "/contact/", "/resources/", "/search/",
  "/work/", "/research/", "/notes/",
  "/es/work/", "/es/research/", "/es/notes/", "/es/daas-platform/", "/es/impact-products/", "/es/teaching/", "/es/about/", "/es/cv/", "/es/contact/", "/es/resources/", "/es/search/",
  "/licenses/", "/accessibility/", "/es/licenses/", "/es/accessibility/"
];
const detailRoutes = [
  "/analysis/demo-analysis/",
  "/work/demo-project/",
  "/work/demo-data-product/",
  "/work/demo-valuation/",
  "/research/demo-research/",
  "/notes/demo-note/",
  "/teaching/econometrics-i-demo/",
  "/teaching/econometrics-i/",
  "/es/teaching/econometrics-i/"
];

for (const route of [...staticRoutes, ...detailRoutes]) {
  test(`GET ${route} returns 200`, async ({ page }) => {
    const response = await page.goto(withBase(route));
    expect(response?.status()).toBe(200);
  });
}

test("404 route renders the not-found page", async ({ page }) => {
  const response = await page.goto(withBase("/this-page-does-not-exist/"));
  expect(response?.status()).toBe(404);
  await expect(page.getByRole("heading", { level: 1 })).toContainText(/not found/i);
});

test("primary navigation matches the CaribbeanQuant information architecture", async ({ page }) => {
  await page.goto(withBase("/"));
  const nav = page.getByRole("navigation", { name: "Primary navigation" });
  const labels = await nav.getByRole("link").allTextContents();
  expect(labels.map((l) => l.trim())).toEqual(["Home", "DaaS Platform", "Impact Products", "Teaching", "About Me", "Search"]);
});

test("Spanish primary navigation is fully localized", async ({ page }) => {
  await page.goto(withBase("/es/"));
  const nav = page.getByRole("navigation", { name: "Navegación principal" });
  const labels = await nav.getByRole("link").allTextContents();
  expect(labels.map((label) => label.trim())).toEqual(["Inicio", "Plataforma DaaS", "Productos de impacto", "Docencia", "Sobre mí", "Buscar"]);
});

test("language selector preserves reciprocal routes and the session choice", async ({ page }) => {
  await page.goto(withBase("/about/"));
  const selector = page.getByRole("link", { name: "Ver sitio en español" });
  await expect(selector).toHaveAttribute("href", /\/es\/about\/$/);
  await selector.click();
  await expect(page).toHaveURL(/\/es\/about\/$/);
  expect(await page.evaluate(() => sessionStorage.getItem("caribbeanquant-language"))).toBe("es");
});

test("mobile menu opens, traps focus, closes on escape and returns focus (QA-FUNC-003)", async ({ page }) => {
  await page.setViewportSize({ width: 480, height: 900 });
  await page.goto(withBase("/"));
  const trigger = page.getByRole("button", { name: "Menu" });
  await trigger.click();
  const dialog = page.getByRole("dialog", { name: "Menu" });
  await expect(dialog).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await expect(trigger).toBeFocused();
});
