// REP-CONFIG-002. Implements IA-NAV-001 through IA-NAV-007 exactly.
// No taxonomy, format or legacy label may be added here (IA-NAV-007).

import type { Language } from "./site";

export interface NavItem {
  key: string;
  label: Record<Language, string>;
  route: Record<Language, string>;
}

export const primaryNavigation: NavItem[] = [
  { key: "home", label: { en: "Home", es: "Home" }, route: { en: "/", es: "/" } },
  { key: "daas-platform", label: { en: "DaaS Platform", es: "DaaS Platform" }, route: { en: "/daas-platform/", es: "/daas-platform/" } },
  { key: "impact-products", label: { en: "Impact Products", es: "Impact Products" }, route: { en: "/impact-products/", es: "/impact-products/" } },
  { key: "teaching", label: { en: "Teaching", es: "Teaching" }, route: { en: "/teaching/", es: "/teaching/" } },
  { key: "about", label: { en: "About Me", es: "About Me" }, route: { en: "/about/", es: "/about/" } }
];

export const utilityNavigation: NavItem[] = [
  { key: "cv", label: { en: "CV", es: "CV" }, route: { en: "/cv/", es: "/cv/" } },
  { key: "search", label: { en: "Search", es: "Search" }, route: { en: "/search/", es: "/search/" } }
];

export interface FooterLink {
  key: string;
  label: Record<Language, string>;
  route: Record<Language, string>;
}

export const footerRequiredLinks: FooterLink[] = [
  { key: "contact", label: { en: "Contact", es: "Contact" }, route: { en: "/contact/", es: "/contact/" } },
  { key: "resources", label: { en: "Resources", es: "Resources" }, route: { en: "/resources/", es: "/resources/" } },
  { key: "licenses", label: { en: "Licenses", es: "Licenses" }, route: { en: "/licenses/", es: "/licenses/" } },
  { key: "accessibility", label: { en: "Accessibility", es: "Accessibility" }, route: { en: "/accessibility/", es: "/accessibility/" } },
  { key: "sitemap", label: { en: "Sitemap", es: "Sitemap" }, route: { en: "/sitemap-index.xml", es: "/sitemap-index.xml" } }
];

// IA-NAV-007: forbidden as primary entries. Kept here so lint scripts and
// the header component can assert against a single source of truth.
export const forbiddenPrimaryEntries = [
  "work",
  "research",
  "notes",
  "data",
  "economics",
  "finance",
  "valuation",
  "quantitative-finance",
  "projects",
  "blog"
];
