// REP-CONFIG-002. Implements IA-NAV-001 through IA-NAV-007 exactly.
// No taxonomy, format or legacy label may be added here (IA-NAV-007).

import type { Language } from "./site";

export interface NavItem {
  key: string;
  label: Record<Language, string>;
  route: Record<Language, string>;
}

export const primaryNavigation: NavItem[] = [
  { key: "home", label: { en: "Home", es: "Inicio" }, route: { en: "/", es: "/" } },
  { key: "work", label: { en: "Work", es: "Trabajo" }, route: { en: "/work/", es: "/es/work/" } },
  { key: "research", label: { en: "Research", es: "Investigación" }, route: { en: "/research/", es: "/es/research/" } },
  { key: "notes", label: { en: "Notes", es: "Notas" }, route: { en: "/notes/", es: "/es/notes/" } },
  { key: "teaching", label: { en: "Teaching", es: "Docencia" }, route: { en: "/teaching/", es: "/es/teaching/" } },
  { key: "about", label: { en: "About", es: "Perfil" }, route: { en: "/about/", es: "/es/about/" } }
];

export const utilityNavigation: NavItem[] = [
  { key: "cv", label: { en: "CV", es: "CV" }, route: { en: "/cv/", es: "/es/cv/" } },
  { key: "search", label: { en: "Search", es: "Buscar" }, route: { en: "/search/", es: "/es/search/" } }
];

export interface FooterLink {
  key: string;
  label: Record<Language, string>;
  route: Record<Language, string>;
}

export const footerRequiredLinks: FooterLink[] = [
  { key: "contact", label: { en: "Contact", es: "Contacto" }, route: { en: "/contact/", es: "/es/contact/" } },
  { key: "resources", label: { en: "Resources", es: "Recursos" }, route: { en: "/resources/", es: "/es/resources/" } },
  { key: "licenses", label: { en: "Licenses", es: "Licencias" }, route: { en: "/licenses/", es: "/es/licenses/" } },
  { key: "accessibility", label: { en: "Accessibility", es: "Accesibilidad" }, route: { en: "/accessibility/", es: "/es/accessibility/" } },
  { key: "rss", label: { en: "RSS", es: "RSS" }, route: { en: "/research/rss.xml", es: "/research/rss.xml" } },
  { key: "sitemap", label: { en: "Sitemap", es: "Mapa del sitio" }, route: { en: "/sitemap-index.xml", es: "/sitemap-index.xml" } }
];

// IA-NAV-007: forbidden as primary entries. Kept here so lint scripts and
// the header component can assert against a single source of truth.
export const forbiddenPrimaryEntries = [
  "data",
  "economics",
  "finance",
  "valuation",
  "quantitative-finance",
  "projects",
  "blog"
];
