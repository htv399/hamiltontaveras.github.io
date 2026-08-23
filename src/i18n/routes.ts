// REP-CONFIG-006. IA-REL-003: hreflang and the language switch appear only
// between real equivalents. required_equivalent_pages always have both
// routes; selective_pages only pair up when a translation_key match exists
// in content (checked at render time by the pages themselves).

export const requiredEquivalentPages = ["home", "work", "about", "contact"] as const;
export const selectivePages = [
  "research",
  "notes",
  "project",
  "course",
  "week",
  "material",
  "resources"
] as const;

export const staticRoutePairs: Record<string, { en: string; es: string }> = {
  home: { en: "/", es: "/es/" },
  work: { en: "/work/", es: "/es/work/" },
  research: { en: "/research/", es: "/es/research/" },
  notes: { en: "/notes/", es: "/es/notes/" },
  teaching: { en: "/teaching/", es: "/es/teaching/" },
  about: { en: "/about/", es: "/es/about/" },
  cv: { en: "/cv/", es: "/es/cv/" },
  contact: { en: "/contact/", es: "/es/contact/" },
  resources: { en: "/resources/", es: "/es/resources/" },
  search: { en: "/search/", es: "/es/search/" },
  licenses: { en: "/licenses/", es: "/es/licenses/" },
  accessibility: { en: "/accessibility/", es: "/es/accessibility/" }
};

/** Prepend /es to a dynamic route only for es-language or translated pieces. */
export function localizeDynamicRoute(basePath: string, language: "en" | "es"): string {
  if (language === "es") return `/es${basePath}`;
  return basePath;
}
