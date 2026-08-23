import type { Language } from "../config/site";

export const staticRoutePairs: Record<string, { en: string; es: string }> = {
  home: { en: "/", es: "/es/" },
  daas: { en: "/daas-platform/", es: "/es/daas-platform/" },
  impact: { en: "/impact-products/", es: "/es/impact-products/" },
  teaching: { en: "/teaching/", es: "/es/teaching/" },
  econometrics: { en: "/teaching/econometrics-i/", es: "/es/teaching/econometrics-i/" },
  about: { en: "/about/", es: "/es/about/" },
  cv: { en: "/cv/", es: "/es/cv/" },
  contact: { en: "/contact/", es: "/es/contact/" },
  resources: { en: "/resources/", es: "/es/resources/" },
  search: { en: "/search/", es: "/es/search/" },
  licenses: { en: "/licenses/", es: "/es/licenses/" },
  accessibility: { en: "/accessibility/", es: "/es/accessibility/" }
};

export function languageAlternative(path: string, language: Language): string {
  const pair = Object.values(staticRoutePairs).find((candidate) => candidate[language] === path);
  if (!pair) return language === "en" ? "/es/" : "/";
  return language === "en" ? pair.es : pair.en;
}

export function reciprocalLanguageAlternative(path: string, language: Language): string | undefined {
  const pair = Object.values(staticRoutePairs).find((candidate) => candidate[language] === path);
  if (!pair) return undefined;
  return language === "en" ? pair.es : pair.en;
}

export function localizeDynamicRoute(basePath: string, language: Language): string {
  return language === "es" ? `/es${basePath}` : basePath;
}
