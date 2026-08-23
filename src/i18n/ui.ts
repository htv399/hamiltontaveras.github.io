// REP-CONFIG-005. Interface strings only. Editorial content is never
// auto-translated (IA-001 bilingual.fallback).

import type { Language } from "../config/site";

export const ui = {
  en: {
    skipToContent: "Skip to content",
    menu: "Menu",
    close: "Close",
    search: "Search",
    searchPlaceholder: "Search the portal",
    searchIdle: "Type to search titles, summaries and content.",
    searchLoading: "Searching…",
    searchEmpty: "No results for this query.",
    searchError: "Search is temporarily unavailable.",
    // Kept as plain strings, not a function: Astro serializes island props
    // to JSON when hydrating a client:* component, and functions cannot
    // cross that boundary (Search.tsx builds the count text itself).
    searchResultLabelOne: "result",
    searchResultLabelMany: "results",
    readMore: "Read more",
    viewAll: "View all",
    download: "Download",
    source: "Source",
    sources: "Sources",
    method: "Method",
    dataVintage: "Data as of",
    published: "Published",
    updated: "Updated",
    version: "Version",
    license: "License",
    access: "Access",
    relatedWork: "Related work",
    breadcrumbHome: "Home",
    previousWeek: "Previous week",
    nextWeek: "Next week",
    courseIndex: "Course index",
    contactPending: "A verified contact channel will appear here once approved.",
    demoNotice: "Demonstration content. Not a real record.",
    notFoundTitle: "Page not found",
    notFoundBody: "The page you requested does not exist or has moved.",
    languageSwitch: "Español",
    filterAll: "All",
    filterClear: "Clear filters",
    noResultsHeading: "No matching items",
    filtersHeading: "Filters",
    disclosure: "Disclosure",
    viewCv: "View CV",
    contact: "Contact"
  },
  es: {
    skipToContent: "Saltar al contenido",
    menu: "Menú",
    close: "Cerrar",
    search: "Buscar",
    searchPlaceholder: "Buscar en el portal",
    searchIdle: "Escribe para buscar títulos, resúmenes y contenido.",
    searchLoading: "Buscando…",
    searchEmpty: "Sin resultados para esta búsqueda.",
    searchError: "La búsqueda no está disponible temporalmente.",
    searchResultLabelOne: "resultado",
    searchResultLabelMany: "resultados",
    readMore: "Leer más",
    viewAll: "Ver todo",
    download: "Descargar",
    source: "Fuente",
    sources: "Fuentes",
    method: "Método",
    dataVintage: "Datos al",
    published: "Publicado",
    updated: "Actualizado",
    version: "Versión",
    license: "Licencia",
    access: "Acceso",
    relatedWork: "Trabajo relacionado",
    breadcrumbHome: "Inicio",
    previousWeek: "Semana anterior",
    nextWeek: "Semana siguiente",
    courseIndex: "Índice del curso",
    contactPending: "Un canal de contacto verificado aparecerá aquí una vez aprobado.",
    demoNotice: "Contenido demostrativo. No es un registro real.",
    notFoundTitle: "Página no encontrada",
    notFoundBody: "La página solicitada no existe o fue movida.",
    languageSwitch: "English",
    filterAll: "Todos",
    filterClear: "Limpiar filtros",
    noResultsHeading: "Sin elementos coincidentes",
    filtersHeading: "Filtros",
    disclosure: "Aviso",
    viewCv: "Ver CV",
    contact: "Contacto"
  }
} as const;

export function t(language: Language) {
  return ui[language];
}
