// REP-CONFIG-003. Controlled vocabularies from CM-001. content.config.ts
// derives its zod enums from these arrays so the schema and the UI labels
// can never drift apart.

export const languages = ["en", "es"] as const;
export const editorialStatuses = [
  "draft",
  "review",
  "scheduled",
  "published",
  "working",
  "ongoing",
  "archived"
] as const;
export const accessLevels = ["public", "preview", "restricted", "available-on-request", "summary-only", "private"] as const;

export const domains = [
  "data-systems",
  "economics",
  "corporate-finance",
  "valuation",
  "quant-finance",
  "risk",
  "ai"
] as const;

export const methods = [
  "econometrics",
  "time-series",
  "causal-inference",
  "ml",
  "simulation",
  "optimization",
  "financial-modeling",
  "data-engineering",
  "data-governance"
] as const;

export const objects = ["firm", "asset", "sector", "market", "public-policy", "institution"] as const;

export const outputs = [
  "paper",
  "model",
  "data",
  "code",
  "visualization",
  "dashboard",
  "slides",
  "notebook",
  "report",
  "tool"
] as const;

export const materialTypes = [
  "pdf",
  "html",
  "qmd",
  "r",
  "rmd",
  "csv",
  "xlsx",
  "zip",
  "notebook",
  "external"
] as const;

export const domainLabels: Record<(typeof domains)[number], Record<"en" | "es", string>> = {
  "data-systems": { en: "Data Systems", es: "Sistemas de Datos" },
  economics: { en: "Economics", es: "Economía" },
  "corporate-finance": { en: "Corporate Finance", es: "Finanzas Corporativas" },
  valuation: { en: "Valuation", es: "Valoración" },
  "quant-finance": { en: "Quantitative Finance", es: "Finanzas Cuantitativas" },
  risk: { en: "Risk", es: "Riesgo" },
  ai: { en: "AI", es: "IA" }
};
