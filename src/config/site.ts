// REP-CONFIG-001. Central, typed home for every approval_required decision
// named in 00-README.md. Nothing here is invented: unresolved decisions use
// an explicitly labeled placeholder and are gated out of production by
// scripts/check-placeholders.mjs.

export type Language = "en" | "es";
export type SiteEnv = "preview" | "production";
export type ContactMode = "pending" | "email" | "form" | "email_and_form";

export interface Decision<T> {
  /** Decision id from 00-README.md, e.g. "APP-001". */
  id: string;
  /** true once Hamilton Taveras has approved a final value. */
  resolved: boolean;
  value: T;
}

export const SITE_ENV: SiteEnv = (import.meta.env.SITE_ENV as SiteEnv) || "preview";

/** APP-001: default language. Preview placeholder only. */
export const defaultLanguage: Decision<Language> = {
  id: "APP-001",
  resolved: false,
  value: (import.meta.env.DEFAULT_LANGUAGE as Language) || "es"
};

/** APP-002: professional descriptor. PRD-006 records the recommended text. */
export const descriptor: Decision<string> = {
  id: "APP-002",
  resolved: false,
  value: "Economist working across data systems, quantitative analysis, and valuation."
};

/** APP-003: canonical domain. Preview uses a non-resolving placeholder host. */
export const canonicalBase: Decision<string> = {
  id: "APP-003",
  resolved: false,
  value: import.meta.env.SITE_URL || "https://example-preview.invalid"
};

/** APP-004: public signature and professional email. */
export const publicSignature: Decision<{ name: string; email: string | null }> = {
  id: "APP-004",
  resolved: false,
  value: { name: "Hamilton Taveras", email: null }
};

/** APP-005: contact channel. */
export const contactMode: Decision<ContactMode> = {
  id: "APP-005",
  resolved: false,
  value: (import.meta.env.CONTACT_MODE as ContactMode) || "pending"
};

/** APP-006: license policy per content family. See src/config/licenses.ts. */
export const licensesResolved = false;

/** APP-007: About photography. No photo ships until approved. */
export const aboutPhotoApproved = false;

/** APP-008: real initial inventory and editorial hierarchy. */
export const realInventoryApproved = false;

/** APP-009: privacy-respecting analytics. Disabled by default and until approved. */
export const analyticsMode: Decision<"off" | "on"> = {
  id: "APP-009",
  resolved: false,
  value: (import.meta.env.ANALYTICS_MODE as "off" | "on") || "off"
};

/** Faithful Spanish rendering of descriptor.value, used only for <title>
 * and meta description on Spanish pages (never auto-translated editorial
 * content — this is UI/meta chrome, same rule as src/i18n/ui.ts). */
export const descriptorEs =
  "Economista que trabaja entre sistemas de datos, análisis cuantitativo y valoración.";

export const siteThesis =
  "Information, Models, Judgment, and Value: a working method for turning data into economic and financial evidence.";

export const siteConfig = {
  env: SITE_ENV,
  name: publicSignature.value.name,
  descriptor: descriptor.value,
  thesis: siteThesis,
  defaultLanguage: defaultLanguage.value,
  canonicalBase: canonicalBase.value,
  contactMode: contactMode.value,
  analyticsEnabled: analyticsMode.value === "on"
};

/** Decisions that must be resolved before a production build may ship. */
export const criticalProductionDecisions = [
  defaultLanguage,
  descriptor,
  canonicalBase,
  publicSignature,
  contactMode
];

export function unresolvedCriticalDecisions(): Decision<unknown>[] {
  return criticalProductionDecisions.filter((d) => !d.resolved);
}
