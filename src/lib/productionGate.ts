// Enforces the "empty_state" clauses in 05-page-contracts.yml that say
// "Production is blocked until X" / "Production must fail" for About, CV
// and Contact. Astro build throws at render time when SITE_ENV=production
// and the gated decision is still unresolved, which fails `astro build`
// exactly the way TECH-CI-001's `build` step is expected to fail.
export function assertNotBlockedInProduction(resolved: boolean, pageName: string, decisionId: string) {
  const siteEnv = import.meta.env.SITE_ENV || "preview";
  if (siteEnv === "production" && !resolved) {
    throw new Error(
      `[production-gate] ${pageName} cannot ship to production: ${decisionId} is still unresolved. ` +
        `Resolve the decision in src/config/site.ts (or the relevant config) before building for production.`
    );
  }
}
