// REP-CONFIG-004. APP-006 (approval_required): license policy per content
// family. No family is resolved yet, so every license reference must use
// one of the explicit placeholder ids below until Hamilton Taveras approves
// a real policy. INTERNAL-DEMO-ONLY is reserved for SEED-001 fixtures and
// is always blocked from production regardless of this file.

export type LicenseFamily = "text" | "code" | "data" | "graphics" | "teaching";

export interface LicensePolicy {
  family: LicenseFamily;
  id: string;
  resolved: boolean;
  label: string;
}

export const APP_006_RESOLVED = false;

export const licensePolicies: LicensePolicy[] = [
  { family: "text", id: "PENDING-TEXT-LICENSE", resolved: false, label: "License pending approval" },
  { family: "code", id: "PENDING-CODE-LICENSE", resolved: false, label: "License pending approval" },
  { family: "data", id: "PENDING-DATA-LICENSE", resolved: false, label: "License pending approval" },
  { family: "graphics", id: "PENDING-GRAPHICS-LICENSE", resolved: false, label: "License pending approval" },
  { family: "teaching", id: "PENDING-TEACHING-LICENSE", resolved: false, label: "License pending approval" }
];

export const DEMO_LICENSE_ID = "INTERNAL-DEMO-ONLY";

export function licenseIsPublishable(licenseId: string): boolean {
  if (licenseId === DEMO_LICENSE_ID) return false;
  return !licensePolicies.some((p) => p.id === licenseId && !p.resolved);
}
