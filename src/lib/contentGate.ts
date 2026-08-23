// Shared "is this item allowed to appear" gate, used by every page that
// lists content. Implements SEED-001.production_policy and CM-VAL-001/002:
// preview may show draft/demo fixtures (explicitly labeled), production
// only shows verified published content.
export type SiteEnv = "preview" | "production";

interface Gateable {
  status: string;
  access: string;
  placeholder?: boolean;
}

export function isEligible(item: Gateable, siteEnv: SiteEnv): boolean {
  if (item.access === "private") return false;
  if (siteEnv === "production") {
    if (item.placeholder) return false;
    return ["published", "working", "ongoing", "archived"].includes(item.status);
  }
  // preview: everything except private is visible so seed fixtures render.
  return true;
}
