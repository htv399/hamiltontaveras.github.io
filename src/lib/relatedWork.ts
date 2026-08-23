// IA-REL-001. Deterministic metadata scoring for "related work", shared by
// every detail page that renders CMP-RELATED-WORK. Not a manifest entry of
// its own: it is implementation plumbing behind CMP-RELATED-WORK and
// REP-CMP-005.
export interface RelatableItem {
  id: string;
  href: string;
  title: string;
  summary: string;
  domains: string[];
  methods?: string[];
  objects?: string[];
  language: "en" | "es";
  access: "public" | "summary-only" | "private";
  status: string;
  placeholder?: boolean;
  relatedIds?: string[];
}

const SCORE = { sameDomain: 3, sameMethod: 2, sameObject: 2, explicitRelated: 5, sameLanguage: 1 };

export function relatedWork(current: RelatableItem, pool: RelatableItem[], siteEnv: "preview" | "production"): RelatableItem[] {
  const eligible = pool.filter((item) => {
    if (item.id === current.id) return false;
    if (item.access === "private") return false;
    if (item.status === "draft") return false;
    if (siteEnv === "production" && item.placeholder) return false;
    return true;
  });

  const scored = eligible
    .map((item) => {
      let score = 0;
      if (current.relatedIds?.includes(item.id)) score += SCORE.explicitRelated;
      if (item.domains.some((d) => current.domains.includes(d))) score += SCORE.sameDomain;
      if (item.methods?.some((m) => current.methods?.includes(m))) score += SCORE.sameMethod;
      if (item.objects?.some((o) => current.objects?.includes(o))) score += SCORE.sameObject;
      if (item.language === current.language) score += SCORE.sameLanguage;
      return { item, score };
    })
    .filter((entry) => entry.score > 0)
    .sort((a, b) => b.score - a.score || a.item.id.localeCompare(b.item.id));

  return scored.slice(0, 4).map((entry) => entry.item);
}
