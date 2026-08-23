// REP-TEST-001 / QA-IA-005. Related work scoring must be deterministic,
// exclude the current item, and never expose private content.
import { describe, expect, it } from "vitest";
import { relatedWork, type RelatableItem } from "../../src/lib/relatedWork";

function item(overrides: Partial<RelatableItem>): RelatableItem {
  return {
    id: "ITEM",
    href: "/x/",
    title: "Title",
    summary: "Summary",
    domains: ["economics"],
    language: "en",
    access: "public",
    status: "published",
    ...overrides
  };
}

describe("relatedWork", () => {
  it("excludes the current item", () => {
    const current = item({ id: "A" });
    const pool = [item({ id: "A" }), item({ id: "B" })];
    const result = relatedWork(current, pool, "preview");
    expect(result.every((r) => r.id !== "A")).toBe(true);
  });

  it("excludes private content", () => {
    const current = item({ id: "A" });
    const pool = [item({ id: "B", access: "private" })];
    const result = relatedWork(current, pool, "preview");
    expect(result).toHaveLength(0);
  });

  it("excludes production placeholders in production", () => {
    const current = item({ id: "A" });
    const pool = [item({ id: "B", placeholder: true })];
    expect(relatedWork(current, pool, "production")).toHaveLength(0);
    expect(relatedWork(current, pool, "preview")).toHaveLength(1);
  });

  it("ranks explicit related_ids above domain matches, deterministically", () => {
    const current = item({ id: "A", domains: ["economics"], relatedIds: ["C"] });
    const pool = [
      item({ id: "B", domains: ["economics"] }),
      item({ id: "C", domains: ["valuation"] })
    ];
    const result = relatedWork(current, pool, "preview");
    expect(result[0].id).toBe("C");
  });

  it("caps results at four items", () => {
    const current = item({ id: "A", domains: ["economics"] });
    const pool = Array.from({ length: 6 }, (_, i) => item({ id: `P${i}`, domains: ["economics"] }));
    expect(relatedWork(current, pool, "preview")).toHaveLength(4);
  });
});
