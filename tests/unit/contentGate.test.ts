// REP-TEST-001 / QA-CM-002 / SEED-003. Production must exclude
// placeholder content and everything private; preview surfaces demo
// fixtures.
import { describe, expect, it } from "vitest";
import { isEligible } from "../../src/lib/contentGate";

describe("isEligible", () => {
  it("hides private content in both environments", () => {
    expect(isEligible({ status: "published", access: "private" }, "preview")).toBe(false);
    expect(isEligible({ status: "published", access: "private" }, "production")).toBe(false);
  });

  it("hides placeholder content only in production", () => {
    const item = { status: "draft", access: "public", placeholder: true };
    expect(isEligible(item, "preview")).toBe(true);
    expect(isEligible(item, "production")).toBe(false);
  });

  it("requires a published-family status in production", () => {
    expect(isEligible({ status: "draft", access: "public" }, "production")).toBe(false);
    expect(isEligible({ status: "published", access: "public" }, "production")).toBe(true);
  });
});
