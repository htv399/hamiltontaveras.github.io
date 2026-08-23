// REP-TEST-001 / QA-IA-003. Every published slug must match IA-SLUG-001's
// pattern and be unique within its collection. This walks the real content
// files rather than a synthetic fixture, so it catches a bad slug the
// moment it is authored.
import { describe, expect, it } from "vitest";
import { readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import yaml from "js-yaml";

const SLUG_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const CONTENT_DIR = path.resolve(process.cwd(), "src/content");

function walk(dir: string): string[] {
  if (!statSync(dir, { throwIfNoEntry: false })) return [];
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) return walk(full);
    return full.endsWith(".md") || full.endsWith(".mdx") || full.endsWith(".yml") ? [full] : [];
  });
}

function readSlug(file: string): string | undefined {
  const raw = readFileSync(file, "utf8");
  if (file.endsWith(".yml")) return (yaml.load(raw) as { slug?: string })?.slug;
  const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!match) return undefined;
  return (yaml.load(match[1]) as { slug?: string })?.slug;
}

describe("content slugs", () => {
  const collections = ["analysis", "research", "work", "notes", "courses", "weeks", "materials", "catalog-entries", "dashboards", "impact-products", "video-resources"];

  for (const collection of collections) {
    const files = walk(path.join(CONTENT_DIR, collection));
    const slugs = files.map(readSlug).filter((s): s is string => Boolean(s));

    it(`${collection}: every slug matches IA-SLUG-001's pattern`, () => {
      for (const slug of slugs) expect(slug).toMatch(SLUG_PATTERN);
    });

    it(`${collection}: slugs are unique`, () => {
      expect(new Set(slugs).size).toBe(slugs.length);
    });
  }
});
