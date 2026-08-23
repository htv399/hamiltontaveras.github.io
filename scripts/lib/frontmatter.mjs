// Tiny frontmatter reader shared by the content/placeholder/materials
// scripts. Avoids adding gray-matter as a dependency since js-yaml already
// covers the one thing we need: parsing the block between --- fences.
import { readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import yaml from "js-yaml";

export function walkFiles(dir, predicate) {
  const results = [];
  if (!statSync(dir, { throwIfNoEntry: false })) return results;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) results.push(...walkFiles(full, predicate));
    else if (predicate(full)) results.push(full);
  }
  return results;
}

export function readFrontmatter(filePath) {
  const raw = readFileSync(filePath, "utf8");
  if (filePath.endsWith(".yml") || filePath.endsWith(".yaml")) {
    return yaml.load(raw) ?? {};
  }
  const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!match) return {};
  return yaml.load(match[1]) ?? {};
}
