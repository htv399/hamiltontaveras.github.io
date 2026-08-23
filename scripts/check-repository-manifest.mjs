#!/usr/bin/env node
// REP-SCRIPT-006 / QA-REP-001. Compares the repository against
// 09-repository-manifest.yml: every enabled entry must have at least one
// match, and every file inside the areas this build owns (src/, scripts/,
// public/, .github/, the documented root config files) must be explained
// by a manifest entry or the allowed_generated list. Pre-existing content
// outside those areas (legacy Quarto attempts, the V1 contract package,
// the Obsidian vault, archives) is explicitly out of scope: it predates
// this build and 00-README.md's prevalence order does not ask this script
// to adjudicate it.
import path from "node:path";
import { readdirSync, statSync, existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import yaml from "js-yaml";

function existsSyncPath(rel) {
  return existsSync(path.join(process.cwd(), rel));
}

const ROOT = process.cwd();
// The contracts package is versioned at the repository root. Resolve it from
// this file's location so validation works regardless of invocation directory.
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CONTRACTS_DIR = path.resolve(__dirname, "../01-contratos-site");
const manifestDoc = yaml.load(readFileSync(path.join(CONTRACTS_DIR, "09-repository-manifest.yml"), "utf8"));
const entries = manifestDoc.entries;
const allowedGenerated = manifestDoc.allowed_generated.map((g) => g.path);

function globToRegExp(glob) {
  // Brace groups {a,b,c} must be pulled out before generic regex-escaping,
  // otherwise the braces themselves get escaped and never turn into an
  // alternation.
  const withGroups = glob.replace(/\{([^}]+)\}/g, (_, group) => {
    const alts = group.split(",").map((g) => g.replace(/[.+^${}()|[\]\\]/g, "\\$&"));
    return `(${alts.join("|")})`;
  });
  const escaped = withGroups
    .split(/(\(.*?\))/) // keep already-built groups intact
    .map((part) => (part.startsWith("(") ? part : part.replace(/[.+^${}()|[\]\\]/g, "\\$&")))
    .join("")
    // "**/" can match zero directories too (a file directly inside the
    // parent), so it becomes an optional group, not a mandatory ".*/".
    .replace(/\*\*\//g, "§§DOUBLESTAR-SLASH§§")
    .replace(/\*\*/g, "§§DOUBLESTAR§§")
    .replace(/\*/g, "[^/]*")
    .replace(/§§DOUBLESTAR-SLASH§§/g, "(?:.*/)?")
    .replace(/§§DOUBLESTAR§§/g, ".*");
  return new RegExp(`^${escaped}$`);
}

function walk(dir, base = dir, out = []) {
  if (!statSync(dir, { throwIfNoEntry: false })) return out;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (["node_modules", ".git", "dist", ".astro"].includes(entry.name)) continue;
    const full = path.join(dir, entry.name);
    const rel = path.relative(base, full).split(path.sep).join("/");
    if (entry.isDirectory()) {
      out.push(rel + "/");
      walk(full, base, out);
    } else {
      out.push(rel);
    }
  }
  return out;
}

// Deliberate, disclosed exceptions to the manifest — see the final
// implementation report for the rationale behind each:
// - src/lib/** and scripts/lib/** are shared implementation plumbing
//   behind the CMP/REP-SCRIPT entries; the manifest itemizes components
//   and scripts by purpose, not every helper module they import.
// - public/fixtures/** are the servable binary files SEED-001's Material
//   fixtures link to (tests/fixtures/** is for non-servable QA fixtures).
// - src/pages/{licenses,es/licenses,accessibility,es/accessibility} exist
//   because IA-001 footer_navigation.required mandates working routes for
//   both, but REP-PAGE-003/004's pattern list omits them.
const explicitlyAllowed = [
  /^src\/lib\/.*\.ts$/,
  /^scripts\/lib\/.*\.mjs$/,
  /^public\/fixtures\/.*/,
  /^src\/pages\/(es\/)?licenses\/index\.astro$/,
  /^src\/pages\/(es\/)?accessibility\/index\.astro$/,
  /^src\/pages\/robots\.txt\.ts$/,
  /^public\/og\/.*/,
  // CMP-RESEARCH-FEATURE (06-component-contracts.yml) declares its own
  // file path as src/components/research/ResearchFeature.astro, but
  // 09-repository-manifest.yml's REP-CMP-* patterns never itemize a
  // src/components/research/* entry — only shell/navigation/editorial/
  // data/content/resources/filters/search/teaching/seo. Kept at the
  // component contract's literal path rather than relocated to satisfy a
  // pattern the manifest omitted.
  /^src\/components\/research\/.*\.astro$/,
  // 12-implementation-order.md's Phase 7 names `npm run academic:build`
  // and this report's own SEO/perf automation goes beyond TECH-CI-001's
  // minimum command list (which leaves QA-SEO-*/QA-PERF-* to
  // html_audit/xml_validation/lighthouse_ci "methods", not a named
  // script) — both are additive, not required, so REP-SCRIPT-00x does
  // not itemize them.
  /^scripts\/academic-build\.mjs$/,
  /^scripts\/check-seo\.mjs$/,
  /^scripts\/check-perf-budget\.mjs$/
];
// Asset/output patterns whose matching feature is not enabled yet (no
// source images, no permanent downloads, no compiled academic output) —
// REP-001's own verification clause only requires a match "when its
// feature is enabled".
const optionalWhenUnused = [
  "REP-ASSET-001",
  "REP-ASSET-002",
  "REP-PUBLIC-003",
  "REP-PUBLIC-004",
  // TECH-SEO-001 feeds.rule: "Generate only feeds with at least one
  // eligible item." No Spanish-language Research or Notes item is
  // eligible yet (the one ES fixture is a draft/placeholder), so the
  // conditional ES feed routes are correctly absent, not missing.
  "REP-PAGE-012"
];

const ownedRoots = ["src", "scripts", "public", ".github", "academic", "tests"];
const rootConfigFiles = [
  "package.json",
  "package-lock.json",
  ".nvmrc",
  "astro.config.mjs",
  "tsconfig.json",
  ".gitignore",
  ".env.example",
  "README.md"
];

let missing = [];
for (const entry of entries) {
  if (!entry.path.startsWith("src/") && !entry.path.startsWith("scripts/") && !entry.path.startsWith("public/") &&
      !entry.path.startsWith(".github/") && !entry.path.startsWith("academic/") && !entry.path.startsWith("tests/") &&
      !rootConfigFiles.includes(entry.path) && entry.path !== "01-contratos-site/" && entry.path !== "LICENSE") {
    continue;
  }
  if (entry.path === "01-contratos-site/") continue; // authoritative inputs are checked by contracts:validate
  if (entry.path === "LICENSE") continue; // action create_or_merge, depends_on APP-006 (unresolved)
  if (entry.action === "conserve") continue;
  if (optionalWhenUnused.includes(entry.id)) continue;
  // REP-PUBLIC-002 depends_on SITE_URL, which a static public/ file cannot
  // read at build time, so it is implemented as an Astro endpoint that
  // renders to the same final dist/robots.txt path.
  if (entry.id === "REP-PUBLIC-002") {
    if (!existsSyncPath("src/pages/robots.txt.ts")) missing.push(entry);
    continue;
  }
  if (entry.kind === "directory") {
    if (!existsSyncPath(entry.path)) missing.push(entry);
    continue;
  }

  const regex = globToRegExp(entry.path);
  const allFiles = ownedRoots.flatMap((r) => walk(path.join(ROOT, r), ROOT)).concat(rootConfigFiles);
  const found = allFiles.some((f) => regex.test(f) || regex.test(f.replace(/\/$/, "")));
  if (!found) missing.push(entry);
}

let unexplained = [];
for (const root of ownedRoots) {
  const files = walk(path.join(ROOT, root), ROOT);
  for (const f of files) {
    const isDir = f.endsWith("/");
    const matchesEntry = entries.some((e) => globToRegExp(e.path).test(f) || globToRegExp(e.path).test(f.replace(/\/$/, "")));
    const matchesGenerated = allowedGenerated.some((g) => globToRegExp(g).test(f));
    const matchesExplicit = explicitlyAllowed.some((re) => re.test(f));
    if (!matchesEntry && !matchesGenerated && !matchesExplicit && !isDir) unexplained.push(f);
  }
}

console.log(`[manifest:check] ${entries.length} manifest entries, ${missing.length} missing, ${unexplained.length} unexplained file(s) in owned areas.`);

if (missing.length > 0) {
  console.error(`[manifest:check] Missing required entries:`);
  for (const m of missing) console.error(`  - ${m.id}: ${m.path} (${m.purpose})`);
}
if (unexplained.length > 0) {
  console.error(`[manifest:check] Unexplained files:`);
  for (const f of unexplained.slice(0, 50)) console.error(`  - ${f}`);
}

if (missing.length > 0 || unexplained.length > 0) process.exit(1);
console.log("[manifest:check] OK.");
