#!/usr/bin/env node
// REP-SCRIPT-001 / QA-TRACE-001. Parses every YAML contract in
// 01-contratos-site, confirms it is well-formed, and confirms that every
// referenced requirement id (maps_to, origin, component:, model:, decision
// ids...) resolves to an id declared somewhere in the same contract corpus.
// This does not re-implement editorial judgment: it only catches typos and
// dangling references between contract files.
import {
  loadYamlContracts,
  collectDeclaredIds,
  collectReferencedIds,
  collectIdsDeclaredInMarkdown
} from "./lib/contracts.mjs";

let failed = false;

let docs;
try {
  docs = loadYamlContracts();
} catch (err) {
  console.error(`[contracts:validate] FAIL — a contract file did not parse as YAML.\n${err.message}`);
  process.exit(1);
}

console.log(`[contracts:validate] Parsed ${Object.keys(docs).length} YAML contract files.`);

const declared = new Set([...collectDeclaredIds(docs), ...collectIdsDeclaredInMarkdown()]);
const referenced = collectReferencedIds(docs);

// A handful of ids are referenced by design but declared only as *values*,
// not as an `id:` field (e.g. component file globs, discriminators, or a
// fixture's business "code" rather than a contract requirement). Keep an
// explicit allowlist instead of silently ignoring unknown prefixes.
const externalAllowlist = new Set([
  "INTERNAL-DEMO-ONLY", // SEED-001 demo license marker, defined in src/config/licenses.ts
  "DEMO-ECON-001" // SEED-001 COURSE-DEMO-001.code, a content value, not a requirement id
]);

const dangling = [...referenced].filter((id) => !declared.has(id) && !externalAllowlist.has(id));

if (dangling.length > 0) {
  failed = true;
  console.error(`[contracts:validate] FAIL — ${dangling.length} referenced id(s) do not resolve to a declared id:`);
  for (const id of dangling.sort()) console.error(`  - ${id}`);
} else {
  console.log(`[contracts:validate] OK — every referenced id resolves (${declared.size} declared, ${referenced.size} referenced).`);
}

process.exit(failed ? 1 : 0);
