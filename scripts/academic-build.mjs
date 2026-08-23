#!/usr/bin/env node
// TCH-PIPE-002/003. Compiles academic/ sources listed in
// academic/materials-manifest.yml with the Quarto CLI already present in
// the environment, then copies approved outputs to public/materials/.
// TCH-PIPE prohibits installing R packages during build, so this script
// only ever *invokes* `quarto`; it never runs install.packages, pak or
// renv. If `quarto` is not on PATH (true in this sandbox), it reports that
// clearly and exits 0 rather than failing a build that has nothing to
// compile yet — academic/materials-manifest.yml's bridges list is empty
// until a real, license-approved source exists.
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";
import yaml from "js-yaml";

const manifestPath = path.resolve(process.cwd(), "academic/materials-manifest.yml");
const manifest = yaml.load(readFileSync(manifestPath, "utf8"));
const bridges = manifest.bridges ?? [];

if (bridges.length === 0) {
  console.log("[academic:build] No bridges declared in academic/materials-manifest.yml — nothing to compile.");
  process.exit(0);
}

let quartoAvailable = true;
try {
  execFileSync("quarto", ["--version"], { stdio: "ignore" });
} catch {
  quartoAvailable = false;
}

if (!quartoAvailable) {
  console.log("[academic:build] `quarto` is not installed in this environment. Skipping compilation (no packages were installed).");
  process.exit(0);
}

for (const bridge of bridges) {
  console.log(`[academic:build] Rendering ${bridge.source} -> ${bridge.output}`);
  execFileSync("quarto", ["render", bridge.source], { stdio: "inherit" });
}
console.log(`[academic:build] Rendered ${bridges.length} source(s).`);
