import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import react from "@astrojs/react";
import sitemap from "@astrojs/sitemap";
import { readFile, readdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

// APP-003 (approval_required): SITE_URL is a preview placeholder until the
// canonical domain is approved. Production builds fail separately if this
// placeholder is still active (see scripts/check-placeholders.mjs).
const SITE_URL = process.env.SITE_URL || "https://example-preview.invalid";
const BASE_PATH = process.env.BASE_PATH || "/";
const NORMALIZED_BASE_PATH = BASE_PATH === "/" ? "" : `/${BASE_PATH.replace(/^\/+|\/+$/g, "")}`;

// Astro prefixes bundled assets automatically, but authored root-relative
// links and the route values serialized into hydrated component props remain
// rooted at the domain. Normalize both in final HTML so project-page builds
// are deployable under BASE_PATH while root deployments remain unchanged.
const basePathHtml = {
  name: "hamilton-base-path-html",
  hooks: {
    "astro:build:done": async ({ dir, logger }) => {
      if (!NORMALIZED_BASE_PATH) return;
      const outputDir = fileURLToPath(dir);
      const entries = await readdir(outputDir, { recursive: true, withFileTypes: true });
      const htmlFiles = entries
        .filter((entry) => entry.isFile() && entry.name.endsWith(".html"))
        .map((entry) => `${entry.parentPath}/${entry.name}`);

      const prefix = (value) => {
        if (!value.startsWith("/") || value.startsWith("//")) return value;
        if (value === NORMALIZED_BASE_PATH || value.startsWith(`${NORMALIZED_BASE_PATH}/`)) return value;
        return `${NORMALIZED_BASE_PATH}${value}`;
      };

      await Promise.all(htmlFiles.map(async (file) => {
        const input = await readFile(file, "utf8");
        const output = input
          .replace(/\b(href|src|action)="(\/(?!\/)[^"]*)"/g, (_, attribute, value) => `${attribute}="${prefix(value)}"`)
          .replace(/&quot;(\/(?!\/)[^&<]*?)&quot;/g, (_, value) => `&quot;${prefix(value)}&quot;`);
        if (output !== input) await writeFile(file, output, "utf8");
      }));
      logger.info(`Normalized root-relative links in ${htmlFiles.length} HTML files for ${NORMALIZED_BASE_PATH}.`);
    }
  }
};

export default defineConfig({
  site: SITE_URL,
  base: BASE_PATH,
  trailingSlash: "always",
  output: "static",
  integrations: [
    mdx(),
    react(),
    basePathHtml,
    sitemap({
      i18n: {
        defaultLocale: "en",
        locales: { en: "en", es: "es" }
      },
      filter: (page) => !page.includes("/fixtures/")
    })
  ],
  markdown: {
    shikiConfig: {
      theme: "github-light"
    }
  }
});
