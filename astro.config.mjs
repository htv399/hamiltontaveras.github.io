import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import react from "@astrojs/react";
import sitemap from "@astrojs/sitemap";

// APP-003 (approval_required): SITE_URL is a preview placeholder until the
// canonical domain is approved. Production builds fail separately if this
// placeholder is still active (see scripts/check-placeholders.mjs).
const SITE_URL = process.env.SITE_URL || "https://example-preview.invalid";
const BASE_PATH = process.env.BASE_PATH || "/";

export default defineConfig({
  site: SITE_URL,
  base: BASE_PATH,
  trailingSlash: "always",
  output: "static",
  integrations: [
    mdx(),
    react(),
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
