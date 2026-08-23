// REP-PUBLIC-002. Astro endpoint (not a static public/ file) so it can
// read SITE_URL at build time instead of hardcoding a domain.
import type { APIRoute } from "astro";
import { canonicalBase } from "../config/site";

export const GET: APIRoute = () => {
  const base = canonicalBase.value.replace(/\/$/, "");
  const body = `User-agent: *\nAllow: ${import.meta.env.BASE_URL}\n\nSitemap: ${base}/sitemap-index.xml\n`;
  return new Response(body, { headers: { "Content-Type": "text/plain; charset=utf-8" } });
};
