// REP-PAGE-011 / TECH-SEO-001. feeds.research: "/research/rss.xml".
// Never includes private or placeholder content, regardless of SITE_ENV.
import rss from "@astrojs/rss";
import { getCollection } from "astro:content";
import { siteConfig } from "../../config/site";

export async function GET() {
  const all = await getCollection("research");
  const items = all.filter((e) => e.data.access !== "private" && !e.data.placeholder && e.data.status !== "draft");
  return rss({
    title: `${siteConfig.name} — Research`,
    description: "Applied research, working papers, replications and documented datasets.",
    site: siteConfig.canonicalBase,
    items: items.map((entry) => ({
      title: entry.data.title,
      description: entry.data.summary,
      pubDate: entry.data.published_at,
      link: `/research/${entry.data.slug}/`
    }))
  });
}
