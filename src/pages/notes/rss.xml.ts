// REP-PAGE-011 / TECH-SEO-001. feeds.notes: "/notes/rss.xml".
import rss from "@astrojs/rss";
import { getCollection } from "astro:content";
import { siteConfig } from "../../config/site";

export async function GET() {
  const all = await getCollection("notes");
  const items = all.filter((e) => e.data.access !== "private" && !e.data.placeholder && e.data.status !== "draft");
  return rss({
    title: `${siteConfig.name} — Notes`,
    description: "Concise notes on data, economics, valuation, markets and methods.",
    site: siteConfig.canonicalBase,
    items: items.map((entry) => ({
      title: entry.data.title,
      description: entry.data.summary,
      pubDate: entry.data.published_at,
      link: `${import.meta.env.BASE_URL}notes/${entry.data.slug}/`
    }))
  });
}
