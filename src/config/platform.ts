import type { Indicator } from "../components/data/IndicatorStrip.astro";

export const platformContent = {
  daas: {
    title: "DaaS Platform",
    summary: "A structured home for documented data sources, coverage, access conditions, dashboards and reusable resources as they are approved for publication."
  },
  impact: {
    title: "Impact Products",
    summary: "Quantitative products will be published here only when their data, method, version, limitations and access terms are documented."
  },
  teaching: {
    title: "Teaching",
    summary: "Courses and academic resources published as durable, accessible materials rather than as a learning management system."
  },
  indicators: [] as Indicator[],
  meeting: { label: "Request a platform demonstration", href: null as string | null },
  businessExplorer: { publishable: false, title: "Business Data Explorer" }
} as const;
