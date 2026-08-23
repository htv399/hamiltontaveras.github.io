// PG-HOME view-model assembly, shared by src/pages/index.astro and
// src/pages/es/index.astro so both languages follow the exact same
// section-order and fallback rules from 05-page-contracts.yml.
import { getCollection } from "astro:content";
import { isEligible, type SiteEnv } from "./contentGate";
import type { RowItem } from "../components/editorial/EditorialRow.astro";

// Dynamic detail routes stay unprefixed in v1 regardless of viewing
// language; see the comment in src/pages/analysis/[...slug].astro for why.
function analysisHref(_language: "en" | "es", slug: string) {
  return `/analysis/${slug}/`;
}
function workHref(_language: "en" | "es", slug: string) {
  return `/work/${slug}/`;
}
function researchHref(_language: "en" | "es", slug: string) {
  return `/research/${slug}/`;
}
function noteHref(_language: "en" | "es", slug: string) {
  return `/notes/${slug}/`;
}
function courseHref(_language: "en" | "es", slug: string) {
  return `/teaching/${slug}/`;
}

export async function buildHomeViewModel(language: "en" | "es", siteEnv: SiteEnv) {
  const [analysisAll, workAll, researchAll, notesAll, coursesAll] = await Promise.all([
    getCollection("analysis"),
    getCollection("work"),
    getCollection("research"),
    getCollection("notes"),
    getCollection("courses")
  ]);

  // IA-001 bilingual model is "selective": Home requires an equivalent route
  // in both languages, but individual editorial pieces are never
  // auto-translated. Both language shells surface the same eligible
  // inventory, each item in the language it was authored in; only the UI
  // chrome (labels, nav) changes with `language`.
  const analysis = analysisAll.filter((e) => isEligible(e.data, siteEnv));
  const work = workAll.filter((e) => isEligible(e.data, siteEnv));
  const research = researchAll.filter((e) => isEligible(e.data, siteEnv));
  const notes = notesAll.filter((e) => isEligible(e.data, siteEnv));
  const courses = coursesAll.filter((e) => isEligible(e.data, siteEnv));

  const lead = analysis.find((a) => a.data.featured) ?? analysis[0];

  const toRow = (item: { title: string; summary: string; published_at?: Date; domains: string[] }, href: string): RowItem => ({
    title: item.title,
    href,
    summary: item.summary,
    publishedAt: item.published_at,
    tags: item.domains
  });

  const eligibleFeatured = [
    ...work.filter((w) => w.data.featured).map((w) => toRow(w.data, workHref(language, w.data.slug))),
    ...research.filter((r) => r.data.featured).map((r) => toRow(r.data, researchHref(language, r.data.slug))),
    ...notes.filter((n) => n.data.featured).map((n) => toRow(n.data, noteHref(language, n.data.slug)))
  ];

  const workRows = work
    .filter((w) => w.id !== lead?.id)
    .map((w) => toRow(w.data, workHref(language, w.data.slug)));

  return {
    lead,
    hasLead: Boolean(lead),
    featured: eligibleFeatured,
    hasFeaturedGrid: eligibleFeatured.length >= 3,
    workRows,
    hasWorkSplit: workRows.length >= 2,
    research: research.filter((r) => r.id !== undefined),
    hasResearch: research.length >= 1,
    notesRows: notes.slice(0, 4).map((n) => toRow(n.data, noteHref(language, n.data.slug))),
    hasNotes: notes.length >= 1,
    courses,
    hasCourses: courses.length >= 1,
    researchHref: (slug: string) => researchHref(language, slug),
    analysisHref: (slug: string) => analysisHref(language, slug),
    courseHref: (slug: string) => courseHref(language, slug)
  };
}
