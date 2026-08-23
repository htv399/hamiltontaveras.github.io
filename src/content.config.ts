// REP-CONFIG-007. Astro content collections implementing CM-001 through
// CM-RESOURCE. Every model in 04-content-models.yml maps to a collection or
// a reusable schema fragment below. Global validation rules (CM-VAL-001..008)
// that cannot be expressed as static zod constraints are enforced at build
// time by scripts/validate-content.mjs and scripts/check-placeholders.mjs.

import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";
import {
  languages,
  editorialStatuses,
  accessLevels,
  domains,
  methods,
  objects,
  outputs,
  materialTypes
} from "./config/taxonomies";

const languageEnum = z.enum(languages);
const statusEnum = z.enum(editorialStatuses);
const accessEnum = z.enum(accessLevels);
const domainEnum = z.enum(domains);
const methodEnum = z.enum(methods);
const objectEnum = z.enum(objects);
const outputEnum = z.enum(outputs);
const materialTypeEnum = z.enum(materialTypes);

const slugPattern = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

const resourceSchema = z.object({
  id: z.string(),
  label: z.string(),
  kind: z.enum(["pdf", "data", "code", "notebook", "model", "repository", "citation", "source", "external", "image", "video"]),
  href: z.string(),
  access: accessEnum,
  license: z.string(),
  format: z.string().optional(),
  file_size_bytes: z.number().int().positive().optional(),
  checksum_sha256: z.string().optional(),
  version: z.string().optional(),
  description: z.string().optional()
});

const figureSchema = z.object({
  id: z.string(),
  title: z.string(),
  unit: z.string(),
  data_vintage: z.string(),
  source: z.array(z.string()).min(1),
  method_note: z.string(),
  accessible_summary: z.string(),
  media: z.string().optional()
});

const profileRefSchema = z.object({
  id: z.string(),
  display_name: z.string(),
  role_label: z.string()
});

// CM-001 common_fields, shared by every content-bearing model.
const commonFields = {
  id: z.string().regex(/^[A-Z0-9][A-Z0-9-]+$/),
  title: z.string().min(3).max(120),
  slug: z.string().regex(slugPattern),
  summary: z.string().min(80).max(220),
  language: languageEnum,
  status: statusEnum,
  access: accessEnum,
  authors: z.array(profileRefSchema).min(1),
  published_at: z.coerce.date().optional(),
  updated_at: z.coerce.date().optional(),
  domains: z.array(domainEnum).min(1).max(2),
  featured: z.boolean().default(false),
  placeholder: z.boolean().default(false),
  demo: z.boolean().default(false),
  subtitle: z.string().max(180).optional(),
  dek: z.string().max(260).optional(),
  methods: z.array(methodEnum).max(3).optional(),
  objects: z.array(objectEnum).max(3).optional(),
  outputs: z.array(outputEnum).optional(),
  data_vintage: z.string().optional(),
  version: z.string().optional(),
  translation_key: z.string().optional(),
  alternate_summary: z.record(languageEnum, z.string()).optional(),
  hero_media: z.string().optional(),
  thumbnail: z.string().optional(),
  sources: z.array(resourceSchema).optional(),
  resources: z.array(resourceSchema).optional(),
  related_ids: z.array(z.string()).optional(),
  citation: z.object({ text: z.string(), year: z.number().int().optional() }).optional(),
  license: z.string().optional(),
  disclosure: z.string().optional(),
  review_due: z.coerce.date().optional(),
  featured_order: z.number().int().min(1).optional()
};

const analysis = defineCollection({
  loader: glob({ pattern: "**/index.mdx", base: "./src/content/analysis" }),
  schema: z.object({
    ...commonFields,
    question: z.string().min(20),
    key_finding: z.string().min(40).max(360),
    evidence: z.array(figureSchema).min(1)
  })
});

const research = defineCollection({
  loader: glob({ pattern: "**/index.mdx", base: "./src/content/research" }),
  schema: z.object({
    ...commonFields,
    format: z.enum(["paper", "working-paper", "applied-research", "replication", "research-dataset"]),
    abstract: z.string().min(120),
    question: z.string(),
    contribution: z.string(),
    key_finding: z.string(),
    version: z.string(),
    paper_pdf: resourceSchema.optional(),
    reproduction: z.array(resourceSchema).optional()
  })
});

// CM-PROJECT, CM-DATA-PRODUCT and CM-VALUATION share the "work" collection
// with a `kind` discriminator, per REP-CONTENT-002.
const workBase = {
  ...commonFields,
  question: z.string(),
  context: z.string().optional()
};

const work = defineCollection({
  loader: glob({ pattern: "**/index.mdx", base: "./src/content/work" }),
  schema: z.discriminatedUnion("kind", [
    z.object({
      kind: z.literal("project"),
      ...workBase,
      results: z.string(),
      evidence: z.array(figureSchema).min(1),
      data: z.string().optional(),
      architecture: z.string().optional(),
      methodology: z.string().optional(),
      model: z.string().optional(),
      analysis: z.string().optional(),
      valuation: z.string().optional()
    }),
    z.object({
      kind: z.literal("data-product"),
      ...workBase,
      coverage: z.object({ geography: z.string(), period: z.string(), frequency: z.string() }),
      provenance: z.array(z.object({ source: z.string(), transformation: z.string() })).min(1),
      update_policy: z.string(),
      result: z.string(),
      architecture: z.string().optional(),
      data_dictionary: resourceSchema.optional(),
      changelog_url: z.string().optional(),
      service_level: z.string().optional()
    }),
    z.object({
      kind: z.literal("valuation"),
      ...workBase,
      valuation_date: z.coerce.date(),
      data_and_adjustments: z.string(),
      value_drivers: z.string(),
      model: z.string(),
      scenarios: z.array(z.object({ name: z.string(), value: z.number() })).min(2),
      interpretation: z.string(),
      limitations: z.string(),
      disclosure: z.string(),
      currency: z.string().optional(),
      estimate_range: z.tuple([z.number(), z.number()]).optional(),
      model_resource: resourceSchema.optional()
    })
  ])
});

const notes = defineCollection({
  loader: glob({ pattern: "**/index.mdx", base: "./src/content/notes" }),
  schema: z.object({
    ...commonFields,
    note_type: z.enum(["data-note", "economic-note", "valuation-note", "market-note", "methods-note"]),
    key_point: z.string().optional(),
    series: z.string().optional(),
    historical_update_note: z.string().optional()
  })
});

const courses = defineCollection({
  loader: glob({ pattern: "*.mdx", base: "./src/content/courses" }),
  schema: z.object({
    ...commonFields,
    code: z.string(),
    description: z.string(),
    objectives: z.array(z.string()).min(1),
    period: z.string(),
    syllabus: z.string().optional(),
    institution: z.string().optional(),
    prerequisites: z.array(z.string()).optional(),
    archived: z.boolean().default(false)
  })
});

const weeks = defineCollection({
  loader: glob({ pattern: "**/*.mdx", base: "./src/content/weeks" }),
  schema: z.object({
    ...commonFields,
    course_id: z.string(),
    week_number: z.number().int().min(1),
    question: z.string(),
    overview: z.string(),
    resources: z.array(z.string()).min(1),
    version: z.string(),
    license: z.string(),
    concepts: z.array(z.string()).optional(),
    learning_objectives: z.array(z.string()).optional(),
    previous_week_id: z.string().optional(),
    next_week_id: z.string().optional()
  })
});

const materials = defineCollection({
  loader: glob({ pattern: "**/*.yml", base: "./src/content/materials" }),
  schema: z.object({
    id: z.string(),
    title: z.string(),
    slug: z.string().regex(slugPattern),
    language: languageEnum,
    material_type: materialTypeEnum,
    access: accessEnum,
    status: statusEnum,
    version: z.string(),
    license: z.string(),
    href: z.string(),
    file_size_bytes: z.number().int().positive().optional(),
    mime_type: z.string().optional(),
    checksum_sha256: z.string().optional(),
    source_href: z.string().optional(),
    description: z.string().optional(),
    updated_at: z.coerce.date().optional(),
    demo: z.boolean().default(false),
    placeholder: z.boolean().default(false),
    // TCH-001 resource_groups: which of the eight ordered groups (slides,
    // notes, manual_exercises, lab, data, code, quiz, bibliography) this
    // material belongs to on the Week page. Not in CM-MATERIAL's own field
    // list, but required to implement resource_groups.render_rule.
    group: z.enum(["slides", "notes", "manual_exercises", "lab", "data", "code", "quiz", "bibliography"]).optional()
  })
});

const profiles = defineCollection({
  loader: glob({ pattern: "*.yml", base: "./src/content/profiles" }),
  schema: z.object({
    id: z.string(),
    display_name: z.string(),
    role_label: z.string(),
    bio_short: z.string().max(300).optional(),
    photo: z.string().optional(),
    orcid: z.string().optional(),
    website: z.string().optional(),
    social: z.array(z.object({ label: z.string(), href: z.string() })).optional()
  })
});

const catalogEntries = defineCollection({
  loader: glob({ pattern: "**/*.yml", base: "./src/content/catalog-entries" }),
  schema: z.object({
    name: z.string().min(3),
    slug: z.string().regex(slugPattern),
    provider: z.string(),
    geography: z.array(z.string()).min(1),
    topic: z.array(z.string()).min(1),
    frequency: z.string(),
    time_coverage: z.string(),
    update_status: z.string(),
    last_update: z.coerce.date().optional(),
    access_level: accessEnum,
    description: z.string().min(80).max(400),
    available_formats: z.array(z.string()).min(1),
    methodological_note: z.string().optional(),
    source_link: z.string().url().optional(),
    request_access_action: z.string().url().optional(),
    language: languageEnum,
    status: statusEnum
  })
});

const dashboards = defineCollection({
  loader: glob({ pattern: "**/*.yml", base: "./src/content/dashboards" }),
  schema: z.object({
    kind: z.enum(["dashboard", "monitor"]),
    title: z.string().min(3),
    slug: z.string().regex(slugPattern),
    summary: z.string().min(80).max(300),
    topic: z.array(z.string()).min(1),
    geography: z.array(z.string()).min(1),
    data_sources: z.array(z.string()).min(1),
    last_update: z.coerce.date().optional(),
    access_level: accessEnum,
    thumbnail: z.string().optional(),
    embed_url: z.string().url().optional(),
    external_url: z.string().url().optional(),
    methodology: z.string().optional(),
    status: statusEnum,
    cta: z.string().optional(),
    language: languageEnum
  })
});

const impactProducts = defineCollection({
  loader: glob({ pattern: "**/index.mdx", base: "./src/content/impact-products" }),
  schema: z.object({
    title: z.string().min(3),
    slug: z.string().regex(slugPattern),
    summary: z.string().min(80).max(300),
    problem: z.string(),
    geography: z.array(z.string()).min(1),
    topic: z.array(z.string()).min(1),
    method: z.array(z.string()).min(1),
    data_sources: z.array(z.string()).min(1),
    update_frequency: z.string().optional(),
    status: statusEnum,
    version: z.string(),
    main_result: z.string().optional(),
    exhibits: z.array(figureSchema).optional(),
    methodology: z.string(),
    limitations: z.string(),
    downloads: z.array(resourceSchema).optional(),
    repository: z.string().url().optional(),
    license: z.string().optional(),
    access_level: accessEnum,
    featured: z.boolean().default(false),
    language: languageEnum
  })
});

const videoResources = defineCollection({
  loader: glob({ pattern: "**/*.yml", base: "./src/content/video-resources" }),
  schema: z.object({
    title: z.string().min(3),
    slug: z.string().regex(slugPattern),
    language: languageEnum,
    provider: z.literal("youtube"),
    url: z.string().url(),
    playlist_url: z.string().url().optional(),
    access_level: accessEnum,
    status: statusEnum,
    course_id: z.string().optional(),
    week_id: z.string().optional(),
    description: z.string().optional(),
    duration: z.string().optional()
  })
});

export const collections = {
  analysis,
  research,
  work,
  notes,
  courses,
  weeks,
  materials,
  profiles,
  catalogEntries,
  dashboards,
  impactProducts,
  videoResources
};
