# Hamilton Taveras — professional portal

Static Astro publication (Home, Work, Research, Notes, Teaching, About and
utilities) with a separate Quarto/R Markdown pipeline for academic
materials. Specified by [`01-contratos-site/`](01-contratos-site/00-README.md) — that
directory is the source of truth and is not edited by this codebase. This
folder (`Portal_Hamilton_V3/`) holds only the implementation.

## Requirements

- Node.js version pinned in [`.nvmrc`](.nvmrc) (20.14.0 or newer within the
  same major; Astro 5 also supports Node 22).
- `npm` (uses `package-lock.json`; CI always runs `npm ci`, never `npm i`).

## Getting started

```bash
npm ci
cp .env.example .env   # edit if you need non-default preview values
npm run dev
```

## Scripts

| Command | What it does |
| --- | --- |
| `npm run dev` | Astro dev server |
| `npm run build` | Static build to `dist/` |
| `npm run preview` | Serve the built `dist/` locally |
| `npm run check` | `astro check` (TypeScript + template diagnostics) |
| `npm run contracts:validate` | Cross-references ids across `01-contratos-site/*` |
| `npm run content:validate` | CM-VAL-* rules beyond what zod schemas express |
| `npm run materials:check` | Validates Material fixtures, checksums and MIME types |
| `npm run academic:build` | Renders `academic/materials-manifest.yml` bridges via Quarto, if installed |
| `npm run search:index` | Builds the Pagefind index (run after `build`) |
| `npm run links:check` / `routes:check` | Crawls `dist/` for broken links / empty routes |
| `npm run manifest:check` | Compares the repo against `09-repository-manifest.yml` |
| `npm run placeholders:check` | Blocks `placeholder:true` content and stray "TODO" text in production |
| `npm run seo:check` | Title/description/canonical/OG/sitemap presence |
| `npm run perf:budget` | Font and JS-bundle size budgets (TECH-PERF-001) |
| `npm run og:generate` | Renders `public/og/default.png` |
| `npm run test:unit` | Vitest (`tests/unit/`) |
| `npm run test:e2e` / `test:a11y` / `test:responsive` / `test:visual` | Playwright (`tests/e2e/`, `tests/visual/`) — run `npx playwright install chromium` once first |
| `npm run ci` | The aggregate preview-mode gate (matches `.github/workflows/validate.yml`) |

## Environments

Set `SITE_ENV=production` to build for release. Production mode:

- Fails the build if any content still has `placeholder: true` (SEED-001
  demo fixtures) or a page gated by an unresolved `approval_required`
  decision (`src/lib/productionGate.ts`) — currently **About, CV and
  Contact**, blocked on **APP-008** (real inventory/biography) and
  **APP-005** (contact channel).
- Requires `SITE_URL`, `DEFAULT_LANGUAGE` and `CONTACT_MODE` to be set to
  real, approved values (see `.env.example`).

Preview mode (the default) renders SEED-001's demo fixtures so every page
template, component and route can be reviewed before real content exists.

## Pending decisions (APP-001..APP-009)

Tracked in `src/config/site.ts` and `src/config/licenses.ts`, never
inferred. See `01-contratos-site/00-README.md` for the full list and
`../decisiones-pendientes.md` / `../decisiones-placeholder-en-uso.md`
(one level up) for the pre-Astro (V1) record of the same open questions.

## Repository layout

- `src/content/` — MDX/YAML content collections (all current entries are
  SEED-001 demo fixtures; no real content has been authored yet).
- `src/components/`, `src/layouts/`, `src/pages/` — the Astro site.
- `academic/` — Quarto/R Markdown sources, isolated from the public
  framework per TECH-001; `academic/materials-manifest.yml` bridges a
  source to a `public/materials/` output and a `src/content/materials/*.yml`
  record once a course is ready to publish.
- `scripts/` — the validation scripts listed above.
- `tests/unit/`, `tests/e2e/`, `tests/visual/` — Vitest and Playwright.
- `01-contratos-site/` — the
  authoritative specification, conserved and not modified by this build.

## Deployment

GitHub Pages via `.github/workflows/deploy.yml` (build) and
`actions/deploy-pages` (publish), gated by `.github/workflows/validate.yml`
on every push and pull request. Set repository variables `SITE_URL`,
`BASE_PATH`, `DEFAULT_LANGUAGE` and `CONTACT_MODE` before the first
production deploy.
