# CaribbeanQuant

Static Astro portal for CaribbeanQuant, the professional platform led by Hamilton Taveras. The public information architecture is Home, DaaS Platform, Impact Products, Teaching and About Me. The Quarto/R Markdown academic pipeline remains isolated under `academic/`.

## Local development

Requires the Node.js version declared in `.nvmrc` (or a compatible newer release) and npm.

```bash
npm ci
npm run dev
```

The deployment defaults are documented in `.env.example`. No secret belongs in that file or in client-side code.

## Validation

| Command | Purpose |
| --- | --- |
| `npm run check` | Astro and TypeScript diagnostics |
| `npm run contracts:validate` | Contract IDs and references |
| `npm run content:validate` | Content models and taxonomies |
| `npm run academic:validate` | Published academic-material metadata |
| `npm run test:unit` | Unit tests |
| `npm run build` | Static site build |
| `npm run search:index` | Pagefind index after the build |
| `npm run links:check` / `npm run routes:check` | Generated links and routes |
| `npm run manifest:check` | Repository inventory |
| `npm run placeholders:check` | Production placeholder guard |
| `npm run seo:check` | Generated SEO metadata |
| `npm run perf:budget` | JavaScript and font budgets |
| `npm run test:e2e` / `npm run test:visual` | Browser and visual regression tests |
| `npm run ci` | Main continuous-integration gate |

## Content architecture

- `src/content/catalog-entries/` and `src/content/dashboards/` support the DaaS Platform.
- `src/content/impact-products/` supports versioned quantitative products.
- `src/content/courses/`, `weeks/`, `materials/` and `video-resources/` support Teaching.
- `src/content/profiles/` contains approved public profile facts.
- Legacy demo records remain private validation fixtures and never generate public pages or downloads.
- `src/config/platform.ts` controls optional public modules such as indicators, meeting links and Business Data Explorer.

The contracts in `01-contratos-site/` describe the current implementation and are validated in CI. Redirects preserve old public URLs without keeping the retired Work, Research, Notes or Spanish navigation structures in the active interface.

## Deployment

GitHub Actions validates and publishes `main` to GitHub Pages. The production workflow builds with:

- `SITE_URL=https://htv399.github.io`
- `BASE_PATH=/hamiltontaveras.github.io`
- `DEFAULT_LANGUAGE=en`

The generated artifact is `dist/`; dependencies, caches and local environment files are not committed.
