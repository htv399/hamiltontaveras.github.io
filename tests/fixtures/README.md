# Test fixtures

REP-TEST-004 / QA-CM-001. These fixtures are inputs for future schema
tests, not servable files (compare with `public/fixtures/`, which holds the
binary files SEED-001's Material records actually link to and download).

`invalid-analysis-missing-question.yml` is a deliberately invalid Analysis
record — it omits `question`, a required field — for exercising "one
invalid fixture per required field fails with a useful message"
(QA-CM-001). The live schema validation for this repository runs through
`astro check` / `astro build` against `src/content.config.ts`; wiring an
isolated zod-schema unit test harness (without duplicating the schema
outside `astro:content`) is tracked as follow-up work, not done in this
pass.
