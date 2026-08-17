# Current Task

> Rewritten at the start of each phase — do not append history here. Anything
> worth keeping past this phase belongs in `PROJECT_MEMORY.md` instead.

## Intent

Three documentation-only surgical corrections to the architecture contract
created in the prior phase: (1) replace stale root `ARCHITECTURE.md` with a
pointer to `docs/architecture/overview.md`, (2) fix the `rag/`↔`knowledge/`
dependency contract so `rag/` consumes indexes/stores through
`infrastructure/` contracts instead of depending on `knowledge/` internals,
(3) soften `AGENT_RULES.md`'s "never rewritten" wording and harden the
`PROJECT_MEMORY.md` token-efficiency contract (headings-first retrieval).

## Scope

`ARCHITECTURE.md`, `docs/architecture/overview.md`, `AGENT_RULES.md`,
`PROJECT_MEMORY.md` (only where these decisions need recording). No runtime
code.

## Relevant Memory

`PROJECT_MEMORY.md` → Architecture Decisions (now includes both corrections
made this phase) and Current State (ARCHITECTURE.md status).

## Success Criteria

Zero runtime source changes; `git diff --check` clean; `output/competitions/`
untouched; no two files describe the architecture independently.

## Out of Scope

Any code migration, fixing other known bugs, committing or pushing.
