# Project agent rules

Status: draft. Until replaced, the approval gates below are binding.

## Scope and objective

These rules apply to the entire repository.

The objective is a minimal, traceable revision of the density-ratio estimation
paper, together with a verified scientific implementation and a reproducible
journal submission package.

Keep four workstreams separate:

1. independent scientific-code audit;
2. reviewer-driven revision;
3. migration and path stabilization;
4. final reproducibility packaging.

Do not mix their changes in one patch or commit.

## Non-negotiable approval gates

### Scientific and experimental code

Protected scientific files include:

- `R/**`
- `src/**`
- `Experiments/**`
- future experiment, analysis, test, configuration, dependency-lock, and
  reproducibility code;
- any file whose change can affect data, RNG, numerical results, tables, or
  figures.

Before editing a protected file, the agent must:

1. diagnose the issue without modifying the file;
2. identify the exact files and lines;
3. present the minimal proposed change;
4. state whether numerical results or RNG streams may change;
5. state the validation and rollback plan;
6. receive explicit user approval after presenting that proposal.

The initial request, a broad instruction to revise, or a reviewer comment is not
approval for a scientific-code patch. Approval is scoped to the proposed patch.
Unrelated formatting or refactoring is not permitted.

Path-only changes inside protected files use the same approval process.

### Manuscript and Overleaf

- Reading and comparing manuscript files is allowed.
- Before any manuscript edit, present the proposed locations and change, then
  receive explicit approval.
- Approval to edit locally is not approval to push.
- Before every Overleaf push, show the exact diff/commits and receive a
  separate explicit approval.
- Do not pull, merge, rebase, force-push, or resolve Overleaf conflicts without
  approval. A read-only fetch or remote inspection is allowed.

### Data, results, migration, and external state

- Treat input data and existing results as read-only.
- Do not overwrite, rename, move, or delete them.
- Do not run a canonical or costly experiment, or write into an official
  results directory, without approval.
- Syntax checks and reduced smoke tests are allowed only when they write to an
  isolated temporary directory and cannot overwrite official results.
- Do not copy or move the project out of Dropbox, initialize its replacement
  repository, change remotes, or perform the cutover without approval.
- Do not delete the Dropbox copy unless separately and explicitly approved
  after verification.

## Actions allowed without prior approval

- Read-only inventory, search, static analysis, parsing, and comparison.
- Reading Git state and remote metadata.
- Creating or updating governance, audit, and planning documents that do not
  change scientific claims or manuscript text.
- Running lightweight diagnostics in an isolated temporary directory.
- Drafting proposed patches without applying them.

Report any files created or changed.

## Required working method

1. Read `AGENTS.md` and `PROJECT_OPERATIONS_DRAFT.md`.
2. Inspect current files and preserve user changes.
3. Determine the authoritative code and data source before reorganizing.
4. Prefer the smallest reversible action.
5. Separate diagnosis from implementation.
6. Stop at an approval gate and wait.
7. After an approved change, validate in proportion to scientific risk.
8. Report exact files, checks performed, results, limitations, and remaining
   decisions.

Never present an incomplete environment check as scientific validation.
Never claim reproducibility from a successful syntax or package build alone.

## Scientific integrity and minimal revision

- Preserve the submitted estimand, model, priors, hyperparameters, seeds,
  sample sizes, scenarios, and evaluation metrics unless the approved change
  explicitly alters them.
- Label every proposed scientific change as reviewer-requested, independent bug
  fix, performance-only, or reproducibility infrastructure.
- State whether a change can alter published numbers.
- Do not optimize by changing RNG order, parallel reduction order, posterior
  sampling, convergence criteria, numerical precision, or method defaults
  without treating it as a scientific change.
- Do not regenerate published outputs merely to make them look cleaner.
- Prefer a narrow correction and an explicit limitation over an unrequested
  expansion.

## Reproducibility rules

- Use project-relative paths for repository files.
- Use explicit external data and result roots. Do not hard-code usernames,
  Dropbox, drive-specific research paths, or `setwd()`.
- Validate inputs before computation and create output directories explicitly.
- Never silently reuse or overwrite partial output.
- Each official run must record:
  commit hash, configuration, R and package versions, compiler and OS, RNG kind,
  seed mapping, parallel backend and core count, input hashes, elapsed time, and
  output hashes.
- Maintain separate smoke and canonical configurations. Smoke settings must
  never be mistaken for paper settings.
- Pin the known-working environment before considering dependency upgrades.
- Keep generated object files, DLLs, histories, caches, and ordinary generated
  results out of source control. Preserve approved reference artifacts by an
  explicit policy, not by accident.

## Git rules

- Do not initialize Git or create/change remotes without approval.
- Once Git is established, use focused branches and atomic commits.
- Keep independent audit fixes separate from reviewer revisions.
- Do not commit secrets, local `.Renviron`, unrestricted data, or machine-local
  paths.
- Do not rewrite shared history or use destructive cleanup without explicit
  approval.
- The code repository and Overleaf-connected paper repository must remain
  separate.

## Current known risks

Before executing experiments, account for these unresolved findings:

- active scripts use `BATTS::boots()` and `BATTS::batts()`, while this directory
  declares package `balancePM` with a different public API;
- `Experiments/experiment_1D.R` appears to save an undefined `out` object for
  repeats after the first;
- active scripts contain Dropbox-specific input and output paths;
- output directories are generally assumed to exist;
- package metadata, dependency declarations, tests, and environment locking are
  incomplete;
- source, compiled artifacts, results, and old experiments are intermingled.

These are findings, not authorization to fix them.

