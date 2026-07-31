# Density-ratio revision project: operation draft

Status: draft  
Initial audit date: 2026-07-31

## 1. Purpose

This project has four separate workstreams.

1. Audit and stabilize the scientific code independently of reviewer requests.
2. Plan and implement the smallest revision that answers the reviewers.
3. Move the working project out of Dropbox and remove machine-specific paths.
4. Assemble the journal-specific reproducibility package after the revision is settled.

The workstreams should remain distinguishable in Git history. A reviewer-driven
change must not be mixed with an unrelated code cleanup.

## 2. Initial read-only audit

### Highest-priority findings

1. **The active experiments do not use the package in this directory.**
   `DESCRIPTION` declares package `balancePM`, whose exported R API is
   `estimate_balancing_weight_*()`. The active experiments instead attach
   `BATTS` and call `boots()` and `batts()`. The local package and the program
   that produced the submitted results may therefore be different codebases.
   The reproducibility directory supplied later must be compared with both.

2. **`Experiments/experiment_1D.R` appears to fail for repeats 2--50.**
   `out` is created only when `index_repeat == 1`, but `saveRDS(out, ...)` is
   unconditional. This is a proposed bug finding only. No correction has been
   made.

3. **Active scripts contain Dropbox-specific absolute paths.**
   Inputs and outputs depend on `C:/Users/<user>/Dropbox/...`. Output
   directories are assumed to exist. This blocks portable execution and makes
   migration fragile.

4. **The present directory is not a Git repository.**
   There is no commit-level provenance for the code, results, or future
   revision.

### Other findings to resolve before a reproducibility release

- `R/check_multi_scenarios.R` is an executable exploratory script inside the
  package's `R/` directory. It attaches packages, sources files, simulates data,
  and plots at top level. Package code should not have these load-time side
  effects.
- `DESCRIPTION` still contains placeholder title, author, description, and
  license fields. Runtime dependencies used by helper and simulation code are
  not fully declared.
- Source code, old experiments, compiled objects, a DLL, R history, figures,
  and large result files are mixed in the package root.
- There is no dependency lockfile, automated test suite, run manifest, or
  documented entry point.
- All current R files parse except
  `Experiments/old_files/test_check_temperature.R`, which has a syntax error at
  line 144.
- `R/utilities_for_experiment.R` hard-codes `nu = 0.01` in the final AdaBoost
  fit instead of using its `learn_rate` argument. Its optional kappa branch
  minimizes kappa rather than maximizing it. Current main settings may mask
  these defects.
- `Experiments/experiment_multi_add.R` replaces negative spectral estimates
  before taking a logarithm, but not zero estimates. Zero can therefore produce
  an infinite value.
- An isolated `R CMD check` reached compilation but did not complete because
  the local R/Rtools compiler path was not resolved by the check subprocess.
  It also warned about `src/adaboost_functions.R`. This is an incomplete
  environment-level check, not evidence that the C++ code is correct or
  incorrect.

No scientific or experimental source file was changed during this audit.

## 3. Proposed repository layout

Keep the code and manuscript in separate repositories so that an ordinary code
push cannot update Overleaf.

```text
C:\Users\user\projects\
├── density-ratio\          # scientific code and reproducibility materials
└── density-ratio-paper\    # separate clone connected to Overleaf
```

The code repository should eventually have the following conceptual layout.
Existing directories should not be renamed until the submitted
reproducibility directory has been identified as the source of truth.

```text
density-ratio\
├── R\                      # package R code
├── src\                    # package C++ code
├── man\                    # generated package documentation
├── tests\                  # unit, invariant, and smoke tests
├── Experiments\            # canonical experiment code; retain name initially
├── scripts\                # controlled run entry points
├── config\                 # versioned example and run configurations
├── data\
│   └── README.md           # acquisition and checksum instructions
├── results\
│   ├── reference\          # approved small reference summaries/checksums
│   └── generated\          # ignored regenerable outputs
├── review\                 # comments, response matrix, decision log
├── reproducibility\        # journal release bundle, created late
└── docs\                   # audit notes and run documentation
```

Large or restricted data should remain outside Git. The repository should store
only acquisition instructions, schema, license/usage constraints, and
cryptographic checksums. Large indispensable artifacts can use Git LFS only
after an explicit decision.

## 4. Migration procedure

Migration should be a verified cutover, not an immediate move.

1. Freeze the Dropbox source for the duration of migration.
2. Record an inventory with relative path, byte size, modification time, and
   SHA-256 hash.
3. Copy to `C:\Users\user\projects\density-ratio`.
4. Recompute the inventory and require exact hash agreement.
5. Initialize Git in the destination and create an immutable baseline commit.
6. Run parse, package, unit, and reduced smoke checks from the destination.
7. Switch normal work to the destination.
8. Keep the Dropbox copy as a read-only fallback until the user separately
   approves archival or removal.

Each write, copy, Git initialization, or cutover step requires approval before
execution. The current audit does not authorize migration.

## 5. Path and environment policy

- Resolve repository files relative to the project root.
- Do not use `setwd()` in production scripts.
- Use environment variables for external locations, provisionally:
  `DRE_DATA_DIR`, `DRE_RESULTS_DIR`, and `DRE_REPRO_DIR`.
- Track a non-secret example configuration. Keep the user's real `.Renviron`
  or local configuration ignored.
- Validate required inputs before starting a costly run.
- Create output directories explicitly.
- Never overwrite an approved result. A rerun receives a new run ID.
- Record R version, package versions, compiler, OS, RNG kind, seed map,
  parallel backend, input hashes, configuration hash, commit hash, and output
  hashes in a run manifest.

Changing path handling inside an experiment is still an experimental-code
change and must pass the approval gate.

## 6. Git and release policy

Suggested history:

- `main`: verified, usable states only.
- `audit/<topic>`: scientific-code audit fixes.
- `revision/<reviewer-id>`: reviewer-requested changes.
- `repro/<release>`: journal reproducibility packaging.

Suggested tags:

- `submission-original`
- `revision-start`
- `revision-candidate`
- `revision-submitted`

Use small commits with one purpose. Do not mix formatting, optimization, and
scientific changes. Record whether each scientific change is:

- reviewer-requested,
- independent bug fix,
- performance-only, or
- reproducibility infrastructure.

The paper repository should have its own Overleaf remote. Fetching and
inspecting remote state is distinct from modifying the manuscript. Local
manuscript edits require approval after a proposed diff. Any Overleaf push
requires a second, explicit approval.

## 7. Scientific-code audit workflow

1. Import the submitted reproducibility directory.
2. Build a file-level and API-level comparison against this directory.
3. Identify the exact implementation that produced each submitted table and
   figure.
4. Preserve original code and output hashes as the baseline.
5. Classify findings by whether they can change numerical results.
6. For each proposed correction, present:
   affected files, minimal diff, scientific consequence, validation plan,
   expected output changes, and rollback.
7. Wait for explicit approval.
8. Apply only the approved diff.
9. Run reduced tests first. Run canonical experiments only after separate
   approval.

Optimizations should first target orchestration, duplicated I/O, avoidable
copies, checkpointing, and safe parallel scheduling. Algorithmic changes should
not be labeled as optimization when they can change the estimand, RNG sequence,
posterior sample, convergence behavior, or floating-point order.

## 8. Reviewer workflow

Create a response matrix after the review files arrive.

| Field | Meaning |
|---|---|
| Comment ID | Stable reviewer/comment identifier |
| Request | Literal request |
| Interpretation | What must be satisfied |
| Proposed response | Minimal response strategy |
| Manuscript location | Exact section/line |
| Code or experiment impact | None, analysis, rerun, or new experiment |
| Decision | Accept, clarify, or respectfully decline |
| Approval state | Proposed, approved, implemented, verified |
| Evidence | Test, output, or manuscript diff |

Prefer clarification and local edits over expansion. New experiments,
algorithmic changes, or broad rewrites need a specific justification. The
independent code audit remains separate from the reviewer response unless a
finding directly affects a claim.

## 9. Reproducibility release

After the revision is stable and the journal template is supplied:

1. Map every reported table and figure to one command and one configuration.
2. Provide a fast smoke mode and an unchanged canonical mode.
3. Pin the working dependency environment without opportunistic upgrades.
4. Include data acquisition or placement instructions and checksums.
5. Capture expected run time, memory, core count, and storage.
6. Verify from a clean directory or clean machine-equivalent environment.
7. Compare key summaries and hashes against the approved reference.
8. Package according to the journal's rules without changing scientific code.

The journal template may alter the packaging layer. It should not silently
alter the numerical implementation.

