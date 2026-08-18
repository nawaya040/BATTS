# R2 full computation handoff

## Final status

- Status: `PASS`
- Source branch: `revision/r2-full-computation-prep`
- Source commit used for every scientific computation: `f921525711faef2a326c561820749e0c70f1bf42`
- Scientific completion: 2026-08-19 02:50:08 JST
- Comparator summary completion: 2026-08-19 03:20:06 JST
- Scientific warnings: 0
- Scientific failures: 0
- SHA-256, sidecar, canonical-status, and CDC dependency failures: 0
- Comparator data-hash disagreements: 0

The launcher status logs counted successful child-process completions as
`failed` because of the known launcher accounting artifact. Those counters are
not scientific failures. Every saved result was independently checked against
its sidecar and expected completion count.

## Frozen inputs and output location

- Frozen grid on the originating PC:
  `C:\Users\naway\Dropbox\Rcpp_experiments\active_programs\balancePM_backup\tmp\preflight-r2-full-computation-plan-final-20260813\revision_run_plan.csv`
- Frozen-grid SHA-256:
  `881bedefcd4c6d38c09c6ddb109a15e0fa0169754fad942934b294d26aef2391`
- Output root on the originating PC:
  `C:\Users\naway\Dropbox\Rcpp_experiments\active_programs\balancePM_backup\results\generated\r2-full-computation-canonical-20260813`
- Transformed-BART contract hash:
  `ce2def5a8ecb12416da12e58974eef2a6bf6dd30150f53bc59f79a4adbedf93b`

## Valid scientific outputs

| Phase | Valid units | Result-inventory SHA-256 |
|---|---:|---|
| `bart_transformed` | 200 result / `.done.rds` pairs | `e793246fc5d44aee2093f3adf9c711e2e38b226ec9e707aa40eb76ac7cbc2043` |
| `boosting` | 400 result / manifest pairs | `6b4d80dc6533476a483cf8d2fbbaab48336a3709dc8d287ade95b3646f816ed4` |
| `kernel` | 800 result / `.done.rds` pairs | `690645d5176e2fbc6501f3157565afc960280ba42f72894b663ce6cb2df1f0e3` |
| `cdc` | 400 result / `.done.rds` pairs | `20ef180c51265f8967f4765d9b2ea7dbfbd2ff40a741cad6a8cbd43f6359ec6e` |
| `coverage` | 900 result / manifest pairs | `584b81995e65d0dd663c54b8c447ff0d2aa7f75f4741b87c146627be34bca076` |

The inventory hashes above are SHA-256 digests of the sorted relative result
paths and their independently recomputed result SHA-256 values.

Coverage contains 200 valid 1D, 300 valid 2D, and 400 valid 20D units. All 400
CDC inputs were re-hashed against their recorded boosting dependency hashes.

## Comparator summary

Summary directory:
`C:\Users\naway\Dropbox\Rcpp_experiments\active_programs\balancePM_backup\results\generated\r2-full-computation-canonical-20260813\summaries\comparator-summary-20260819T0317JST`

The approved summarizer read 400 boosting, 800 kernel, and 400 CDC outputs.
It produced 3,600 job-level metric rows, 72 aggregate rows, and 1,600
data-hash audit rows. `data_hashes_consistent` is `TRUE`, with zero disagreeing
task keys.

| Summary file | SHA-256 |
|---|---|
| `comparator_metrics_by_job.csv` | `dc15cd05b5bbaa9aac9823dfa433330bd58dccb65a372da5635876529ecda0fa` |
| `comparator_metrics_summary.csv` | `efbd689ffcf8901a439dd526f11ada0cade3a6e2ee810142c7b626b9a547169a` |
| `comparator_summary.rds` | `2c78f3fb418038c685c80ce37dbdd842b3efeddba747a1baeca9309f8daf3555` |
| `data_hash_audit.csv` | `dc2601dc6775bf2da1a48f2abc5fc7d50ed3ca6e40b05af93d93d252727ec561` |

## Worker history

- Maximum scientific workers at all times: 4.
- BART transformed and boosting: 4 workers.
- Kernel: restarted on 2026-08-17 12:41 JST with 3 workers after free RAM fell
  below approximately 1.5 GB; it completed with that reduced limit.
- CDC: 4 workers after memory recovered.
- Coverage: 4 workers.

## Continue on another PC

1. Fetch and check out `revision/r2-full-computation-prep` from `origin`.
2. Wait for Dropbox to finish syncing the complete output root shown above,
   translated to the Dropbox path on the other PC.
3. Verify the frozen-grid and summary hashes in this record before analysis.
4. Use the isolated comparator summary directory as the validated compact
   entry point. The canonical RDS files remain available for detailed checks.
5. Do not rerun or overwrite canonical outputs unless a separately approved
   scientific change requires it.

Generated RDS files and ordinary scientific results remain outside Git. They
must be present through Dropbox synchronization; the Git branch contains only
source and this lightweight handoff record.
