# BATTS 0.1.0 validation

Validation date: 2026-10-01 (JST).

## Scope and source identity

The release starts from development commit
`909ea357bdb3f97018613552ef13430d1bb60c6e`. Numerical source changes in this
release preparation are limited to renaming the prediction-only C++ type
and selecting C++17. Public R function bodies and defaults were checked
against that baseline and are unchanged. The type-name transformation was
also checked against the baseline source token for token after normalizing
line endings.

The tested implementation, regression suite and API documentation are at
`e3c1c1613454516f4664380e18c810c2b9639a81`. The final release additionally
contains this validation record. The earlier public main at `6f625ba`
lacks the nine development commits summarized in NEWS.md; equivalence to
that older implementation is not claimed.

## Environment

- Windows 11 x64, R 4.5.2, Rtools45 / GCC 14.3.0.
- Rcpp 1.1.1.1.
- RcppArmadillo 14.4.1.1 (existing environment) and 15.6.0.1 (isolated library).
- R's default BLAS; LAPACK 3.12.1; `LANG=C`, `LC_ALL=C`.
- Ordinary optimized build: `-O2`; debug build: `-O0 -g`.
- OpenMP disabled using the existing `ARMA_DONT_USE_OPENMP` setting.

User libraries, compiler configuration and existing scientific results were
not overwritten. All builds and numerical checks used separate output and
library directories.

## Results

| Check | Result |
| --- | --- |
| Baseline C++11 with RcppArmadillo 14.4.1.1 | Installs |
| Baseline C++11 with RcppArmadillo 15.6.0.1 | Expected compilation failure: C++14 or newer required |
| Candidate C++17 with either dependency version | Installs and loads |
| Candidate C++17 at -O0 | Installs and loads |
| Include both fitting and prediction headers in one translation unit | Baseline fails with Node redefinition; candidate compiles |
| Numerical regression suite on candidate, debug and modern-dependency builds | Passes in all three configurations |
| Same-seed comparisons against 909ea357 | Complete fitted objects, predictions and RNG states exactly identical in all comparisons below |
| R CMD check of archived source, with modern dependency | 0 errors, 0 warnings, 1 NOTE |

The remaining NOTE is the existing unused namespace declaration for
RcppArmadillo in Imports. RcppArmadillo is used through LinkingTo for C++
headers. The dependency declaration was retained in this scoped patch.

### Numerical comparisons

Six baseline snapshots cover FS, GB and BAT at group sizes (40,40) and
(60,20), with two numeric covariates. With seed 1001, the first coordinate
is standard normal in group 0 and normal with mean 0.6 in group 1; the
second is standard normal. Each fit starts at seed 1002.

Boosting uses a maximum of 15 trees, four-fold CV, depth two and eight bins.
BAT uses eight trees, 20 burn-in iterations, 40 saved draws and saved forests.
Predictions are evaluated at the training points. Each snapshot contains the
entire fitted R object, prediction output and final `.Random.seed`.

Each of the six snapshots was compared with the candidate ordinary build,
the candidate -O0 build and the candidate modern-dependency build: all 18
comparisons passed `identical()`, with maximum absolute numeric difference
zero and identical final RNG states. Prediction left RNG states unchanged.
These are reduced regression cases, not paper-scale experiments or evidence
that every possible input produces bitwise-identical results.

### Regression suite

`tests/regression.R` checks:

- FS/GB training weights against evaluated forests, including normalization;
- deterministic fits and final RNG state under repeated seeds;
- group-permuted inputs under CV and finite CV losses;
- a hand-specified tree's boundary predictions and out-of-domain rejection;
- fitted predictions on repeated values at candidate cut points;
- training posterior draws against repeatedly evaluated saved forests;
- unchanged training draws and RNG when forest saving is switched off;
- errors for unavailable posterior forests, unsupported initialization,
  missing draw count, invalid labels, constant columns, invalid CV count,
  and the unsupported nondefault ratio parameter.

The first boundary test used a vector expectation for an Rcpp-returned
one-column matrix; its shape comparison was corrected before the final
check. The numerical boundary values were correct. The initial worktree
build also included its `.git` pointer; the checked distribution archive
was instead built from a clean Git export and contains no Git metadata.

## Repeating the checks

From a clean export of the tagged source, with Rtools on PATH on Windows:

```text
R CMD build BATTS
R CMD INSTALL --preclean --library=<isolated-library> BATTS_0.1.0.tar.gz
R CMD check --no-manual --no-multiarch BATTS_0.1.0.tar.gz
```

Set `R_LIBS_USER` to select the isolated dependency and package libraries.
The regression test is run by R CMD check; it can also be run using
`Rscript --vanilla tests/regression.R` with the candidate on the library path.
For the debug build, use an isolated `R_MAKEVARS_USER` with
`CXX17FLAGS = -O0 -g -Wall -mfpmath=sse -msse2 -mstackrealign`.

## Limits

No full simulation rerun, case-study convergence study, macOS/Linux build,
or PDF reference-manual build was performed. Rd syntax, cross-references,
usage, examples and code/documentation consistency passed R CMD check.
The toolchain has no available AddressSanitizer runtime, so no sanitizer
claim is made. Header compilation, -O0 regression and repeated posterior
prediction provide the checks used here for the Node correction.

Existing API restrictions and inferential limits are documented in the
function help and README. Short regression chains establish interface and
numerical invariants; they do not establish MCMC convergence or calibrated
credible intervals for substantive applications.
