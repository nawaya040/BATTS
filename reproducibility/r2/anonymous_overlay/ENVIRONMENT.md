# Anonymous source and software environment

The proposed method is supplied as ReviewPkg 0.0.0.9001 in
`code/methods/ReviewPkg/`. It corresponds to the Round 2 computation source,
with anonymous package/API names and the same numerical implementation and
compiler options. Its `Review-Source-ID` field is an anonymous source identifier.
The unchanged historical ReviewPkg 0.0.0.9000 and corrected densratio 0.2.1
remain in `reference/r1/code/methods/`. Install R1 and R2 into different libraries.
Use the setup commands in README; no public method repository is required.

Local checks use R 4.5.2 on Windows 11 x64 and Rtools45 / GCC 14.3.0, with
Rcpp 1.1.1-1 and RcppArmadillo 14.4.1-1. The anonymous and original R2 source
packages were built under the same conditions and compared in separate R
processes. Twelve cases cover both boosting losses with cross-validation,
Bayesian fits, posterior samples, prediction, and RNG state in 2D and 20D.
All compared objects and RNG states were exactly identical.

Drawing versions used in the current checks are ggplot2 4.0.3, patchwork 1.3.2,
scales 1.4.0, mvtnorm 1.3-3, and pracma 2.4.6. Light execution also uses ada
2.0-5 and the supplied corrected densratio. R2 workflows additionally use digest,
matrixStats, and rpart. The technical table program uses Python 3.14.4 and only
the Python standard library. The Section 5 Python environment requires coauthor
confirmation. Exact records of the checked packages are in verification/.

These checks use a new method library with dependencies resolved from the
existing local environment. A completely clean dependency installation, a
locked dependency environment, and complete resource profiling remain open.
The optional CRAN setup route is not a dependency lockfile.

Anonymous release file hashes are in SOURCE_MANIFEST.csv. The mapping to
original identities is retained privately. Saved numerical values and input
samples are preserved; seven RDS files have only identity metadata adapted.
Full canonical results are not recomputed by anonymization.
