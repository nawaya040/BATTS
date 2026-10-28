## Submission

This is the first submission of BATTS to CRAN.

## Test environments

* Local: Windows 11, R 4.5.0 (ucrt), Rtools45 (GCC 14.2.0)
* win-builder: R-devel and R-release (to be run before submission)
* macOS builder: R-release (to be run before submission)

## R CMD check results

Local `R CMD check --as-cran`: 0 errors | 0 warnings | 1 note

* checking CRAN incoming feasibility ... NOTE
  Maintainer: 'Naoki Awaya <nawaya@waseda.jp>'
  New submission

  This is a new submission.

## Notes for the CRAN team

* The method is described in Awaya, Xu, and Ma (2025)
  <doi:10.48550/arXiv.2508.03059>, cited in the Description field.
* The package compiles C++17 code linking to RcppArmadillo; OpenMP is not
  used.
