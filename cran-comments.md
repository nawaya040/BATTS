## Submission

This is the first submission of BATTS to CRAN.

## Test environments

* Local: Windows 11, R 4.5.0 (ucrt), Rtools45 (GCC 14.2.0)
* Local: Windows 11, R-devel (2026-10-26 r90766 ucrt), Rtools45 (GCC 14.3.0)
* win-builder: R-release 4.6.1 (ucrt) and R-oldrelease 4.5.3 (ucrt)
* R-hub (GitHub Actions): Linux R-devel and R-release, Windows R-devel,
  macOS R-devel (x86_64 and arm64), gcc and clang AddressSanitizer,
  clang UndefinedBehaviorSanitizer, and valgrind

## R CMD check results

0 errors | 0 warnings | 1 note

* checking CRAN incoming feasibility ... NOTE
  Maintainer: 'Naoki Awaya <nawaya@waseda.jp>'
  New submission
  Possibly misspelled words in DESCRIPTION: Awaya (10:5), Xu (10:12)

  This is a new submission. The flagged words are the surnames of the
  authors of the cited paper, Awaya, Xu, and Ma (2025), and are spelled
  correctly.

The sanitizer and valgrind runs reported no errors and no leaks.

## Notes for the CRAN team

* The method is described in Awaya, Xu, and Ma (2025)
  <doi:10.48550/arXiv.2508.03059>, cited in the Description field.
* The package compiles C++17 code linking to RcppArmadillo; OpenMP is not
  used.
