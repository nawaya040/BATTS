# Install one review source variant into a new, explicitly selected library.
arg <- function(name, default=NULL) {
  a <- commandArgs(TRUE); hit <- a[startsWith(a,paste0("--",name,"="))]
  if(length(hit)) substring(hit[[1]],nchar(name)+4L) else default
}
variant <- arg("variant","r2"); lib <- arg("lib-dir")
if(!variant %in% c("r1","r2") || is.null(lib)) stop("Use --variant=r1|r2 --lib-dir=NEW_PATH [--install-cran=true]")
if(dir.exists(lib)) stop("Choose a new library directory; R1 and R2 must remain separate")
script <- sub("^--file=","",grep("^--file=",commandArgs(FALSE),value=TRUE))
root <- normalizePath(file.path(dirname(script),".."),winslash="/",mustWork=TRUE)
source(file.path(root,"code/r2/scripts/release_guard.R"))
review_verify_release(file.path(root,"code/r2"))
dir.create(lib,recursive=TRUE);lib <- normalizePath(lib,winslash="/",mustWork=TRUE)
.libPaths(c(lib,.libPaths()))
deps <- c("Rcpp","RcppArmadillo","ada","rpart","mvtnorm","pracma","digest","matrixStats")
if(identical(arg("install-cran","false"),"true"))
  install.packages(deps,lib=lib,repos="https://cloud.r-project.org",dependencies=c("Depends","Imports","LinkingTo"))
missing <- deps[!vapply(deps,requireNamespace,logical(1),quietly=TRUE)]
if(length(missing)) stop("Missing dependencies: ",paste(missing,collapse=", "),". Install compatible versions and select a new library.")
pkg <- if(variant=="r2") file.path(root,"code/methods/ReviewPkg") else file.path(root,"reference/r1/code/methods/ReviewPkg")
sources <- c(pkg,file.path(root,"reference/r1/code/methods/densratio"))
for(src in sources) {
  # Compilation writes only to a temporary source copy.
  stage <- tempfile("review-source-");dir.create(stage)
  if(!file.copy(src,stage,recursive=TRUE)) stop("Source staging failed")
  install.packages(file.path(stage,basename(src)),repos=NULL,type="source",lib=lib)
}
expected <- if(variant=="r2") "0.0.0.9001" else "0.0.0.9000"
stopifnot(requireNamespace("ReviewPkg",quietly=TRUE),
          identical(as.character(packageVersion("ReviewPkg",lib.loc=lib)),expected),
          identical(normalizePath(find.package("ReviewPkg",lib.loc=lib),winslash="/"),paste0(lib,"/ReviewPkg")))
writeLines(c(paste0("variant=",variant),paste0("ReviewPkg=",expected),capture.output(sessionInfo())),file.path(lib,"REVIEW_INSTALLATION.txt"))
cat("Installed",variant,"ReviewPkg",expected,"into",lib,"\n")
