#!/usr/bin/env Rscript
# Use the submitted estimator body with an explicitly selected review source and the exact
# plotted simulation seed. Outputs are demonstrations, not paper estimates.

arg <- function(name, default = NULL) {
  x <- commandArgs(trailingOnly = TRUE)
  prefix <- paste0("--", name, "=")
  hit <- x[startsWith(x, prefix)]
  if (length(hit)) substring(hit[[1L]], nchar(prefix) + 1L) else default
}
required <- c("family", "scenario", "n0", "n1", "seed", "output-dir", "lib-dir")
missing <- required[vapply(required, function(x) is.null(arg(x)), logical(1))]
if (length(missing)) stop("Missing arguments: ", paste(missing, collapse = ", "))
family <- arg("family")
if (!family %in% c("2d", "20d")) stop("--family must be 2d or 20d")
n0 <- as.integer(arg("n0")); n1 <- as.integer(arg("n1")); seed <- as.integer(arg("seed"))
if (anyNA(c(n0, n1, seed)) || any(c(n0, n1, seed) < 1L)) stop("Invalid sample sizes or seed")
out <- arg("output-dir")
if (file.exists(out)) stop("Refusing to overwrite an existing output directory")
lib <- normalizePath(arg("lib-dir"), winslash = "/", mustWork = TRUE)
.libPaths(c(lib, .libPaths()))
variant <- arg("source-variant", "r2")
if (!variant %in% c("r1", "r2")) stop("--source-variant must be r1 or r2")
expected_version <- if (variant == "r2") "0.0.0.9001" else "0.0.0.9000"
if (!file.exists(file.path(lib,"ReviewPkg","DESCRIPTION")) ||
    !identical(as.character(utils::packageVersion("ReviewPkg",lib.loc=lib)),expected_version))
  stop("Selected library does not contain the requested ReviewPkg source variant")
required_packages <- c("ReviewPkg", "densratio", "mvtnorm", "ada")
if (family == "20d") required_packages <- c(required_packages, "pracma")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages)) stop("Missing R packages: ", paste(missing_packages, collapse = ", "))

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_arg) != 1L) stop("Run with Rscript")
script_path <- normalizePath(sub("^--file=", "", script_arg), winslash = "/", mustWork = TRUE)
root <- normalizePath(file.path(dirname(script_path), "../.."), winslash = "/", mustWork = TRUE)
original <- file.path(root, "reference", "r1", "code", "scripts")
section <- if (family == "2d") "section41_2d" else "section42_multi"
source(file.path(original, paste0(section, "_common.R")), local = TRUE)
source(file.path(original, paste0(section, "_models.R")), local = TRUE)
source(file.path(original, "section41_2d_utilities.R"), local = TRUE)

# Evaluate only named function definitions from the archived runner. This
# preserves the original estimator body while bypassing its all-cell launcher.
definitions <- parse(file = file.path(original, paste0("run_", section, ".R")))
needed <- c("compute_sq_error", "compute_calibrated_ada", "run_single_setting_repeat", "save_single_result")
for (expression in definitions) {
  if (is.call(expression) && identical(expression[[1L]], as.name("<-")) &&
      is.symbol(expression[[2L]]) && as.character(expression[[2L]]) %in% needed) {
    eval(expression)
  }
}
if (!all(vapply(needed, function(x) exists(x, envir = .GlobalEnv, mode = "function", inherits = FALSE), logical(1)))) {
  stop("Archived runner function extraction failed")
}

if (family == "2d") {
  config <- get_section41_config("light")
  settings <- section41_full_settings
} else {
  config <- get_section42_config("light", transformed = FALSE)
  settings <- get_section42_settings(FALSE)
}
matches <- which(vapply(settings, function(x) {
  identical(x[[1L]], arg("scenario")) && identical(x[[2L]], n0) && identical(x[[3L]], n1)
}, logical(1)))
if (length(matches) != 1L) stop("The requested setting is outside the submitted simulation grid")
config$settings <- list(settings[[matches]])
config$figure_index_repeat <- seed
config$figure_index_settings <- 1L
config$n_repeat <- 1L
config$repeat_ids <- seed
config$release_light_only <- TRUE
config$source <- paste("Submitted runner function; R1 light hyperparameters; method source",variant)

dir.create(out, recursive = TRUE, showWarnings = FALSE)
result <- run_single_setting_repeat(seed, 1L, config)
save_single_result(result, out, config$lambda_0)
dput(config, file = file.path(out, "config.dput"))
writeLines(c(
  "SMOKE_NOT_FOR_PAPER=true", paste0("source_variant=",variant),
  paste0("family=", family), paste0("scenario=", arg("scenario")),
  paste0("n0=", n0), paste0("n1=", n1), paste0("seed=", seed),
  paste0("R=", R.version.string),
  vapply(required_packages, function(x) paste0(x, "=", as.character(packageVersion(x))), character(1))
), file.path(out, "RUN_METADATA.txt"))
cat("Light demonstration completed: ", normalizePath(out, winslash = "/"), "\n", sep = "")
