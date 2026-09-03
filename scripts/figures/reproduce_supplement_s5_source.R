#!/usr/bin/env Rscript

# Reproduce the two source graphics used in Supplementary Figure S5.
# This script only regenerates the fixed-seed simulated data and renders it;
# it does not fit or tune an estimator.

parse_args <- function(args) {
  result <- list()
  for (arg in args) {
    if (!startsWith(arg, "--") || !grepl("=", arg, fixed = TRUE)) {
      stop("Arguments must have the form --name=value: ", arg)
    }
    pair <- strsplit(substring(arg, 3L), "=", fixed = TRUE)[[1L]]
    result[[pair[[1L]]]] <- paste(pair[-1L], collapse = "=")
  }
  result
}

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_arg)) stop("This script must be run with Rscript")
script_path <- normalizePath(
  sub("^--file=", "", script_arg[[1L]]), winslash = "/", mustWork = TRUE
)
repo_root <- normalizePath(
  file.path(dirname(script_path), "..", ".."), winslash = "/", mustWork = TRUE
)
args <- parse_args(commandArgs(trailingOnly = TRUE))

model_path <- file.path(
  repo_root, "scripts", "coverage", "models", "section42_multi_models.R"
)
output_dir <- if (is.null(args[["output-dir"]])) {
  file.path(repo_root, "output", "figures")
} else {
  args[["output-dir"]]
}
output_x12 <- file.path(output_dir, "supplement_s5_source_x12.png")
output_x34 <- file.path(output_dir, "supplement_s5_source_x34.png")
metadata_path <- file.path(output_dir, "supplement_s5_source_metadata.txt")

if (!file.exists(model_path)) stop("Missing 20D simulation model: ", model_path)
for (package in c("mvtnorm", "pracma")) {
  if (!requireNamespace(package, quietly = TRUE)) stop("Missing package: ", package)
}
expected_model_hash <- "ad2090da5fbc01123e65687b1364a6064f2e5b3b17824ad3b9c4df7152e6eb40"
model_hash <- tolower(unname(tools::sha256sum(model_path)))
if (!identical(model_hash, expected_model_hash)) {
  stop("20D model source hash differs from the verified revision source")
}

model_environment <- new.env(parent = globalenv())
sys.source(model_path, envir = model_environment)

seed <- 2L
n0 <- 500L
n1 <- 500L
dimension <- 20L
scenario <- "latent_location_shift"
uniform_weight <- 0.2
transformed <- TRUE

set.seed(seed)
generated <- model_environment$simulation_multi_latent(
  n0 = n0,
  n1 = n1,
  d = dimension,
  scenario = scenario,
  unif_w = uniform_weight,
  transform = transformed
)
data <- as.matrix(generated$data)
labels <- as.integer(generated$group_labels)

if (!identical(dim(data), c(n0 + n1, dimension))) {
  stop("Generated S5 data have unexpected dimensions")
}
if (!identical(as.integer(table(labels)), c(n0, n1))) {
  stop("Generated S5 data have unexpected group sizes")
}
if (any(!is.finite(data)) || any(data < 0) || any(data > 1)) {
  stop("Generated transformed S5 data are outside the expected finite range")
}

group0 <- labels == 0L
group1 <- labels == 1L
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

render_pair <- function(first_dimension, second_dimension, output_path) {
  grDevices::png(
    filename = output_path, width = 1200, height = 1200, res = 200,
    pointsize = 12, bg = "white"
  )
  old_par <- graphics::par(no.readonly = TRUE)
  on.exit({
    graphics::par(old_par)
    grDevices::dev.off()
  }, add = TRUE)
  graphics::par(mar = c(4.8, 4.8, 1.0, 1.0))
  graphics::plot(
    data[group0, first_dimension],
    data[group0, second_dimension],
    xlab = paste0("X", first_dimension),
    ylab = paste0("X", second_dimension),
    xlim = c(0, 0.5),
    ylim = c(0, 0.5)
  )
  graphics::points(
    data[group1, first_dimension],
    data[group1, second_dimension],
    col = "red"
  )
}

render_pair(1L, 2L, output_x12)
render_pair(3L, 4L, output_x34)

writeLines(c(
  "figure=Supplementary Figure S5 source reproduction",
  paste0("scenario=", scenario),
  paste0("n0=", n0),
  paste0("n1=", n1),
  paste0("dimension=", dimension),
  paste0("uniform_weight=", uniform_weight),
  paste0("transformed=", tolower(as.character(transformed))),
  paste0("seed=", seed),
  paste0("model_sha256=", model_hash),
  "estimator_fitting=false",
  "group0=black_open_circles",
  "group1=red_open_circles",
  "axis_limits=0,0.5",
  paste0("x12_png=", normalizePath(output_x12, winslash = "/", mustWork = TRUE)),
  paste0("x34_png=", normalizePath(output_x34, winslash = "/", mustWork = TRUE)),
  paste0("x12_sha256=", tolower(unname(tools::sha256sum(output_x12)))),
  paste0("x34_sha256=", tolower(unname(tools::sha256sum(output_x34))))
), metadata_path, useBytes = TRUE)

cat("Saved S5 source X1/X2 PNG: ", normalizePath(output_x12), "\n", sep = "")
cat("Saved S5 source X3/X4 PNG: ", normalizePath(output_x34), "\n", sep = "")
cat("Saved metadata: ", normalizePath(metadata_path), "\n", sep = "")
