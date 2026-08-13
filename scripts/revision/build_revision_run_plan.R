#!/usr/bin/env Rscript

parse_args <- function(args) {
  out <- list()
  for (arg in args) {
    if (!startsWith(arg, "--") || !grepl("=", arg, fixed = TRUE)) {
      stop("Arguments must use --name=value")
    }
    pair <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
    out[[pair[[1L]]]] <- paste(pair[-1L], collapse = "=")
  }
  out
}

args <- parse_args(commandArgs(TRUE))
if (is.null(args[["output-dir"]])) stop("Required: --output-dir=PATH")
output_dir <- normalizePath(args[["output-dir"]], winslash = "/", mustWork = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
if (!dir.exists(output_dir)) stop("Unable to create output directory")

rows <- list()
row_index <- 0L
add_row <- function(phase, job_id, units, script, family = "", scenario = "",
                    n0 = NA_integer_, n1 = NA_integer_, transformed = NA,
                    seed = NA_integer_, method = "", dependency_id = "") {
  row_index <<- row_index + 1L
  rows[[row_index]] <<- data.frame(
    phase = phase, job_id = job_id, scientific_units = as.integer(units),
    script = script, family = family, scenario = scenario,
    n0 = as.integer(n0), n1 = as.integer(n1), transformed = transformed,
    seed = as.integer(seed), method = method, dependency_id = dependency_id,
    stringsAsFactors = FALSE
  )
}

# One controlled runner manages 200 transformed BART tasks internally.
add_row(
  "bart_transformed", "bart_transformed_global_null_batch", 200L,
  "scripts/revision/20d_global_shift/run_20d_global_null_transformed_canonical.R",
  family = "20d", transformed = TRUE
)

sizes_20d <- list(c(5000L, 5000L), c(9000L, 1000L))
for (seed in 1:50) for (scenario in c("global_shift", "null")) {
  for (sizes in sizes_20d) for (transformed in c(FALSE, TRUE)) {
    suffix <- sprintf(
      "%s_n0-%d_n1-%d_transformed-%s_seed-%03d",
      scenario, sizes[[1L]], sizes[[2L]], tolower(as.character(transformed)), seed
    )
    boosting_id <- paste0("global_null_boosting_20d_", suffix)
    add_row(
      "boosting", boosting_id, 1L,
      "scripts/revision/20d_global_shift/run_20d_global_null_boosting_job.R",
      "20d", scenario, sizes[[1L]], sizes[[2L]], transformed, seed, "boosting"
    )
    for (method in c("kliep", "ulsif")) {
      add_row(
        "kernel", paste0("global_null_", method, "_", suffix), 1L,
        "scripts/revision/20d_global_shift/run_20d_global_null_kernel_job.R",
        "20d", scenario, sizes[[1L]], sizes[[2L]], transformed, seed, method
      )
    }
    add_row(
      "cdc", paste0("global_null_cdc_", suffix), 1L,
      "scripts/revision/20d_global_shift/run_20d_global_null_cdc_job.R",
      "20d", scenario, sizes[[1L]], sizes[[2L]], transformed, seed, "cdc",
      dependency_id = boosting_id
    )
  }
}

for (seed in 1:50) for (sizes in list(
  c(500L, 500L), c(100L, 900L), c(2500L, 2500L), c(500L, 4500L)
)) {
  add_row(
    "coverage", sprintf("coverage_1d_n0-%d_n1-%d_seed-%03d", sizes[[1L]], sizes[[2L]], seed),
    1L, "scripts/coverage/run_coverage_1d.R", "1d", "",
    sizes[[1L]], sizes[[2L]], FALSE, seed, "bart_coverage"
  )
}
for (seed in 1:50) for (scenario in c(
  "global_shift", "local_shift", "local_dispersion"
)) for (sizes in sizes_20d) {
  add_row(
    "coverage", sprintf("coverage_2d_%s_n0-%d_n1-%d_seed-%03d",
                         scenario, sizes[[1L]], sizes[[2L]], seed),
    1L, "scripts/coverage/run_coverage_2d.R", "2d", scenario,
    sizes[[1L]], sizes[[2L]], FALSE, seed, "bart_coverage"
  )
}
for (seed in 1:50) for (scenario in c(
  "latent_location_shift", "latent_dispersion"
)) for (sizes in sizes_20d) for (transformed in c(FALSE, TRUE)) {
  add_row(
    "coverage", sprintf("coverage_20d_%s_n0-%d_n1-%d_transformed-%s_seed-%03d",
                         scenario, sizes[[1L]], sizes[[2L]],
                         tolower(as.character(transformed)), seed),
    1L, "scripts/coverage/run_coverage_20d.R", "20d", scenario,
    sizes[[1L]], sizes[[2L]], transformed, seed, "bart_coverage"
  )
}

plan <- do.call(rbind, rows)
expected_rows <- c(bart_transformed = 1L, boosting = 400L, kernel = 800L,
                   cdc = 400L, coverage = 900L)
observed_rows <- table(factor(plan$phase, levels = names(expected_rows)))
if (!identical(as.integer(observed_rows), unname(expected_rows))) {
  stop("Run-plan phase counts are invalid")
}
if (anyDuplicated(plan$job_id)) stop("Duplicate job IDs in run plan")
if (sum(plan$scientific_units) != 2700L) stop("Expected 2700 scientific units")
boosting_ids <- plan$job_id[plan$phase == "boosting"]
if (!all(plan$dependency_id[plan$phase == "cdc"] %in% boosting_ids)) {
  stop("CDC dependency grid is incomplete")
}

plan_path <- file.path(output_dir, "revision_run_plan.csv")
metadata_path <- file.path(output_dir, "revision_run_plan.rds")
hash_path <- file.path(output_dir, "revision_run_plan.sha256")
if (any(file.exists(c(plan_path, metadata_path, hash_path)))) {
  stop("Refusing to overwrite an existing run plan")
}
utils::write.csv(plan, plan_path, row.names = FALSE, na = "")
plan_hash <- unname(tools::sha256sum(plan_path))
saveRDS(list(
  schema_version = "1.0", created_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  plan_sha256 = plan_hash, rows = nrow(plan), scientific_units = sum(plan$scientific_units),
  phase_rows = as.list(stats::setNames(as.integer(observed_rows), names(expected_rows)))
), metadata_path, compress = "xz")
writeLines(plan_hash, hash_path, useBytes = TRUE)
cat(sprintf("Created %d scheduler rows for %d scientific units\n",
            nrow(plan), sum(plan$scientific_units)))
