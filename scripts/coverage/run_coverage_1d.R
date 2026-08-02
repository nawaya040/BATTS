script_path <- normalizePath(
  sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]]),
  mustWork = TRUE
)
source(file.path(dirname(script_path), "coverage_common.R"), local = TRUE)

args <- coverage_parse_args(commandArgs(trailingOnly = TRUE))
mode <- coverage_mode(args)
seed <- coverage_int(coverage_arg(args, "seed", required = TRUE), "seed")
resume <- coverage_bool(coverage_arg(args, "resume", "false"), "resume")
output_root <- coverage_arg(args, "output-dir", required = TRUE)
batts_lib <- coverage_load_batts(coverage_arg(args, "batts-lib", required = TRUE))
project_root <- coverage_project_root(script_path)
coverage_require_clean_git(project_root, mode)

n0 <- coverage_int(coverage_arg(args, "n0", if (mode == "smoke") "80" else NULL,
                                required = mode == "canonical"), "n0")
n1 <- coverage_int(coverage_arg(args, "n1", if (mode == "smoke") "80" else NULL,
                                required = mode == "canonical"), "n1")
if (mode == "canonical") {
  allowed <- c("500_500", "100_900", "2500_2500", "500_4500")
  if (!paste(n0, n1, sep = "_") %in% allowed) {
    stop("Unsupported canonical 1D sample-size setting")
  }
}

settings <- coverage_bart_settings(mode, "1d")
job_id <- sprintf("coverage_1d_n0-%d_n1-%d_seed-%03d", n0, n1, seed)
config <- c(list(n0 = n0, n1 = n1, mode = mode), settings)
source_files <- c(script_path, file.path(dirname(script_path), "coverage_common.R"))
metadata <- coverage_metadata(
  mode = mode,
  family = "1d",
  job_id = job_id,
  seed = seed,
  config = config,
  project_root = project_root,
  source_files = source_files,
  batts_lib = batts_lib
)
paths <- coverage_output_paths(output_root, mode, job_id)
if (coverage_resume_state(paths, resume, metadata)) {
  message("Verified existing output; skipping: ", paths$result)
  quit(save = "no", status = 0L)
}

set.seed(seed)
data <- matrix(
  c(stats::rnorm(n0, 0, 1), stats::rnorm(n1, 1, 1.5)),
  ncol = 1L
)
group_labels <- c(rep(0L, n0), rep(1L, n1))
truth <- as.numeric(
  stats::dnorm(data, 0, 1, log = TRUE) -
    stats::dnorm(data, 1, 1.5, log = TRUE)
)

result <- coverage_fit(
  data = data,
  group_labels = group_labels,
  truth = truth,
  settings = settings,
  reset_seed = NULL,
  output_ensembles = identical(mode, "canonical") && seed == 1L
)
metadata$timing <- result$timing
metadata$output_dimensions <- list(
  observations = result$n_observations,
  posterior_draws = result$n_draws,
  nominal_levels = length(result$nominal_mass)
)
coverage_atomic_save(result, metadata, paths)
message("Completed: ", paths$result)
