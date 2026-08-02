script_path <- normalizePath(
  sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]]),
  mustWork = TRUE
)
source(file.path(dirname(script_path), "coverage_common.R"), local = TRUE)
model_file <- file.path(dirname(script_path), "models", "section42_multi_models.R")
source(model_file, local = TRUE)

args <- coverage_parse_args(commandArgs(trailingOnly = TRUE))
mode <- coverage_mode(args)
seed <- coverage_int(coverage_arg(args, "seed", required = TRUE), "seed")
resume <- coverage_bool(coverage_arg(args, "resume", "false"), "resume")
output_root <- coverage_arg(args, "output-dir", required = TRUE)
batts_lib <- coverage_load_batts(coverage_arg(args, "batts-lib", required = TRUE))
project_root <- coverage_project_root(script_path)
coverage_require_clean_git(project_root, mode)
scenario <- coverage_arg(args, "scenario", if (mode == "smoke") "latent_dispersion" else NULL,
                         required = mode == "canonical")
if (!scenario %in% c("latent_location_shift", "latent_dispersion")) {
  stop("Unsupported 20D scenario")
}
n0 <- coverage_int(coverage_arg(args, "n0", if (mode == "smoke") "80" else NULL,
                                required = mode == "canonical"), "n0")
n1 <- coverage_int(coverage_arg(args, "n1", if (mode == "smoke") "80" else NULL,
                                required = mode == "canonical"), "n1")
transformed <- coverage_bool(coverage_arg(args, "transformed", "false"), "transformed")
if (mode == "canonical" && !paste(n0, n1, sep = "_") %in% c("5000_5000", "9000_1000")) {
  stop("Unsupported canonical 20D sample-size setting")
}

settings <- coverage_bart_settings(mode, "20d")
d <- if (mode == "canonical") 20L else 5L
job_id <- sprintf(
  "coverage_20d_%s_n0-%d_n1-%d_transformed-%s_seed-%03d",
  scenario,
  n0,
  n1,
  tolower(as.character(transformed)),
  seed
)
config <- c(list(
  scenario = scenario,
  n0 = n0,
  n1 = n1,
  d = d,
  unif_w = 0.2,
  transformed = transformed,
  mode = mode
), settings)
source_files <- c(
  script_path,
  file.path(dirname(script_path), "coverage_common.R"),
  model_file
)
metadata <- coverage_metadata(
  mode = mode,
  family = "20d",
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
generated <- simulation_multi_latent(
  n0 = n0,
  n1 = n1,
  d = d,
  scenario = scenario,
  unif_w = 0.2,
  transform = transformed
)
result <- coverage_fit(
  data = generated$data,
  group_labels = generated$group_labels,
  truth = generated$true_log_w_obs,
  settings = settings,
  reset_seed = seed,
  output_ensembles = FALSE
)
metadata$timing <- result$timing
metadata$output_dimensions <- list(
  observations = result$n_observations,
  posterior_draws = result$n_draws,
  nominal_levels = length(result$nominal_mass)
)
coverage_atomic_save(result, metadata, paths)
message("Completed: ", paths$result)
