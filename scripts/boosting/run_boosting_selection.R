options(stringsAsFactors = FALSE)

script_path <- normalizePath(
  sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]]),
  mustWork = TRUE
)
script_dir <- dirname(script_path)
source(file.path(script_dir, "boosting_selection_common.R"), local = TRUE)

args <- boosting_parse_args(commandArgs(trailingOnly = TRUE))
mode <- boosting_mode(args)
family <- boosting_arg(args, "family", if (mode == "smoke") "2d" else NULL,
                       required = mode == "canonical")
scenario <- boosting_arg(
  args, "scenario", if (mode == "smoke") "local_shift" else NULL,
  required = mode == "canonical"
)
seed <- boosting_int(boosting_arg(args, "seed", required = TRUE), "seed")
resume <- boosting_bool(boosting_arg(args, "resume", "false"), "resume")
transformed <- boosting_bool(
  boosting_arg(args, "transformed", "false"), "transformed"
)
output_root <- boosting_arg(args, "output-dir", required = TRUE)
r_lib <- boosting_arg(args, "r-lib", required = TRUE)

if (!family %in% c("2d", "20d")) {
  stop("--family must be 2d or 20d")
}
if (identical(family, "2d")) {
  if (!scenario %in% c("global_shift", "local_shift", "local_dispersion")) {
    stop("Unsupported 2D scenario")
  }
  if (isTRUE(transformed)) {
    stop("--transformed=true is only supported for 20D")
  }
} else if (!scenario %in% c("latent_location_shift", "latent_dispersion")) {
  stop("Unsupported 20D scenario")
}

if (identical(mode, "canonical")) {
  n0 <- boosting_int(boosting_arg(args, "n0", required = TRUE), "n0")
  n1 <- boosting_int(boosting_arg(args, "n1", required = TRUE), "n1")
  if (!paste(n0, n1, sep = "_") %in% c("5000_5000", "9000_1000")) {
    stop("Unsupported canonical sample-size setting")
  }
  settings <- list(
    folds = 5L,
    max_trees = 1000L,
    min_trees_classification = 10L,
    learn_rate = 0.01,
    ada_depth = 4L,
    ada_bag_fraction = 0.5,
    proposed_depth = 4L
  )
} else {
  n0 <- boosting_int(boosting_arg(args, "n0", "80"), "n0")
  n1 <- boosting_int(boosting_arg(args, "n1", "80"), "n1")
  settings <- list(
    folds = 2L,
    max_trees = 60L,
    min_trees_classification = 5L,
    learn_rate = 0.01,
    ada_depth = 4L,
    ada_bag_fraction = 0.5,
    proposed_depth = 4L
  )
}

project_root <- boosting_project_root(script_path)
boosting_require_clean_git(project_root, mode)
r_lib <- boosting_load_packages(r_lib)

model_files <- c(
  file.path(project_root, "scripts", "coverage", "models", "section41_2d_models.R"),
  file.path(project_root, "scripts", "coverage", "models", "section42_multi_models.R")
)
model_environment <- new.env(parent = globalenv())
for (model_file in model_files) {
  sys.source(model_file, envir = model_environment)
}

generate_data <- function(n0, n1) {
  if (identical(family, "2d")) {
    return(model_environment$simulation_2d(
      n0 = n0, n1 = n1, scenario = scenario, n_grid_per_dim = 20L
    ))
  }
  model_environment$simulation_multi_latent(
    n0 = n0, n1 = n1, d = 20L, scenario = scenario,
    unif_w = 0.2, transform = transformed
  )
}

job_id <- sprintf(
  "boosting_selection_%s_%s_n0-%d_n1-%d_transformed-%s_seed-%03d",
  family, scenario, n0, n1, tolower(as.character(transformed)), seed
)
seed_map <- boosting_seed_map(seed)
config <- list(
  family = family,
  scenario = scenario,
  n0 = n0,
  n1 = n1,
  transformed = transformed,
  mode = mode,
  settings = settings,
  seed_map = seed_map,
  criterion_roles = list(
    classification_error = "legacy",
    exponential_loss = "primary",
    balancing_loss = "counterfactual_diagnostic"
  ),
  balancing_loss_definition = paste0(
    "mean_group0(exp(-log_ratio/2)) + ",
    "mean_group1(exp(log_ratio/2)); stored on log scale"
  )
)
source_files <- c(
  script_path,
  file.path(script_dir, "boosting_selection_common.R"),
  file.path(script_dir, "README.md"),
  model_files
)
metadata <- boosting_metadata(
  mode = mode,
  family = family,
  job_id = job_id,
  seed = seed,
  config = config,
  project_root = project_root,
  source_files = source_files,
  r_lib = r_lib
)
paths <- boosting_output_paths(output_root, mode, job_id)
if (boosting_resume_state(paths, resume, metadata)) {
  message("Verified existing output; skipping: ", paths$result)
  quit(save = "no", status = 0L)
}

started <- proc.time()
set.seed(seed_map$data)
training <- generate_data(n0, n1)
data <- as.matrix(training$data)
labels_train <- as.integer(training$group_labels)
truth_train <- as.numeric(training$true_log_w_obs)
if (nrow(data) != length(labels_train) || length(labels_train) != length(truth_train)) {
  stop("Training data, labels, and truth have incompatible dimensions")
}
if (!identical(sort(unique(labels_train)), 0:1)) {
  stop("Training labels must contain groups 0 and 1")
}

fold_id <- boosting_make_stratified_folds(
  labels_train, settings$folds, seed_map$folds
)
ada_cv <- boosting_compute_ada_cv(
  data = data,
  labels = labels_train,
  fold_id = fold_id,
  settings = settings,
  seed_base = seed_map$ada_cv_base
)
ada_selections <- boosting_select_ada_trees(
  ada_cv, settings$min_trees_classification, settings$max_trees
)
ada_fold_argmins <- boosting_fold_argmins(ada_cv)
ada_cross_loss <- boosting_cross_loss(ada_cv, ada_selections)
max_selected_ada <- max(ada_selections$final_selected_trees)
ada_model <- boosting_fit_ada_final(
  data, labels_train, max_selected_ada, settings, seed_map$ada_final
)

ada_estimates <- list(train = list())
ada_metric_rows <- list()
metric_index <- 1L
for (index in seq_len(nrow(ada_selections))) {
  criterion <- ada_selections$criterion[[index]]
  selected_trees <- ada_selections$final_selected_trees[[index]]
  train_estimate <- boosting_ada_log_ratio(
    ada_model, data, selected_trees, n0, n1
  )
  ada_estimates$train[[criterion]] <- train_estimate
  ada_metric_rows[[metric_index]] <- boosting_metric_row(
    "adaboost", criterion, "train",
    train_estimate, truth_train, labels_train
  )
  metric_index <- metric_index + 1L
}
ada_metrics <- do.call(rbind, ada_metric_rows)
ada_nodes <- boosting_ada_node_summary(ada_model, ada_selections)

# BATTS::boots() creates stratified folds before fitting. Resetting the same
# seed for GB and FS gives both methods the recorded fold allocation while
# keeping their fitted paths separate.
gb <- boosting_run_proposed(
  data = data,
  labels = labels_train,
  truth_train = truth_train,
  settings = settings,
  use_gradient = TRUE,
  seed = seed_map$folds,
  method_name = "gb"
)
fs <- boosting_run_proposed(
  data = data,
  labels = labels_train,
  truth_train = truth_train,
  settings = settings,
  use_gradient = FALSE,
  seed = seed_map$folds,
  method_name = "fs"
)
if (!identical(gb$diagnostics$fold_id, fold_id) ||
    !identical(fs$diagnostics$fold_id, fold_id)) {
  stop("Recorded folds differ across AdaBoost, GB, and FS")
}

timing <- proc.time() - started
metadata$timing <- c(
  user_seconds = unname(timing[["user.self"]]),
  system_seconds = unname(timing[["sys.self"]]),
  elapsed_seconds = unname(timing[["elapsed"]])
)
metadata$output_dimensions <- list(
  training_observations = nrow(data),
  predictors = ncol(data),
  folds = settings$folds,
  maximum_trees = settings$max_trees
)

result <- list(
  design = list(
    fold_id = fold_id,
    labels_train = labels_train,
    truth_train = truth_train
  ),
  adaboost = list(
    selections = ada_selections,
    fold_argmins = ada_fold_argmins,
    cross_loss = ada_cross_loss,
    cv_curves = ada_cv,
    node_summary = ada_nodes,
    estimates = ada_estimates,
    metrics = ada_metrics
  ),
  proposed = list(
    gb = gb$diagnostics,
    fs = fs$diagnostics,
    estimates = list(gb = gb$estimates, fs = fs$estimates),
    metrics = rbind(gb$metrics, fs$metrics)
  ),
  timing = metadata$timing
)

boosting_atomic_save(result, metadata, paths)
message("Completed: ", paths$result)
