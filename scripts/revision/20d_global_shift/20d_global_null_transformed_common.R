source_local_common <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg) != 1L) {
    stop("Unable to determine the canonical script path")
  }
  script_path <- normalizePath(
    sub("^--file=", "", file_arg),
    winslash = "/",
    mustWork = TRUE
  )
  common_path <- file.path(dirname(script_path), "20d_global_shift_common.R")
  source(common_path, local = globalenv())
  common_path
}
canonical_result_stem <- function(scenario, n0, n1, seed) {
  sprintf(
    "%s_n0_%05d_n1_%05d_seed_%02d",
    scenario,
    as.integer(n0),
    as.integer(n1),
    as.integer(seed)
  )
}

atomic_save_rds <- function(object, final_path, compress = "gzip") {
  temporary_path <- tempfile(
    pattern = paste0(basename(final_path), ".tmp-"),
    tmpdir = dirname(final_path)
  )
  on.exit(unlink(temporary_path), add = TRUE)
  saveRDS(object, temporary_path, compress = compress)
  if (!file.rename(temporary_path, final_path)) {
    stop("Atomic rename failed: ", final_path)
  }
  on.exit(NULL, add = FALSE)
  invisible(final_path)
}

canonical_task_complete <- function(task, output_dir, contract_hash) {
  stem <- canonical_result_stem(task$scenario, task$n0, task$n1, task$seed)
  result_path <- file.path(output_dir, paste0(stem, ".rds"))
  sidecar_path <- file.path(output_dir, paste0(stem, ".done.rds"))
  if (!file.exists(result_path) || !file.exists(sidecar_path)) {
    return(NULL)
  }
  sidecar <- tryCatch(readRDS(sidecar_path), error = function(e) NULL)
  if (is.null(sidecar) ||
      !identical(sidecar$schema_version, "1.0") ||
      !identical(sidecar$contract_hash, contract_hash) ||
      !identical(sidecar$scenario, task$scenario) ||
      !identical(sidecar$n0, task$n0) ||
      !identical(sidecar$n1, task$n1) ||
      !identical(sidecar$seed, task$seed)) {
    return(NULL)
  }
  observed_hash <- sha256_file(result_path)
  if (!identical(observed_hash, sidecar$result_sha256)) {
    stop("Checksum mismatch for completed result: ", result_path)
  }
  sidecar
}

run_transformed_canonical_task <- function(task, output_dir, contract_hash, source_paths,
                               code_commit) {
  assert_required_packages()
  batts_description <- assert_batts_version()

  completed <- canonical_task_complete(task, output_dir, contract_hash)
  if (!is.null(completed)) {
    return(data.frame(
      status = "skipped_verified",
      scenario = task$scenario,
      n0 = task$n0,
      n1 = task$n1,
      seed = task$seed,
      elapsed_seconds = completed$elapsed_seconds,
      result_path = completed$result_path,
      result_sha256 = completed$result_sha256,
      warnings = paste(completed$warnings, collapse = " | "),
      stringsAsFactors = FALSE
    ))
  }

  stem <- canonical_result_stem(task$scenario, task$n0, task$n1, task$seed)
  result_path <- file.path(output_dir, paste0(stem, ".rds"))
  sidecar_path <- file.path(output_dir, paste0(stem, ".done.rds"))
  if (file.exists(result_path) || file.exists(sidecar_path)) {
    stop("Incomplete or invalid pre-existing task output: ", stem)
  }

  if (!isTRUE(task$transformed)) {
    stop("The transformed canonical task requires transformed = TRUE")
  }
  simulation <- generate_global_null_20d(task)
  truth_error <- validate_global_null_simulation(simulation)
  data_hash <- global_null_data_hash(simulation)

  warnings <- character()
  set_reproducible_rng(task$seed)
  start_time <- Sys.time()
  fit <- withCallingHandlers(
    BATTS::batts(
      data = simulation$data,
      group_labels = simulation$group_labels,
      num_trees = task$num_trees,
      max_resol = 0,
      learn_rate = 0.01,
      n_bins = 100,
      alpha_cutpoint = 1,
      n_min_obs_per_node = 1,
      n_ratio_per_node = 1e-100,
      margin_scale = 0.1,
      use_gradient = FALSE,
      size_burnin = task$size_burnin,
      size_backfitting = task$size_backfitting,
      thin = 1,
      prob_moves = c(1 / 3, 1 / 3, 1 / 3),
      lambda_0 = 5,
      lambda_prior_parameters = c(1, 1),
      omega_prior_parameters = c(1, 1),
      update_lambda = FALSE,
      tree_priors = c(0.95, 2),
      output_BART_ensembles = FALSE,
      quiet = TRUE
    ),
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  fit_end_time <- Sys.time()

  balance_draws <- fit$balance_weight_BART_data
  expected_dimension <- c(task$n0 + task$n1, task$size_backfitting)
  if (!is.matrix(balance_draws) || !identical(dim(balance_draws), expected_dimension)) {
    stop(
      "Unexpected posterior dimensions; expected ",
      paste(expected_dimension, collapse = "x"),
      "; observed ", paste(dim(balance_draws), collapse = "x")
    )
  }
  if (any(!is.finite(balance_draws)) || any(balance_draws <= 0)) {
    stop("Non-positive or non-finite posterior balancing weights")
  }
  posterior_log_ratio_draws <- 2 * log(balance_draws)
  rm(balance_draws, fit)

  posterior_mean <- rowMeans(posterior_log_ratio_draws)
  intervals <- matrixStats::rowQuantiles(
    posterior_log_ratio_draws,
    probs = c(0.025, 0.975),
    na.rm = FALSE,
    drop = FALSE
  )
  lower <- intervals[, 1L]
  upper <- intervals[, 2L]
  calibration_curve <- compute_calibration_curve(
    posterior_log_ratio_draws,
    simulation$true_log_ratio,
    simulation$group_labels,
    task$calibration_levels
  )
  metrics <- compute_metrics(
    simulation$true_log_ratio,
    posterior_mean,
    lower,
    upper,
    simulation$group_labels
  )
  binned_metrics <- compute_binned_metrics(
    simulation$true_log_ratio,
    posterior_mean,
    lower,
    upper
  )

  if (any(!is.finite(posterior_log_ratio_draws)) ||
      any(!is.finite(calibration_curve$empirical_coverage)) ||
      any(lower > upper)) {
    stop("Invalid posterior summaries")
  }
  curve_95 <- calibration_curve$empirical_coverage[
    calibration_curve$group == "all" &
      abs(calibration_curve$nominal_level - 0.95) < 1e-12
  ]
  stored_95 <- metrics$coverage_95[metrics$group == "all"]
  if (length(curve_95) != 1L || !identical(curve_95, stored_95)) {
    stop("95% calibration point does not match direct coverage")
  }

  source_hashes <- data.frame(
    path = normalizePath(source_paths, winslash = "/", mustWork = TRUE),
    sha256 = vapply(source_paths, sha256_file, character(1)),
    stringsAsFactors = FALSE
  )
  elapsed_seconds <- as.numeric(difftime(fit_end_time, start_time, units = "secs"))
  result <- list(
    schema_version = "1.0",
    scenario = task$scenario,
    task = task,
    posterior_log_ratio_draws = posterior_log_ratio_draws,
    pointwise = list(
      group_labels = simulation$group_labels,
      true_log_ratio = simulation$true_log_ratio,
      posterior_mean = posterior_mean,
      lower_95 = lower,
      upper_95 = upper
    ),
    metrics = metrics,
    binned_metrics = binned_metrics,
    calibration_curve = calibration_curve,
    loading = simulation$loading,
    signal_direction = simulation$signal_direction,
    transform = simulation$parameters$transform,
    data_hash = data_hash,
    validation = list(
      loading_gram_max_error = simulation$parameters$gram_error,
      analytic_truth_max_error = truth_error,
      posterior_dimension = dim(posterior_log_ratio_draws),
      calibration_95_matches_direct = TRUE,
      transform_preserves_truth = TRUE,
      transformed_data_finite = TRUE
    ),
    provenance = list(
      contract_hash = contract_hash,
      code_commit = code_commit,
      batts_version = batts_description[["Version"]],
      batts_remote_sha = batts_description[["RemoteSha"]],
      batts_built = batts_description[["Built"]],
      r_version = R.version.string,
      platform = R.version$platform,
      os = Sys.info(),
      rng_kind = RNGkind(),
      parallel_backend = "independent PSOCK worker; one seed per task",
      omp_num_threads = Sys.getenv("OMP_NUM_THREADS", unset = NA_character_),
      openblas_num_threads = Sys.getenv("OPENBLAS_NUM_THREADS", unset = NA_character_),
      mkl_num_threads = Sys.getenv("MKL_NUM_THREADS", unset = NA_character_),
      start_time_utc = format(start_time, tz = "UTC", usetz = TRUE),
      end_time_utc = format(fit_end_time, tz = "UTC", usetz = TRUE),
      elapsed_seconds = elapsed_seconds,
      warnings = unique(warnings),
      source_hashes = source_hashes,
      session_info = capture.output(utils::sessionInfo())
    )
  )

  atomic_save_rds(result, result_path, compress = "gzip")
  result_hash <- sha256_file(result_path)
  sidecar <- list(
    schema_version = "1.0",
    contract_hash = contract_hash,
    scenario = task$scenario,
    n0 = task$n0,
    n1 = task$n1,
    seed = task$seed,
    result_path = normalizePath(result_path, winslash = "/", mustWork = TRUE),
    result_bytes = file.info(result_path)$size,
    result_sha256 = result_hash,
    elapsed_seconds = elapsed_seconds,
    warnings = unique(warnings),
    metrics = metrics,
    binned_metrics = binned_metrics,
    calibration_curve = calibration_curve,
    validation = result$validation,
    data_hash = data_hash,
    transform = simulation$parameters$transform
  )
  atomic_save_rds(sidecar, sidecar_path, compress = "gzip")

  data.frame(
    status = "completed",
    scenario = task$scenario,
    n0 = task$n0,
    n1 = task$n1,
    seed = task$seed,
    elapsed_seconds = elapsed_seconds,
    result_path = sidecar$result_path,
    result_sha256 = result_hash,
    warnings = paste(unique(warnings), collapse = " | "),
    stringsAsFactors = FALSE
  )
}
