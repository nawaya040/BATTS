EXPECTED_BATTS_SHA <- "6f625bad83702b36e5480be1ed1343258a9b075a"

required_packages <- c("BATTS", "digest", "matrixStats", "mvtnorm", "pracma")

assert_required_packages <- function() {
  missing <- required_packages[
    !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
  ]
  if (length(missing) > 0L) {
    stop("Missing required packages: ", paste(missing, collapse = ", "))
  }
}

assert_batts_version <- function(expected_sha = EXPECTED_BATTS_SHA) {
  desc <- utils::packageDescription("BATTS")
  observed_sha <- desc[["RemoteSha"]]
  if (is.null(observed_sha) || !identical(tolower(observed_sha), tolower(expected_sha))) {
    stop(
      "BATTS RemoteSha mismatch. Expected ", expected_sha,
      "; observed ", if (is.null(observed_sha)) "<missing>" else observed_sha
    )
  }
  invisible(desc)
}

set_reproducible_rng <- function(seed) {
  RNGkind("Mersenne-Twister", "Inversion", "Rejection")
  set.seed(as.integer(seed))
}

sha256_file <- function(path) {
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}

generate_global_shift_20d <- function(
    n0,
    n1,
    seed,
    d = 20L,
    latent_dim = 4L,
    shift = 1,
    noise_sd = 0.1) {
  stopifnot(
    n0 > 0L,
    n1 > 0L,
    d >= latent_dim,
    latent_dim >= 1L,
    shift >= 0,
    noise_sd > 0
  )

  set_reproducible_rng(seed)

  q_full <- pracma::randortho(d)
  loading <- q_full[, seq_len(latent_dim), drop = FALSE]

  e1 <- numeric(latent_dim)
  e1[1L] <- 1
  mean0 <- -0.5 * shift * e1
  mean1 <- 0.5 * shift * e1

  z0 <- matrix(stats::rnorm(n0 * latent_dim), nrow = n0, ncol = latent_dim)
  z1 <- matrix(stats::rnorm(n1 * latent_dim), nrow = n1, ncol = latent_dim)
  z0 <- sweep(z0, 2L, mean0, "+")
  z1 <- sweep(z1, 2L, mean1, "+")
  latent <- rbind(z0, z1)

  noise <- matrix(stats::rnorm((n0 + n1) * d, sd = noise_sd),
                  nrow = n0 + n1, ncol = d)
  data <- latent %*% t(loading) + noise
  group_labels <- c(rep.int(0L, n0), rep.int(1L, n1))

  signal_direction <- loading[, 1L]
  true_log_ratio <- -shift * drop(data %*% signal_direction) /
    (1 + noise_sd^2)

  gram_error <- max(abs(crossprod(loading) - diag(latent_dim)))
  if (!is.finite(gram_error) || gram_error > 1e-10) {
    stop("Loading matrix is not orthonormal; maximum error = ", gram_error)
  }
  if (!all(is.finite(data)) || !all(is.finite(true_log_ratio))) {
    stop("Non-finite generated values")
  }
  if (!identical(
    tabulate(group_labels + 1L, nbins = 2L),
    as.integer(c(n0, n1))
  )) {
    stop("Unexpected group-label counts")
  }

  list(
    data = data,
    latent = latent,
    group_labels = group_labels,
    true_log_ratio = true_log_ratio,
    loading = loading,
    signal_direction = signal_direction,
    parameters = list(
      n0 = n0,
      n1 = n1,
      seed = seed,
      d = d,
      latent_dim = latent_dim,
      shift = shift,
      noise_sd = noise_sd,
      gram_error = gram_error
    )
  )
}

validate_analytic_truth <- function(simulation, tolerance = 1e-8) {
  p <- simulation$parameters
  validation_indices <- seq_len(min(25L, nrow(simulation$data)))
  loading <- simulation$loading
  covariance <- loading %*% t(loading) + p$noise_sd^2 * diag(p$d)
  mean0 <- -0.5 * p$shift * simulation$signal_direction
  mean1 <- 0.5 * p$shift * simulation$signal_direction
  x <- simulation$data[validation_indices, , drop = FALSE]

  reference <- mvtnorm::dmvnorm(x, mean = mean0, sigma = covariance, log = TRUE) -
    mvtnorm::dmvnorm(x, mean = mean1, sigma = covariance, log = TRUE)
  observed <- simulation$true_log_ratio[validation_indices]
  max_error <- max(abs(reference - observed))
  if (!is.finite(max_error) || max_error > tolerance) {
    stop("Analytic truth validation failed; maximum error = ", max_error)
  }
  max_error
}

metric_row <- function(group, truth, posterior_mean, lower, upper) {
  covered <- lower <= truth & truth <= upper
  zero_excluded <- lower > 0 | upper < 0
  nonzero <- truth != 0
  toward_zero_miss <- !covered & nonzero & (
    (truth > 0 & upper < truth) |
      (truth < 0 & lower > truth)
  )

  data.frame(
    group = group,
    n = length(truth),
    mse = mean((posterior_mean - truth)^2),
    bias = mean(posterior_mean - truth),
    mean_abs_error = mean(abs(posterior_mean - truth)),
    coverage_95 = mean(covered),
    zero_exclusion_rate = mean(zero_excluded),
    mean_interval_width = mean(upper - lower),
    mean_magnitude_attenuation = mean(abs(truth) - abs(posterior_mean)),
    toward_zero_miss_rate = mean(toward_zero_miss),
    toward_zero_share_of_misses = if (any(!covered)) {
      mean(toward_zero_miss[!covered])
    } else {
      NA_real_
    },
    stringsAsFactors = FALSE
  )
}

compute_metrics <- function(truth, posterior_mean, lower, upper, group_labels) {
  rbind(
    metric_row("all", truth, posterior_mean, lower, upper),
    metric_row(
      "group0",
      truth[group_labels == 0L],
      posterior_mean[group_labels == 0L],
      lower[group_labels == 0L],
      upper[group_labels == 0L]
    ),
    metric_row(
      "group1",
      truth[group_labels == 1L],
      posterior_mean[group_labels == 1L],
      lower[group_labels == 1L],
      upper[group_labels == 1L]
    )
  )
}

compute_binned_metrics <- function(
    truth,
    posterior_mean,
    lower,
    upper,
    breaks = c(0, 0.25, 0.5, 1, 2, Inf)) {
  bins <- cut(
    abs(truth),
    breaks = breaks,
    include.lowest = TRUE,
    right = FALSE
  )
  covered <- lower <= truth & truth <= upper
  zero_excluded <- lower > 0 | upper < 0
  split_indices <- split(seq_along(truth), bins, drop = FALSE)

  do.call(rbind, lapply(names(split_indices), function(bin_name) {
    indices <- split_indices[[bin_name]]
    if (length(indices) == 0L) {
      return(data.frame(
        abs_truth_bin = bin_name,
        n = 0L,
        coverage_95 = NA_real_,
        zero_exclusion_rate = NA_real_,
        mse = NA_real_,
        mean_interval_width = NA_real_,
        mean_magnitude_attenuation = NA_real_,
        stringsAsFactors = FALSE
      ))
    }
    data.frame(
      abs_truth_bin = bin_name,
      n = length(indices),
      coverage_95 = mean(covered[indices]),
      zero_exclusion_rate = mean(zero_excluded[indices]),
      mse = mean((posterior_mean[indices] - truth[indices])^2),
      mean_interval_width = mean(upper[indices] - lower[indices]),
      mean_magnitude_attenuation = mean(
        abs(truth[indices]) - abs(posterior_mean[indices])
      ),
      stringsAsFactors = FALSE
    )
  }))
}

compute_calibration_curve <- function(
    log_ratio_draws,
    truth,
    group_labels,
    nominal_levels) {
  nominal_levels <- sort(unique(as.numeric(nominal_levels)))
  if (length(nominal_levels) == 0L ||
      any(!is.finite(nominal_levels)) ||
      any(nominal_levels <= 0 | nominal_levels >= 1)) {
    stop("Calibration levels must be finite and strictly between 0 and 1")
  }

  lower_probs <- (1 - nominal_levels) / 2
  upper_probs <- 1 - lower_probs
  requested_probs <- c(lower_probs, upper_probs)
  quantiles <- matrixStats::rowQuantiles(
    log_ratio_draws,
    probs = requested_probs,
    na.rm = FALSE,
    drop = FALSE
  )
  n_levels <- length(nominal_levels)
  lower <- quantiles[, seq_len(n_levels), drop = FALSE]
  upper <- quantiles[, n_levels + seq_len(n_levels), drop = FALSE]
  covered <- lower <= truth & truth <= upper

  group_indices <- list(
    all = seq_along(truth),
    group0 = which(group_labels == 0L),
    group1 = which(group_labels == 1L)
  )
  do.call(rbind, lapply(names(group_indices), function(group_name) {
    indices <- group_indices[[group_name]]
    data.frame(
      group = group_name,
      nominal_level = nominal_levels,
      empirical_coverage = colMeans(covered[indices, , drop = FALSE]),
      n = length(indices),
      stringsAsFactors = FALSE
    )
  }))
}

result_filename <- function(n0, n1, seed) {
  sprintf("global_shift_n0_%05d_n1_%05d_seed_%02d.rds", n0, n1, seed)
}

run_global_shift_fit <- function(task, output_dir, source_paths) {
  assert_required_packages()
  batts_description <- assert_batts_version()

  final_path <- file.path(
    output_dir,
    result_filename(task$n0, task$n1, task$seed)
  )
  if (file.exists(final_path)) {
    stop("Refusing to overwrite existing result: ", final_path)
  }

  simulation <- generate_global_shift_20d(
    n0 = task$n0,
    n1 = task$n1,
    seed = task$seed,
    d = task$d,
    latent_dim = task$latent_dim,
    shift = task$shift,
    noise_sd = task$noise_sd
  )
  truth_validation_error <- validate_analytic_truth(simulation)

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
  if (!is.matrix(balance_draws) || nrow(balance_draws) != task$n0 + task$n1) {
    stop("Unexpected posterior draw dimensions")
  }
  if (any(!is.finite(balance_draws)) || any(balance_draws <= 0)) {
    stop("Non-positive or non-finite posterior balancing weights")
  }

  log_ratio_draws <- 2 * log(balance_draws)
  rm(balance_draws, fit)
  posterior_mean <- rowMeans(log_ratio_draws)
  intervals <- matrixStats::rowQuantiles(
    log_ratio_draws,
    probs = c(0.025, 0.975),
    na.rm = FALSE,
    drop = FALSE
  )
  lower <- intervals[, 1L]
  upper <- intervals[, 2L]

  calibration_curve <- NULL
  if (!is.null(task$calibration_levels)) {
    calibration_curve <- compute_calibration_curve(
      log_ratio_draws,
      simulation$true_log_ratio,
      simulation$group_labels,
      task$calibration_levels
    )
  }
  rm(log_ratio_draws)

  if (any(!is.finite(c(posterior_mean, lower, upper))) ||
      (!is.null(calibration_curve) &&
       any(!is.finite(calibration_curve$empirical_coverage)))) {
    stop("Non-finite posterior summaries")
  }
  if (any(lower > upper)) {
    stop("Reversed posterior interval")
  }

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

  source_hashes <- data.frame(
    path = normalizePath(source_paths, winslash = "/", mustWork = TRUE),
    sha256 = vapply(source_paths, sha256_file, character(1)),
    stringsAsFactors = FALSE
  )

  result <- list(
    schema_version = "1.0",
    scenario = "20d_global_shift_matched",
    task = task,
    metrics = metrics,
    binned_metrics = binned_metrics,
    calibration_curve = calibration_curve,
    pointwise = list(
      group_labels = simulation$group_labels,
      true_log_ratio = simulation$true_log_ratio,
      posterior_mean = posterior_mean,
      lower_95 = lower,
      upper_95 = upper
    ),
    loading = simulation$loading,
    signal_direction = simulation$signal_direction,
    validation = list(
      loading_gram_max_error = simulation$parameters$gram_error,
      analytic_truth_max_error = truth_validation_error
    ),
    provenance = list(
      batts_version = batts_description[["Version"]],
      batts_remote_sha = batts_description[["RemoteSha"]],
      r_version = R.version.string,
      platform = R.version$platform,
      rng_kind = RNGkind(),
      parallel_backend = "independent PSOCK worker; one seed per task",
      omp_num_threads = Sys.getenv("OMP_NUM_THREADS", unset = NA_character_),
      openblas_num_threads = Sys.getenv("OPENBLAS_NUM_THREADS", unset = NA_character_),
      mkl_num_threads = Sys.getenv("MKL_NUM_THREADS", unset = NA_character_),
      start_time = format(start_time, tz = "UTC", usetz = TRUE),
      end_time = format(fit_end_time, tz = "UTC", usetz = TRUE),
      elapsed_seconds = as.numeric(difftime(fit_end_time, start_time, units = "secs")),
      warnings = unique(warnings),
      source_hashes = source_hashes,
      session_info = capture.output(utils::sessionInfo())
    )
  )

  temporary_path <- tempfile(
    pattern = paste0(basename(final_path), ".tmp-"),
    tmpdir = output_dir
  )
  saveRDS(result, temporary_path, compress = "gzip")
  if (!file.rename(temporary_path, final_path)) {
    unlink(temporary_path)
    stop("Atomic result rename failed: ", final_path)
  }

  list(
    status = "completed",
    n0 = task$n0,
    n1 = task$n1,
    seed = task$seed,
    elapsed_seconds = result$provenance$elapsed_seconds,
    output = normalizePath(final_path, winslash = "/", mustWork = TRUE),
    sha256 = sha256_file(final_path),
    warnings = paste(unique(warnings), collapse = " | ")
  )
}
