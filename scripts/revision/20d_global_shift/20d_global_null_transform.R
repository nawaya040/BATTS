transform_global_null_20d <- function(simulation, a_beta = 0.5, b_beta = 10) {
  stopifnot(
    is.list(simulation),
    is.matrix(simulation$data),
    is.matrix(simulation$loading),
    is.finite(a_beta), a_beta > 0,
    is.finite(b_beta), b_beta > 0
  )

  raw_data <- simulation$data
  noise_sd <- simulation$parameters$noise_sd
  reference_sd <- sqrt(rowSums(simulation$loading^2) + noise_sd^2)
  if (length(reference_sd) != ncol(raw_data) ||
      any(!is.finite(reference_sd)) || any(reference_sd <= 0)) {
    stop("Invalid marginal reference standard deviations")
  }

  standardized <- sweep(raw_data, 2L, reference_sd, "/")
  probabilities <- stats::pnorm(standardized)
  if (any(!is.finite(probabilities)) ||
      any(probabilities <= 0) || any(probabilities >= 1)) {
    stop("Marginal Gaussian probabilities are not strictly inside (0, 1)")
  }
  transformed_data <- matrix(
    stats::qbeta(as.vector(probabilities), shape1 = a_beta, shape2 = b_beta),
    nrow = nrow(raw_data),
    ncol = ncol(raw_data)
  )
  if (any(!is.finite(transformed_data))) {
    stop("Non-finite values after the marginal Beta transformation")
  }

  simulation$data_before_transform <- raw_data
  simulation$data <- transformed_data
  simulation$parameters$transformed <- TRUE
  simulation$parameters$transform <- list(
    name = "null_reference_gaussian_to_beta",
    reference_mean = 0,
    reference_sd = reference_sd,
    beta_shape1 = a_beta,
    beta_shape2 = b_beta,
    consumes_rng = FALSE
  )
  simulation
}

generate_global_null_20d <- function(task) {
  shift <- if (identical(task$scenario, "global_shift")) 1 else 0
  simulation <- generate_global_shift_20d(
    n0 = task$n0,
    n1 = task$n1,
    seed = task$seed,
    d = task$d,
    latent_dim = task$latent_dim,
    shift = shift,
    noise_sd = task$noise_sd
  )
  if (isTRUE(task$transformed)) {
    simulation <- transform_global_null_20d(
      simulation,
      a_beta = task$beta_shape1,
      b_beta = task$beta_shape2
    )
  } else {
    simulation$data_before_transform <- simulation$data
    simulation$parameters$transformed <- FALSE
  }
  if (identical(task$scenario, "null") &&
      !identical(simulation$true_log_ratio, numeric(task$n0 + task$n1))) {
    stop("Null truth is not identically zero")
  }
  simulation
}

validate_global_null_simulation <- function(simulation, tolerance = 1e-8) {
  validation_input <- simulation
  validation_input$data <- simulation$data_before_transform
  truth_error <- validate_analytic_truth(validation_input, tolerance = tolerance)
  if (!identical(
    simulation$true_log_ratio,
    validation_input$true_log_ratio
  )) {
    stop("The marginal transformation changed the stored truth")
  }
  truth_error
}

global_null_data_hash <- function(simulation) {
  digest::digest(
    list(
      data = simulation$data,
      labels = simulation$group_labels,
      truth = simulation$true_log_ratio,
      loading = simulation$loading,
      transformed = simulation$parameters$transformed
    ),
    algo = "sha256",
    serialize = TRUE
  )
}
