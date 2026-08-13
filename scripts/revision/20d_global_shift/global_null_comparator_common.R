global_null_metrics <- function(estimated, truth, labels) {
  stopifnot(length(estimated) == length(truth), length(truth) == length(labels))
  groups <- list(all = rep(TRUE, length(labels)), group0 = labels == 0L,
                 group1 = labels == 1L)
  rows <- lapply(names(groups), function(group_name) {
    keep <- groups[[group_name]]
    estimate <- estimated[keep]
    target <- truth[keep]
    finite <- is.finite(estimate) & is.finite(target)
    error <- estimate - target
    data.frame(
      group = group_name,
      n = sum(keep),
      finite_n = sum(finite),
      nonfinite_n = sum(!finite),
      mse = if (all(finite)) mean(error^2) else Inf,
      finite_mse = if (any(finite)) mean(error[finite]^2) else NA_real_,
      bias = if (all(finite)) mean(error) else NA_real_,
      finite_bias = if (any(finite)) mean(error[finite]) else NA_real_,
      mean_abs_error = if (all(finite)) mean(abs(error)) else Inf,
      finite_mean_abs_error = if (any(finite)) mean(abs(error[finite])) else NA_real_,
      estimate_sd = if (sum(finite) > 1L) stats::sd(estimate[finite]) else NA_real_,
      max_abs_estimate = if (any(finite)) max(abs(estimate[finite])) else NA_real_,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

global_null_symmetric_mse <- function(metrics) {
  group_rows <- metrics[metrics$group %in% c("group0", "group1"), ]
  if (nrow(group_rows) != 2L || any(!is.finite(group_rows$mse))) Inf else mean(group_rows$mse)
}

global_null_log_kde_at <- function(eval_points, sample_points, bandwidth) {
  vapply(eval_points, function(point) {
    terms <- -0.5 * ((point - sample_points) / bandwidth)^2
    maximum <- max(terms)
    maximum + log(mean(exp(terms - maximum))) - log(bandwidth) - 0.5 * log(2 * pi)
  }, numeric(1))
}

global_null_fit_cdc <- function(logit_score, labels) {
  score0 <- logit_score[labels == 0L]
  score1 <- logit_score[labels == 1L]
  list(
    score0 = score0,
    score1 = score1,
    density0 = stats::density(score0),
    density1 = stats::density(score1)
  )
}

global_null_predict_cdc <- function(object, score) {
  f0 <- stats::approx(object$density0$x, object$density0$y,
                      xout = score, rule = 2)$y
  f1 <- stats::approx(object$density1$x, object$density1$y,
                      xout = score, rule = 2)$y
  submitted <- log(f0 / f1)
  stable <- log(f0) - log(f1)
  repair <- which(!is.finite(stable))
  if (length(repair)) {
    stable[repair] <-
      global_null_log_kde_at(score[repair], object$score0, object$density0$bw) -
      global_null_log_kde_at(score[repair], object$score1, object$density1$bw)
  }
  list(
    submitted = submitted,
    stable = stable,
    diagnostics = list(
      bandwidth0 = object$density0$bw,
      bandwidth1 = object$density1$bw,
      zero_f0 = sum(f0 == 0),
      zero_f1 = sum(f1 == 0),
      submitted_nonfinite = sum(!is.finite(submitted)),
      stable_repaired = length(repair),
      stable_nonfinite = sum(!is.finite(stable))
    )
  )
}

global_null_atomic_save <- function(object, path, compress = "gzip") {
  temporary <- tempfile(paste0(basename(path), ".tmp-"), tmpdir = dirname(path))
  on.exit(unlink(temporary), add = TRUE)
  saveRDS(object, temporary, compress = compress)
  if (!file.rename(temporary, path)) stop("Atomic rename failed: ", path)
  on.exit(NULL, add = FALSE)
  invisible(path)
}

global_null_resume_state <- function(result_path, sidecar_path, resume,
                                     expected_identity) {
  result_exists <- file.exists(result_path)
  sidecar_exists <- file.exists(sidecar_path)
  if (!result_exists && !sidecar_exists) return(FALSE)
  if (!isTRUE(resume)) {
    stop("Output already exists; refusing to overwrite: ", result_path)
  }
  if (!result_exists || !sidecar_exists) {
    stop("Incomplete prior output requires manual audit: ", result_path)
  }
  sidecar <- readRDS(sidecar_path)
  actual_hash <- unname(tools::sha256sum(result_path))
  if (!identical(actual_hash, sidecar$result_sha256)) {
    stop("Existing result hash does not match its sidecar: ", result_path)
  }
  mismatched <- names(expected_identity)[!vapply(
    names(expected_identity),
    function(field) identical(sidecar[[field]], expected_identity[[field]]),
    logical(1)
  )]
  if (length(mismatched)) {
    stop("Existing output identity mismatch: ", paste(mismatched, collapse = ", "))
  }
  TRUE
}
