boosting_parse_args <- function(args) {
  result <- list()
  for (arg in args) {
    if (!startsWith(arg, "--") || !grepl("=", arg, fixed = TRUE)) {
      stop("Arguments must use --name=value syntax: ", arg)
    }
    pair <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1]]
    result[[pair[[1]]]] <- paste(pair[-1], collapse = "=")
  }
  result
}

boosting_arg <- function(args, name, default = NULL, required = FALSE) {
  value <- args[[name]]
  if (is.null(value)) {
    if (isTRUE(required)) {
      stop("Missing required argument --", name, "=...")
    }
    return(default)
  }
  value
}

boosting_int <- function(value, name, minimum = 1L) {
  if (length(value) != 1L || !grepl("^[0-9]+$", value)) {
    stop("--", name, " must be an integer >= ", minimum)
  }
  parsed <- suppressWarnings(as.integer(value))
  if (is.na(parsed) || parsed < minimum) {
    stop("--", name, " must be an integer >= ", minimum)
  }
  parsed
}

boosting_bool <- function(value, name) {
  normalized <- tolower(value)
  if (normalized %in% c("true", "1", "yes")) {
    return(TRUE)
  }
  if (normalized %in% c("false", "0", "no")) {
    return(FALSE)
  }
  stop("--", name, " must be true or false")
}

boosting_mode <- function(args) {
  mode <- boosting_arg(args, "mode", "smoke")
  if (!mode %in% c("smoke", "canonical")) {
    stop("--mode must be smoke or canonical")
  }
  if (identical(mode, "canonical") &&
      !identical(boosting_arg(args, "confirm-canonical", ""), "YES")) {
    stop("Canonical mode requires --confirm-canonical=YES")
  }
  mode
}

boosting_script_path <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (!length(file_arg)) {
    stop("This script must be run with Rscript")
  }
  normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = TRUE)
}

boosting_project_root <- function(script_path) {
  normalizePath(file.path(dirname(script_path), "..", ".."), mustWork = TRUE)
}

boosting_git_commit <- function(project_root) {
  output <- system2(
    "git", c("-C", shQuote(project_root), "rev-parse", "HEAD"),
    stdout = TRUE, stderr = TRUE
  )
  status <- attr(output, "status")
  if (!is.null(status) && status != 0L) {
    stop("Could not determine Git commit: ", paste(output, collapse = "\n"))
  }
  trimws(output[[1]])
}

boosting_require_clean_git <- function(project_root, mode) {
  if (!identical(mode, "canonical")) {
    return(invisible(TRUE))
  }
  output <- system2(
    "git", c("-C", shQuote(project_root), "status", "--porcelain"),
    stdout = TRUE, stderr = TRUE
  )
  status <- attr(output, "status")
  if ((!is.null(status) && status != 0L) || length(output)) {
    stop("Canonical runs require a clean Git worktree")
  }
  invisible(TRUE)
}

boosting_load_packages <- function(r_lib) {
  normalized <- normalizePath(r_lib, mustWork = TRUE)
  .libPaths(c(normalized, .libPaths()))
  required <- c("BATTS", "ada", "rpart", "mvtnorm", "pracma")
  missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) {
    stop("Packages unavailable: ", paste(missing, collapse = ", "))
  }
  suppressPackageStartupMessages(library(BATTS, lib.loc = normalized))
  normalized
}

boosting_source_hashes <- function(files) {
  normalized <- normalizePath(files, mustWork = TRUE)
  hashes <- unname(tools::sha256sum(normalized))
  stats::setNames(as.list(hashes), basename(normalized))
}

boosting_package_versions <- function() {
  packages <- c("BATTS", "ada", "rpart", "Rcpp", "RcppArmadillo", "mvtnorm", "pracma")
  versions <- vapply(packages, function(package) {
    if (!requireNamespace(package, quietly = TRUE)) {
      return(NA_character_)
    }
    as.character(utils::packageVersion(package))
  }, character(1))
  as.list(versions)
}

boosting_install_hashes <- function(r_lib) {
  package_root <- file.path(r_lib, "BATTS")
  candidates <- c(
    file.path(package_root, "DESCRIPTION"),
    file.path(package_root, "Meta", "package.rds"),
    file.path(package_root, "R", "BATTS.rdb"),
    file.path(package_root, "R", "BATTS.rdx"),
    list.files(
      file.path(package_root, "libs"), pattern = "^BATTS[.]dll$",
      recursive = TRUE, full.names = TRUE
    )
  )
  files <- unique(candidates[file.exists(candidates)])
  if (!length(files)) {
    stop("No hashable BATTS installation files were found under --r-lib")
  }
  hashes <- unname(tools::sha256sum(files))
  relative <- substring(
    normalizePath(files, winslash = "/", mustWork = TRUE),
    nchar(normalizePath(package_root, winslash = "/", mustWork = TRUE)) + 2L
  )
  stats::setNames(as.list(hashes), relative)
}

boosting_metadata <- function(mode, family, job_id, seed, config, project_root,
                              source_files, r_lib) {
  list(
    schema_version = 1L,
    status = if (identical(mode, "smoke")) "SMOKE_NOT_FOR_PAPER" else "CANONICAL",
    family = family,
    job_id = job_id,
    seed = seed,
    config = config,
    commit_hash = boosting_git_commit(project_root),
    source_hashes = boosting_source_hashes(source_files),
    r_library = r_lib,
    batts_install_hashes = boosting_install_hashes(r_lib),
    r_version = R.version.string,
    platform = R.version$platform,
    os = as.list(Sys.info()),
    package_versions = boosting_package_versions(),
    rng_kind = RNGkind(),
    parallel_backend = "sequential",
    core_count = 1L,
    created_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)
  )
}

boosting_output_paths <- function(output_root, mode, job_id) {
  mode_dir <- file.path(path.expand(output_root), mode)
  dir.create(mode_dir, recursive = TRUE, showWarnings = FALSE)
  if (!dir.exists(mode_dir)) {
    stop("Could not create output directory: ", mode_dir)
  }
  list(
    result = file.path(mode_dir, paste0(job_id, ".rds")),
    manifest = file.path(mode_dir, paste0(job_id, ".manifest.rds"))
  )
}

boosting_resume_state <- function(paths, resume, expected_metadata) {
  result_exists <- file.exists(paths$result)
  manifest_exists <- file.exists(paths$manifest)
  if (!result_exists && !manifest_exists) {
    return(FALSE)
  }
  if (!isTRUE(resume)) {
    stop("Output already exists; refusing to overwrite: ", paths$result)
  }
  if (!result_exists || !manifest_exists) {
    stop("Incomplete prior output requires manual audit: ", paths$result)
  }
  manifest <- readRDS(paths$manifest)
  actual_hash <- unname(tools::sha256sum(paths$result))
  if (!identical(actual_hash, manifest$output_sha256)) {
    stop("Existing output hash does not match its manifest: ", paths$result)
  }
  identity_fields <- c(
    "schema_version", "status", "family", "job_id", "seed", "config",
    "commit_hash", "source_hashes", "batts_install_hashes"
  )
  mismatched <- identity_fields[!vapply(identity_fields, function(field) {
    identical(manifest[[field]], expected_metadata[[field]])
  }, logical(1))]
  if (length(mismatched)) {
    stop("Existing output identity mismatch: ", paste(mismatched, collapse = ", "))
  }
  TRUE
}

boosting_atomic_save <- function(result, metadata, paths) {
  result_temp <- tempfile(
    pattern = paste0(basename(paths$result), "."),
    tmpdir = dirname(paths$result), fileext = ".tmp"
  )
  manifest_temp <- tempfile(
    pattern = paste0(basename(paths$manifest), "."),
    tmpdir = dirname(paths$manifest), fileext = ".tmp"
  )
  on.exit(unlink(c(result_temp, manifest_temp), force = TRUE), add = TRUE)
  saveRDS(list(metadata = metadata, result = result), result_temp, compress = "xz")
  output_hash <- unname(tools::sha256sum(result_temp))
  if (!file.rename(result_temp, paths$result)) {
    stop("Atomic result rename failed: ", paths$result)
  }
  manifest <- metadata
  manifest$output_file <- basename(paths$result)
  manifest$output_sha256 <- output_hash
  manifest$output_bytes <- file.info(paths$result)$size
  saveRDS(manifest, manifest_temp, compress = "xz")
  if (!file.rename(manifest_temp, paths$manifest)) {
    stop("Atomic manifest rename failed: ", paths$manifest)
  }
  invisible(manifest)
}

boosting_seed_map <- function(seed) {
  list(
    data = seed,
    folds = 100000L + seed,
    ada_cv_base = 200000L + seed * 100L,
    ada_final = 300000L + seed,
    gb = 500000L + seed,
    fs = 600000L + seed
  )
}

boosting_make_stratified_folds <- function(labels, n_folds, seed) {
  fold_id <- integer(length(labels))
  set.seed(seed)
  for (label in sort(unique(labels))) {
    indices <- which(labels == label)
    fold_id[indices] <- sample(rep(seq_len(n_folds), length.out = length(indices)))
  }
  fold_id
}

boosting_make_ada_frame <- function(data, labels) {
  result <- data.frame(y = factor(labels), data, check.names = FALSE)
  colnames(result)[-1L] <- paste0("X", seq_len(ncol(data)))
  result
}

boosting_tree_contributions <- function(model, newdata) {
  predict_one <- model$model$lossObj$predict.type
  contributions <- vapply(
    seq_len(model$iter),
    function(index_tree) {
      as.numeric(predict_one(
        f = model$model$trees[[index_tree]], dat = newdata
      )) * model$model$alpha[[index_tree]]
    },
    numeric(nrow(newdata))
  )
  if (is.null(dim(contributions))) {
    contributions <- matrix(contributions, ncol = 1L)
  }
  contributions
}

boosting_cumulative_margins <- function(contributions) {
  margins <- contributions
  if (ncol(margins) > 1L) {
    for (index_tree in 2L:ncol(margins)) {
      margins[, index_tree] <- margins[, index_tree - 1L] + contributions[, index_tree]
    }
  }
  margins
}

boosting_log_mean_exp_columns <- function(values) {
  maxima <- apply(values, 2L, max)
  centered <- sweep(values, 2L, maxima, FUN = "-")
  maxima + log(colMeans(exp(centered)))
}

boosting_log_add_exp <- function(a, b) {
  maxima <- pmax(a, b)
  maxima + log(exp(a - maxima) + exp(b - maxima))
}

boosting_balance_log_loss <- function(margins, labels, training_log_prior_ratio) {
  group0 <- labels == 0L
  group1 <- labels == 1L
  if (!any(group0) || !any(group1)) {
    stop("Each validation fold must contain both groups")
  }
  log_term0 <- boosting_log_mean_exp_columns(
    margins[group0, , drop = FALSE] - 0.5 * training_log_prior_ratio
  )
  log_term1 <- boosting_log_mean_exp_columns(
    -margins[group1, , drop = FALSE] + 0.5 * training_log_prior_ratio
  )
  boosting_log_add_exp(log_term0, log_term1)
}

boosting_compute_ada_cv <- function(data, labels, fold_id, settings, seed_base) {
  frame <- boosting_make_ada_frame(data, labels)
  n_folds <- max(fold_id)
  class_error <- matrix(NA_real_, nrow = n_folds, ncol = settings$max_trees)
  log_exponential <- matrix(NA_real_, nrow = n_folds, ncol = settings$max_trees)
  log_balancing <- matrix(NA_real_, nrow = n_folds, ncol = settings$max_trees)

  control <- rpart::rpart.control(
    maxdepth = settings$ada_depth, cp = -1, minsplit = 0L
  )
  for (fold in seq_len(n_folds)) {
    train <- fold_id != fold
    validation <- !train
    set.seed(seed_base + fold)
    model <- ada::ada(
      y ~ ., data = frame[train, , drop = FALSE], type = "real",
      control = control, iter = settings$max_trees,
      nu = settings$learn_rate, bag.frac = settings$ada_bag_fraction,
      test.x = frame[validation, -1L, drop = FALSE],
      test.y = frame[validation, 1L]
    )
    if (!identical(as.integer(model$iter), settings$max_trees)) {
      stop("AdaBoost returned ", model$iter, " trees; expected ", settings$max_trees)
    }
    class_error[fold, ] <- model$model$errs[, "test.err"]
    margins <- boosting_cumulative_margins(boosting_tree_contributions(
      model, frame[validation, -1L, drop = FALSE]
    ))
    validation_labels <- labels[validation]
    signed_labels <- ifelse(validation_labels == 1L, 1, -1)
    log_exponential[fold, ] <- boosting_log_mean_exp_columns(
      -signed_labels * margins
    )
    n0_train <- sum(labels[train] == 0L)
    n1_train <- sum(labels[train] == 1L)
    log_balancing[fold, ] <- boosting_balance_log_loss(
      margins, validation_labels, log(n1_train / n0_train)
    )
  }
  list(
    fold_classification_error = class_error,
    fold_log_exponential_loss = log_exponential,
    fold_log_balancing_loss = log_balancing,
    mean_classification_error = colMeans(class_error),
    mean_log_exponential_loss = colMeans(log_exponential),
    mean_log_balancing_loss = colMeans(log_balancing)
  )
}

boosting_select_ada_trees <- function(cv, min_trees_classification, max_trees) {
  raw <- c(
    classification_error = which.min(cv$mean_classification_error),
    exponential_loss = which.min(cv$mean_log_exponential_loss),
    balancing_loss = which.min(cv$mean_log_balancing_loss)
  )
  final <- raw
  final[["classification_error"]] <- max(
    raw[["classification_error"]], min_trees_classification
  )
  data.frame(
    criterion = names(raw),
    role = c("legacy", "primary", "counterfactual_diagnostic"),
    raw_selected_trees = as.integer(raw),
    final_selected_trees = as.integer(final),
    maximum_tree_hit = as.integer(final) == max_trees,
    stringsAsFactors = FALSE
  )
}

boosting_fold_argmins <- function(cv) {
  matrices <- list(
    classification_error = cv$fold_classification_error,
    exponential_loss = cv$fold_log_exponential_loss,
    balancing_loss = cv$fold_log_balancing_loss
  )
  do.call(rbind, lapply(names(matrices), function(criterion) {
    data.frame(
      criterion = criterion,
      fold = seq_len(nrow(matrices[[criterion]])),
      selected_trees = apply(matrices[[criterion]], 1L, which.min),
      stringsAsFactors = FALSE
    )
  }))
}

boosting_cross_loss <- function(cv, selections) {
  curves <- list(
    classification_error = cv$mean_classification_error,
    exponential_loss = cv$mean_log_exponential_loss,
    balancing_loss = cv$mean_log_balancing_loss
  )
  do.call(rbind, lapply(seq_len(nrow(selections)), function(index) {
    selected_by <- selections$criterion[[index]]
    tree <- selections$final_selected_trees[[index]]
    data.frame(
      selected_by = selected_by,
      selected_trees = tree,
      evaluated_with = names(curves),
      value = vapply(curves, function(curve) curve[[tree]], numeric(1)),
      value_scale = c("error_rate", "log_loss", "log_loss"),
      stringsAsFactors = FALSE
    )
  }))
}

boosting_fit_ada_final <- function(data, labels, n_trees, settings, seed) {
  frame <- boosting_make_ada_frame(data, labels)
  set.seed(seed)
  model <- ada::ada(
    y ~ ., data = frame, type = "real",
    control = rpart::rpart.control(
      maxdepth = settings$ada_depth, cp = -1, minsplit = 0L
    ),
    iter = n_trees, nu = settings$learn_rate,
    bag.frac = settings$ada_bag_fraction
  )
  if (!identical(as.integer(model$iter), as.integer(n_trees))) {
    stop("Final AdaBoost fit returned an unexpected number of trees")
  }
  model
}

boosting_ada_log_ratio <- function(model, newdata, n_trees, n0, n1) {
  frame <- data.frame(newdata, check.names = FALSE)
  colnames(frame) <- paste0("X", seq_len(ncol(frame)))
  margin <- as.numeric(stats::predict(
    model, newdata = frame, type = "F", n.iter = n_trees
  ))
  -2 * margin + log(n1 / n0)
}

boosting_metrics <- function(estimated, truth, labels) {
  group0 <- labels == 0L
  group1 <- labels == 1L
  squared_error <- (estimated - truth)^2
  mse0 <- if (all(is.finite(estimated[group0]))) mean(squared_error[group0]) else Inf
  mse1 <- if (all(is.finite(estimated[group1]))) mean(squared_error[group1]) else Inf
  c(
    mse_group0 = mse0,
    mse_group1 = mse1,
    mse_symmetric = (mse0 + mse1) / 2,
    nonfinite = sum(!is.finite(estimated)),
    max_abs_finite = if (any(is.finite(estimated))) {
      max(abs(estimated[is.finite(estimated)]))
    } else {
      NA_real_
    }
  )
}

boosting_metric_row <- function(method, selection, split, estimated, truth, labels) {
  data.frame(
    method = method,
    selection = selection,
    split = split,
    as.list(boosting_metrics(estimated, truth, labels)),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

boosting_ada_node_summary <- function(model, selections) {
  nodes_per_tree <- vapply(
    model$model$trees,
    function(tree) if (is.null(tree$frame)) NA_integer_ else nrow(tree$frame),
    integer(1)
  )
  data.frame(
    criterion = selections$criterion,
    selected_trees = selections$final_selected_trees,
    total_nodes = vapply(selections$final_selected_trees, function(tree) {
      sum(nodes_per_tree[seq_len(tree)], na.rm = TRUE)
    }, numeric(1)),
    mean_nodes_per_tree = vapply(selections$final_selected_trees, function(tree) {
      mean(nodes_per_tree[seq_len(tree)], na.rm = TRUE)
    }, numeric(1)),
    stringsAsFactors = FALSE
  )
}

boosting_run_proposed <- function(data, labels, truth_train, settings,
                                  use_gradient, seed, method_name) {
  expected_fold_id <- boosting_make_stratified_folds(labels, settings$folds, seed)
  set.seed(seed)
  fit <- BATTS::boots(
    data = data,
    group_labels = labels,
    num_trees_max = settings$max_trees,
    K_CV = settings$folds,
    max_resol = settings$proposed_depth,
    learn_rate = settings$learn_rate,
    use_gradient = use_gradient,
    quiet = TRUE
  )
  cv <- fit$loss_CV_store
  selected <- length(fit$tree_list)
  aggregate_selected <- which.min(colMeans(cv))
  if (!identical(as.integer(selected), as.integer(aggregate_selected))) {
    stop(method_name, " selected tree count does not match its CV argmin")
  }
  train_estimate <- 2 * log(as.numeric(fit$balance_weight_boosting_data))
  list(
    diagnostics = list(
      method = method_name,
      selected_trees = selected,
      maximum_tree_hit = selected == settings$max_trees,
      fold_id = expected_fold_id,
      fold_selected_trees = apply(cv, 1L, which.min),
      fold_loss = cv,
      mean_loss = colMeans(cv),
      nodes_per_tree = as.integer(fit$n_nodes),
      total_nodes = sum(fit$n_nodes),
      mean_nodes_per_tree = mean(fit$n_nodes)
    ),
    estimates = list(train = train_estimate),
    metrics = boosting_metric_row(
      method_name, "native_balancing_loss", "train",
      train_estimate, truth_train, labels
    )
  )
}
