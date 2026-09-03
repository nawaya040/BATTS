#!/usr/bin/env Rscript

# Render revised component graphics for Supplementary Figures S2 and S3 from
# completed, checksummed results. No estimator is fitted or tuned here.

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_arg)) stop("This script must be run with Rscript")
script_path <- normalizePath(sub("^--file=", "", script_arg[[1L]]), winslash = "/")
repo_root <- normalizePath(file.path(dirname(script_path), "..", ".."), winslash = "/")
legacy_root <- "C:/Users/user/Dropbox/Rcpp_experiments/active_programs/balancePM_results/experiments_2D"
canonical_root <- paste0("C:/Users/user/Dropbox/balancePM_experiment_backup_0806/",
  "boosting-selection-full-20260804/outputs/canonical")
checksums_path <- paste0("C:/Users/user/Dropbox/Rcpp_experiments/active_programs/balancePM_backup/",
  "tmp/boosting-full-launcher-20260804/github-result-bundle/results/reference/",
  "boosting-selection-full-20260804/output_checksums.csv")
model_path <- file.path(repo_root, "scripts", "coverage", "models", "section41_2d_models.R")
expected_model_hash <- "ed19303c802057751b6a28e4ce63be8570bbe29da0da277c4a24c5f05f0f3790"

settings <- list(
  list(id="s2_global_shift_balanced", figure="S2", scenario="global_shift", n0=5000L, n1=5000L, seed=1L),
  list(id="s2_local_dispersion_balanced", figure="S2", scenario="local_dispersion", n0=5000L, n1=5000L, seed=21L),
  list(id="s3_global_shift_unbalanced", figure="S3", scenario="global_shift", n0=9000L, n1=1000L, seed=1L),
  list(id="s3_local_shift_unbalanced", figure="S3", scenario="local_shift", n0=9000L, n1=1000L, seed=16L),
  list(id="s3_local_dispersion_unbalanced", figure="S3", scenario="local_dispersion", n0=9000L, n1=1000L, seed=16L)
)
for (package in c("ggplot2", "scales", "mvtnorm")) {
  if (!requireNamespace(package, quietly = TRUE)) stop("Missing package: ", package)
}
if (!file.exists(checksums_path) || !file.exists(model_path)) stop("Missing checksum table or 2D model source")
model_hash <- tolower(unname(tools::sha256sum(model_path)))
if (!identical(model_hash, expected_model_hash)) stop("2D model source hash mismatch")
checksums <- utils::read.csv(checksums_path, stringsAsFactors = FALSE)
source(model_path, local = TRUE)

finite_vector <- function(x, n, label) {
  x <- as.numeric(x)
  if (length(x) != n || any(!is.finite(x))) stop("Invalid vector: ", label)
  x
}
pick_quantile <- function(x, probability) {
  probabilities <- as.numeric(sub("%", "", rownames(x), fixed = TRUE)) / 100
  as.numeric(x[which.min(abs(probabilities - probability)), ])
}
palette_colours <- c("darkblue", "blue", "blue", "deepskyblue", "cyan", "#7FFFFF",
  "white", "white", "white", "#FFFF99", "yellow", "orange", "red", "red", "darkred")

render_setting <- function(setting) {
  n <- setting$n0 + setting$n1
  legacy_path <- file.path(legacy_root, setting$scenario, paste(
    setting$scenario, setting$n0, setting$n1, 5L, setting$seed, "details.rds", sep="_"))
  job <- sprintf("boosting_selection_2d_%s_n0-%d_n1-%d_transformed-false_seed-%03d",
    setting$scenario, setting$n0, setting$n1, setting$seed)
  boosting_path <- file.path(canonical_root, paste0(job, ".rds"))
  if (!file.exists(legacy_path) || !file.exists(boosting_path)) stop("Missing input for ", setting$id)
  checksum_row <- checksums[checksums$job_id == job, , drop = FALSE]
  if (nrow(checksum_row) != 1L) stop("Expected one checksum row for ", job)
  actual_hash <- tolower(unname(tools::sha256sum(boosting_path)))
  actual_bytes <- unname(file.info(boosting_path)$size)
  if (!identical(actual_hash, tolower(checksum_row$output_sha256[[1L]])) ||
      !identical(as.numeric(actual_bytes), as.numeric(checksum_row$output_bytes[[1L]]))) {
    stop("Canonical boosting checksum mismatch for ", job)
  }
  legacy <- readRDS(legacy_path); boosting <- readRDS(boosting_path); config <- boosting$metadata$config
  if (!identical(boosting$metadata$status, "CANONICAL") || !identical(boosting$metadata$job_id, job) ||
      !identical(config$family, "2d") || !identical(config$scenario, setting$scenario) ||
      !identical(config$n0, setting$n0) || !identical(config$n1, setting$n1) ||
      !identical(config$transformed, FALSE) || !identical(config$seed_map$data, setting$seed)) {
    stop("Canonical configuration mismatch for ", setting$id)
  }
  set.seed(setting$seed)
  generated <- simulation_2d(setting$n0, setting$n1, setting$scenario, 100L)
  coordinates <- as.matrix(generated$data)
  truth <- finite_vector(boosting$result$design$truth_train, n, "truth")
  truth_difference <- max(abs(truth - finite_vector(generated$true_log_w_obs, n, "regenerated truth")))
  if (truth_difference > 1e-12) stop("Regenerated truth mismatch for ", setting$id)
  values <- list(
    "Truth"=truth,
    "DRT (AdaBoost)"=finite_vector(boosting$result$adaboost$estimates$train$exponential_loss, n, "DRT"),
    "KLIEP"=finite_vector(legacy$log_ratio_KLIEP_data, n, "KLIEP"),
    "uLSIF"=finite_vector(legacy$log_ratio_uLSIF_data, n, "uLSIF"),
    "GB"=finite_vector(boosting$result$proposed$estimates$gb$train, n, "GB"),
    "Bayesian additive trees"=finite_vector(legacy$log_ratio_BART_data_mean, n, "BAT mean"),
    "Lower (2.5%)"=finite_vector(pick_quantile(legacy$log_ratio_BART_data_quantiles, .025), n, "BAT lower"),
    "Upper (97.5%)"=finite_vector(pick_quantile(legacy$log_ratio_BART_data_quantiles, .975), n, "BAT upper"))
  if (any(values[[7L]] > values[[8L]])) stop("BAT quantile order failure for ", setting$id)
  legacy_gb <- finite_vector(legacy$log_ratio_boosting_grad_data, n, "legacy GB scale reference")
  min_true <- min(truth); max_true <- max(truth); min_est <- min(legacy_gb); max_est <- max(legacy_gb)
  threshold <- 0.55
  min_display <- min((min_true + min_est) / 2, -threshold)
  max_display <- max((max_true + max_est) / 2, threshold)
  min_palette <- min(min_true, min_est, -threshold) - 1e-10
  max_palette <- max(max_true, max_est, threshold) + 1e-10
  palette_locations <- scales::rescale(c(min_palette, min_display, min_display*8/10,
    min_display*6/10, min_display*4/10, min_display*2/10, min_display*1/10, 0,
    max_display*1/10, max_display*2/10, max_display*4/10, max_display*6/10,
    max_display*8/10, max_display, max_palette))
  plot_data <- do.call(rbind, lapply(names(values), function(method) data.frame(
    x1=coordinates[,1L], x2=coordinates[,2L], log_weight=values[[method]], method=method)))
  plot_data$method <- factor(plot_data$method, levels=names(values))
  plot_data <- plot_data[sample.int(nrow(plot_data)),,drop=FALSE]
  pad <- function(x) { r <- range(x); r + c(-1,1)*diff(r)*.04 }
  plot <- ggplot2::ggplot(plot_data, ggplot2::aes(x1,x2,colour=log_weight)) +
    ggplot2::geom_point(size=2,alpha=.75,stroke=0) +
    ggplot2::scale_colour_gradientn(colours=palette_colours,values=palette_locations,
      limits=c(min_display,max_display),oob=scales::squish,name="log(density ratio)",
      guide=ggplot2::guide_colourbar(barwidth=grid::unit(1.05,"cm"),
        barheight=grid::unit(6.4,"cm"),title.position="top",title.hjust=.5)) +
    ggplot2::facet_wrap(~method,nrow=2L,ncol=4L) +
    ggplot2::coord_equal(xlim=pad(coordinates[,1L]),ylim=pad(coordinates[,2L]),expand=FALSE) +
    ggplot2::labs(x=expression(x[1]),y=expression(x[2])) +
    ggplot2::theme_gray(base_size=15,base_family="sans") +
    ggplot2::theme(panel.grid.minor=ggplot2::element_blank(),
      panel.border=ggplot2::element_rect(fill=NA,colour="#4D4D4D",linewidth=.45),
      panel.background=ggplot2::element_rect(fill="#EBEBEB",colour=NA),
      strip.background=ggplot2::element_rect(fill="#D9D9D9",colour="#4D4D4D",linewidth=.45),
      strip.text=ggplot2::element_text(size=14,face="bold",margin=ggplot2::margin(5,3,5,3)),
      axis.title=ggplot2::element_text(size=18),
      axis.text=ggplot2::element_text(size=11,colour="#333333"),
      legend.title=ggplot2::element_text(size=16),legend.text=ggplot2::element_text(size=14),
      legend.position="right",plot.margin=ggplot2::margin(7,7,7,7))
  output_png <- file.path(repo_root,"output","figures",paste0("supplement_",setting$id,"_r2_preview.png"))
  output_pdf <- file.path(repo_root,"output","pdf",paste0("supplement_",setting$id,"_r2_preview.pdf"))
  metadata_path <- file.path(repo_root,"output","figures",paste0("supplement_",setting$id,"_r2_preview_metadata.txt"))
  dir.create(dirname(output_png),recursive=TRUE,showWarnings=FALSE)
  dir.create(dirname(output_pdf),recursive=TRUE,showWarnings=FALSE)
  ggplot2::ggsave(output_png,plot,width=13.2,height=9.2,units="in",dpi=300,bg="white")
  ggplot2::ggsave(output_pdf,plot,width=13.2,height=9.2,units="in",device=grDevices::cairo_pdf,bg="white")
  selection <- boosting$result$adaboost$selections
  selection <- selection[selection$criterion=="exponential_loss",,drop=FALSE]
  writeLines(c(paste0("figure=Supplementary Figure ",setting$figure),paste0("scenario=",setting$scenario),
    paste0("n0=",setting$n0),paste0("n1=",setting$n1),paste0("simulation_repeat=",setting$seed),
    paste0("boosting_job_id=",job),paste0("boosting_sha256=",actual_hash),
    paste0("legacy_detail_sha256=",tolower(unname(tools::sha256sum(legacy_path)))),
    paste0("model_sha256=",model_hash),"drt_selection=exponential_loss",
    paste0("drt_selected_trees=",selection$final_selected_trees[[1L]]),"point_size=2","point_alpha=0.75",
    "axis_title_size=18","axis_margin_fraction=0.04_per_side","panel_background=#EBEBEB",
    paste0("truth_alignment_max_abs_difference=",format(truth_difference,scientific=TRUE))),metadata_path)
  cat("Saved ",setting$id," PNG: ",normalizePath(output_png),"\n",sep="")
  cat("Saved ",setting$id," PDF: ",normalizePath(output_pdf),"\n",sep="")
}

invisible(lapply(settings, render_setting))
