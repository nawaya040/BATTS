#!/usr/bin/env Rscript
# Display-only S2/S3/S5 candidates using the approved Main Figure 3 typography.
# Data preparation blocks copied from the corresponding original plot scripts.
a <- commandArgs(TRUE)
if (length(a)!=5L) stop("Arguments: legacy-2d-root legacy-S5-rds canonical-root checksums-csv new-output-dir")
legacy_root <- a[1]; s5_legacy_path <- a[2]; canonical_root <- a[3]; checksums_path <- a[4]; outdir <- a[5]
script_arg <- grep("^--file=",commandArgs(FALSE),value=TRUE)
repo_root <- normalizePath(file.path(dirname(sub("^--file=","",script_arg)),"../.."),winslash="/")
if (dir.exists(outdir)) stop("Candidate directory already exists")
for (p in c("ggplot2","scales","mvtnorm","pracma")) if (!requireNamespace(p,quietly=TRUE)) stop("Missing package: ",p)
model_path <- file.path(repo_root,"scripts/coverage/models/section41_2d_models.R")
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

prepare_setting <- function(setting) {
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
  list(data=plot_data, xlim=pad(coordinates[,1L]), ylim=pad(coordinates[,2L]),
    colours=palette_colours, positions=palette_locations, limits=c(min_display,max_display),
    axes=c("x1","x2"), xlab=expression(x[1]), ylab=expression(x[2]),
    hashes=c(boosting_sha256=actual_hash,legacy_detail_sha256=tolower(unname(tools::sha256sum(legacy_path))),model_sha256=model_hash),
    truth_difference=truth_difference)
}
prepare_s5 <- function() {
legacy_path <- s5_legacy_path
boosting_path <- file.path(canonical_root,"boosting_selection_20d_latent_location_shift_n0-5000_n1-5000_transformed-false_seed-021.rds")
model_path <- file.path(repo_root,"scripts/coverage/models/section42_multi_models.R")
for (path in c(legacy_path, boosting_path, checksums_path, model_path)) {
  if (!file.exists(path)) stop("Missing required input: ", path)
}
for (package in c("ggplot2", "scales", "mvtnorm", "pracma")) {
  if (!requireNamespace(package, quietly = TRUE)) stop("Missing package: ", package)
}

expected_job <- paste0(
  "boosting_selection_20d_latent_location_shift_n0-5000_n1-5000_",
  "transformed-false_seed-021"
)
checksums <- utils::read.csv(checksums_path, stringsAsFactors = FALSE)
row <- checksums[checksums$job_id == expected_job, , drop = FALSE]
if (nrow(row) != 1L) stop("Expected one checksum row for ", expected_job)
actual_hash <- tolower(unname(tools::sha256sum(boosting_path)))
actual_bytes <- unname(file.info(boosting_path)$size)
if (!identical(actual_hash, tolower(row$output_sha256[[1L]])) ||
    !identical(as.numeric(actual_bytes), as.numeric(row$output_bytes[[1L]]))) {
  stop("Canonical boosting RDS does not match its checksum record")
}
expected_model_hash <- "ad2090da5fbc01123e65687b1364a6064f2e5b3b17824ad3b9c4df7152e6eb40"
model_hash <- tolower(unname(tools::sha256sum(model_path)))
if (!identical(model_hash, expected_model_hash)) stop("20D model source hash mismatch")

legacy <- readRDS(legacy_path)
boosting <- readRDS(boosting_path)
config <- boosting$metadata$config
if (!identical(boosting$metadata$status, "CANONICAL") ||
    !identical(boosting$metadata$job_id, expected_job) ||
    !identical(config$family, "20d") ||
    !identical(config$scenario, "latent_location_shift") ||
    !identical(config$n0, 5000L) || !identical(config$n1, 5000L) ||
    !identical(config$transformed, FALSE) ||
    !identical(config$seed_map$data, 21L)) stop("Canonical Figure 4 configuration mismatch")

source(model_path, local = TRUE)
set.seed(21L)
generated <- simulation_multi_latent(5000L, 5000L, 20L, "latent_location_shift", 0.2, FALSE)
coordinates <- as.matrix(generated$data_u)
if (nrow(coordinates) != 10000L || ncol(coordinates) < 2L) {
  stop("Unexpected latent-coordinate dimensions")
}

finite_vector <- function(x, label) {
  x <- as.numeric(x)
  if (length(x) != 10000L || any(!is.finite(x))) stop("Invalid vector: ", label)
  x
}
pick_quantile <- function(x, probability) {
  probabilities <- as.numeric(sub("%", "", rownames(x), fixed = TRUE)) / 100
  as.numeric(x[which.min(abs(probabilities - probability)), ])
}
truth <- finite_vector(boosting$result$design$truth_train, "truth")
generated_truth <- finite_vector(generated$true_log_w_obs, "regenerated truth")
truth_difference <- max(abs(truth - generated_truth))
if (truth_difference > 1e-12) stop("Regenerated Figure 4 data do not align with canonical truth")

values <- list(
  "Truth" = truth,
  "DRT (AdaBoost)" = finite_vector(boosting$result$adaboost$estimates$train$exponential_loss, "DRT"),
  "KLIEP" = finite_vector(legacy$log_ratio_KLIEP_data, "KLIEP"),
  "uLSIF" = finite_vector(legacy$log_ratio_uLSIF_data, "uLSIF"),
  "GB" = finite_vector(boosting$result$proposed$estimates$gb$train, "GB"),
  "Bayesian additive trees" = finite_vector(legacy$log_ratio_BART_data_mean, "BAT mean"),
  "Lower (2.5%)" = finite_vector(pick_quantile(legacy$log_ratio_BART_data_quantiles, 0.025), "BAT lower"),
  "Upper (97.5%)" = finite_vector(pick_quantile(legacy$log_ratio_BART_data_quantiles, 0.975), "BAT upper")
)
if (any(values[[7L]] > values[[8L]])) stop("BAT lower quantile exceeds upper quantile")

legacy_gb <- finite_vector(legacy$log_ratio_boosting_grad_data, "legacy GB scale reference")
minimum_true <- min(truth); maximum_true <- max(truth)
minimum_estimate <- min(legacy_gb); maximum_estimate <- max(legacy_gb)
threshold <- 0.5
minimum_display <- min((minimum_true + minimum_estimate) / 2, -threshold)
maximum_display <- max((maximum_true + maximum_estimate) / 2, threshold)
minimum_palette <- min(minimum_true, minimum_estimate, -threshold) - 1e-10
maximum_palette <- max(maximum_true, maximum_estimate, threshold) + 1e-10
palette_colours <- c("darkblue", "blue", "blue", "deepskyblue", "cyan", "#7FFFFF",
  "white", "white", "white", "#FFFF99", "yellow", "orange", "red", "red", "darkred")
palette_locations <- scales::rescale(c(minimum_palette, minimum_display,
  minimum_display * 8/10, minimum_display * 6/10, minimum_display * 4/10,
  minimum_display * 2/10, minimum_display * 1/10, 0, maximum_display * 1/10,
  maximum_display * 2/10, maximum_display * 4/10, maximum_display * 6/10,
  maximum_display * 8/10, maximum_display, maximum_palette))

plot_data <- do.call(rbind, lapply(names(values), function(method) data.frame(
  z1 = coordinates[, 1L], z2 = coordinates[, 2L], log_weight = values[[method]], method = method
)))
plot_data$method <- factor(plot_data$method, levels = names(values))
plot_data <- plot_data[sample.int(nrow(plot_data)), , drop = FALSE]
pad <- function(x) { r <- range(x); r + c(-1, 1) * diff(r) * 0.04 }

list(data=plot_data,xlim=pad(coordinates[,1L]),ylim=pad(coordinates[,2L]),
 colours=palette_colours,positions=palette_locations,limits=c(minimum_display,maximum_display),
 axes=c("z1","z2"),xlab=expression(z[1]),ylab=expression(z[2]),
 hashes=c(boosting_sha256=actual_hash,legacy_detail_sha256=tolower(unname(tools::sha256sum(legacy_path))),model_sha256=model_hash),
 truth_difference=truth_difference)
}
candidate_path <- function(name) {
 if (basename(name) != name || grepl("[\\\\/]",name)) stop("Candidate filename must not contain directories")
 file.path(outdir,name)
}
render_bundle <- function(b,id,height_cm,read_only_reference_metadata) {
 # Existing metadata is read only to verify source hashes; never written.
 old <- readLines(read_only_reference_metadata)
 for (nm in names(b$hashes)) stopifnot(paste0(nm,"=",b$hashes[[nm]]) %in% old)
 plot_data <- b$data; axis_limits <- list(x=b$xlim,y=b$ylim)
 palette_colours <- b$colours; palette_locations <- b$positions
 minimum_display <- b$limits[1]; maximum_display <- b$limits[2]
base_theme <- ggplot2::theme_gray(base_size = 9, base_family = "sans") +
  ggplot2::theme(
    panel.grid.minor = ggplot2::element_blank(),
    panel.border = ggplot2::element_rect(
      fill = NA, colour = "#4D4D4D", linewidth = 0.25
    ),
    panel.background = ggplot2::element_rect(fill = "#EBEBEB", colour = NA),
    strip.background = ggplot2::element_rect(fill = "#D9D9D9", colour = "#4D4D4D", linewidth = 0.25),
    strip.text = ggplot2::element_text(size = 9, face = "bold", margin = ggplot2::margin(3, 2, 3, 2)),
    axis.title = ggplot2::element_text(size = 9),
    axis.text = ggplot2::element_text(size = 8, colour = "#333333"),
    legend.title = ggplot2::element_text(size = 9),
    legend.text = ggplot2::element_text(size = 8),
    plot.margin = ggplot2::margin(3, 3, 3, 3)
  )

final_plot <- ggplot2::ggplot(
  plot_data,
  ggplot2::aes(x = .data[[b$axes[1]]], y = .data[[b$axes[2]]], colour = log_weight)
) +
  ggplot2::geom_point(size = 0.7, alpha = 0.75, stroke = 0) +
  ggplot2::scale_colour_gradientn(
    colours = palette_colours,
    values = palette_locations,
    limits = c(minimum_display, maximum_display),
    oob = scales::squish,
    name = "log(density\nratio)",
    guide = ggplot2::guide_colourbar(
      barwidth = grid::unit(0.35, "cm"),
      barheight = grid::unit(2.6, "cm"),
      title.position = "top",
      title.hjust = 0.5
    )
  ) +
  ggplot2::facet_wrap(~method, nrow = 2L, ncol = 4L,
    labeller = ggplot2::as_labeller(c("Truth"="Truth", "DRT (AdaBoost)"="DRT\n(AdaBoost)",
      "KLIEP"="KLIEP", "uLSIF"="uLSIF", "GB"="GB",
      "Bayesian additive trees"="Bayesian\nadditive trees",
      "Lower (2.5%)"="Lower\n(2.5%)", "Upper (97.5%)"="Upper\n(97.5%)"))) +
  ggplot2::coord_equal(
    xlim = axis_limits$x, ylim = axis_limits$y, expand = FALSE, clip = "on"
  ) +
  ggplot2::labs(x = b$xlab, y = b$ylab) +
  base_theme +
  ggplot2::theme(legend.position = "right")

 saveRDS(b,candidate_path(paste0(id,"_plotted_data.rds")))
 ggplot2::ggsave(candidate_path(paste0(id,".png")),final_plot,width=13.5,height=height_cm,units="cm",dpi=300,bg="white")
 ggplot2::ggsave(candidate_path(paste0(id,".pdf")),final_plot,width=13.5,height=height_cm,units="cm",device=grDevices::cairo_pdf,bg="white")
 writeLines(c(paste0("figure=",id),"width_cm=13.5",paste0("height_cm=",height_cm),
 "ticks_and_colorbar_pt=8","headings_and_axis_titles_pt=9","point_size=0.7","point_alpha=0.75",
 "source_hashes_match_original_figure=true","estimator_fitting=false",
 paste0("truth_alignment_max_abs_difference=",b$truth_difference),
 paste0(names(b$hashes),"=",b$hashes)),candidate_path(paste0(id,"_validation.txt")))
 cat("Saved ",id,"\n",sep="")
}
dir.create(outdir,recursive=TRUE)
for (setting in settings) {
 b <- prepare_setting(setting)
 # Validate exact repeatability of all plotted values, ordering and mappings.
 stopifnot(identical(b,prepare_setting(setting)))
 h <- if (setting$scenario=="local_shift") 9.3 else if (setting$scenario=="local_dispersion") 7.2 else 8.5
 render_bundle(b,setting$id,h,file.path(repo_root,"output/figures",paste0("supplement_",setting$id,"_r2_preview_metadata.txt")))
}
b <- prepare_s5(); stopifnot(identical(b,prepare_s5()))
render_bundle(b,"s5_latent_location_balanced",8.0,file.path(repo_root,"output/figures/figure4_r2_preview_metadata.txt"))
