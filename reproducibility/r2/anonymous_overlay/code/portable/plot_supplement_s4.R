#!/usr/bin/env Rscript
# Figure S4 rendering from saved coverage results, using the Figure 4 style.
# Original saved-input selection and mean validation are copied verbatim below.
library(grid)
args <- commandArgs(TRUE)
if (length(args) != 2L) stop("Arguments: coverage-summary-dir new-output-dir")
seed_path <- file.path(args[1], "coverage_by_seed.csv")
summary_path <- file.path(args[1], "coverage_by_nominal_mass.csv")
metadata_input_path <- file.path(args[1], "summary_metadata.txt")
outdir <- args[2]
if (dir.exists(outdir)) stop("Output directory exists; choose a new candidate directory")
input_paths <- c(seed_path, summary_path, metadata_input_path)
input_hashes <- unname(tools::sha256sum(input_paths))
stopifnot(identical(input_hashes, c(
 "f7e73e34f82ddd8bb5efc188f1e0783cbc391412d1dee26d1ac7ca1aaad2a2aa",
 "dec443b128389acd2d1f1e4f0bfc60ea5f388098da512ecb113507f59975d876",
 "e8b4bfa61622f93548e88d164ad9952b0c0dce7cc5d8c3ec8d00fddd5802e797")))
seed_all <- utils::read.csv(seed_path, stringsAsFactors = FALSE)
summary_all <- utils::read.csv(summary_path, stringsAsFactors = FALSE)
seed_required <- c(
  "family", "scenario", "n0", "n1", "representation", "seed",
  "nominal_mass", "group", "coverage"
)
summary_required <- c(
  "family", "scenario", "n0", "n1", "representation", "group",
  "nominal_mass", "n_seeds", "mean_coverage"
)
if (!all(seed_required %in% names(seed_all)) ||
    !all(summary_required %in% names(summary_all))) {
  stop("Coverage summaries do not have the required columns")
}

keep_seed <- seed_all$family == "2d" &
  seed_all$representation == "not_applicable" &
  seed_all$group == "all_pooled"
keep_summary <- summary_all$family == "2d" &
  summary_all$representation == "not_applicable" &
  summary_all$group == "all_pooled"
seed_data <- seed_all[keep_seed, seed_required, drop = FALSE]
mean_data <- summary_all[keep_summary, summary_required, drop = FALSE]

scenario_levels <- c("global_shift", "local_shift", "local_dispersion")
expected_sizes <- data.frame(
  n0 = c(5000L, 9000L), n1 = c(5000L, 1000L),
  size_label = c(
    "Balanced (n0 = n1 = 5000)",
    "Unbalanced (n0 = 9000, n1 = 1000)"
  ),
  stringsAsFactors = FALSE
)
expected_nominal <- seq(0.01, 0.99, by = 0.01)
if (nrow(seed_data) != 3L * 2L * 50L * 99L ||
    nrow(mean_data) != 3L * 2L * 99L ||
    !setequal(unique(seed_data$scenario), scenario_levels) ||
    !identical(sort(unique(seed_data$seed)), 1:50) ||
    max(abs(sort(unique(seed_data$nominal_mass)) - expected_nominal)) > 1e-12 ||
    any(!is.finite(seed_data$coverage)) ||
    any(seed_data$coverage < 0 | seed_data$coverage > 1) ||
    any(mean_data$n_seeds != 50L)) {
  stop("Corrected 2D coverage input failed structural validation")
}

cell_sizes <- unique(seed_data[c("n0", "n1")])
if (nrow(merge(cell_sizes, expected_sizes[c("n0", "n1")])) != 2L ||
    nrow(cell_sizes) != 2L) {
  stop("Unexpected 2D sample-size settings")
}

# Independently reproduce every saved mean from the 50 seed-level curves.
recomputed <- stats::aggregate(
  coverage ~ scenario + n0 + n1 + nominal_mass,
  data = seed_data,
  FUN = mean
)
comparison <- merge(
  recomputed,
  mean_data[c("scenario", "n0", "n1", "nominal_mass", "mean_coverage")],
  by = c("scenario", "n0", "n1", "nominal_mass"),
  all = TRUE
)
mean_max_abs_difference <- max(abs(comparison$coverage - comparison$mean_coverage))
if (nrow(comparison) != 3L * 2L * 99L ||
    !is.finite(mean_max_abs_difference) || mean_max_abs_difference > 1e-12) {
  stop("Saved mean curves do not match the seed-level results")
}

scenario_labels <- c(
  global_shift = "Global Shift",
  local_shift = "Local Shift",
  local_dispersion = "Local Dispersion"
)
add_facet_labels <- function(x) {
  x$scenario_label <- factor(
    scenario_labels[x$scenario], levels = unname(scenario_labels)
  )
  size_key <- paste(x$n0, x$n1, sep = "_")
  x$size_label <- factor(
    c(
      "5000_5000" = expected_sizes$size_label[[1L]],
      "9000_1000" = expected_sizes$size_label[[2L]]
    )[size_key],
    levels = expected_sizes$size_label
  )
  if (anyNA(x$scenario_label) || anyNA(x$size_label)) {
    stop("Failed to construct facet labels")
  }
  x
}
seed_data <- add_facet_labels(seed_data)
mean_data <- add_facet_labels(mean_data)
seed_data$curve_id <- interaction(
  seed_data$scenario, seed_data$n0, seed_data$n1, seed_data$seed,
  drop = TRUE
)

# Use explicit physical typography, avoiding base layout's implicit cex scaling.
txt <- function(label, x, y, size = 8, bold = FALSE, ...) {
  grid.text(label, x, y, gp = gpar(fontfamily = "sans", fontsize = size,
    fontface = if (bold) "bold" else "plain", col = "#222222"), ...)
}
draw <- function() {
  grid.newpage()
  lefts <- c(.145, .428, .711)
  pw <- .246
  ph <- pw * 10.5 / 8.4 # Equal physical x/y scales; panels remain square.
  bottoms <- c(.565, .19)
  ticks <- seq(0, 1, .25)
  txt(expression(bold("Balanced: ")~n[0]*" = "*n[1]*" = 5000"), .54, .967, 8.5)
  txt(expression(bold("Unbalanced: ")~n[0] == 9000*","~n[1] == 1000), .54, .529, 8.5)
  for (j in 1:3) txt(unname(scenario_labels[j]), lefts[j]+pw/2, .904, 9, TRUE)
  for (r in 1:2) for (j in 1:3) {
    current_n0 <- expected_sizes$n0[r]; current_n1 <- expected_sizes$n1[r]
    current_scenario <- scenario_levels[j]
    cell_seed <- seed_data[seed_data$scenario == current_scenario &
      seed_data$n0 == current_n0 & seed_data$n1 == current_n1, , drop = FALSE]
    cell_mean <- mean_data[mean_data$scenario == current_scenario &
      mean_data$n0 == current_n0 & mean_data$n1 == current_n1, , drop = FALSE]
    cell_mean <- cell_mean[order(cell_mean$nominal_mass), , drop = FALSE]
    pushViewport(viewport(x=lefts[j], y=bottoms[r], width=pw, height=ph,
      just=c("left", "bottom"), xscale=c(0,1), yscale=c(0,1), clip="on"))
    grid.rect(gp=gpar(fill="white",col=NA))
    for (v in ticks) {
      grid.lines(c(v,v), c(0,1), gp=gpar(col="#E3E3E3", lwd=.45))
      grid.lines(c(0,1), c(v,v), gp=gpar(col="#E3E3E3", lwd=.45))
    }
    # Preserve original drawing order: identity, all 50 repeats, mean.
    grid.lines(c(0,1), c(0,1), gp=gpar(col="#555555",lty="dashed",lwd=.9))
    for (seed in 1:50) {
      curve <- cell_seed[cell_seed$seed == seed, , drop=FALSE]
      curve <- curve[order(curve$nominal_mass), , drop=FALSE]
      stopifnot(nrow(curve)==99L)
      grid.lines(curve$nominal_mass, curve$coverage, gp=gpar(col="#C7C7C7",lwd=.5))
    }
    grid.lines(cell_mean$nominal_mass, cell_mean$mean_coverage, gp=gpar(col="#111111",lwd=1.3))
    grid.rect(gp=gpar(fill=NA,col="#4D4D4D",lwd=.6))
    popViewport()
    if (j == 1) for (v in ticks) {
      yp <- bottoms[r]+v*ph
      txt(sprintf("%.2f",v),lefts[j]-.017,yp,just="right")
    }
    if (r == 2) for (v in ticks) {
      xp <- lefts[j]+v*pw
      grid.lines(c(xp,xp),c(bottoms[r],bottoms[r]-.009),gp=gpar(lwd=.5))
      if (v %in% c(0,.5,1)) txt(as.character(v),xp,bottoms[r]-.027)
    }
  }
  txt("Empirical coverage rate",.022,.55,9,rot=90)
  txt("Nominal credible level",.54,.103,9)
  for (k in 1:3) {
    x <- c(.20,.43,.665)[k]
    grid.lines(c(x,x+.05),c(.035,.035),gp=gpar(
      col=c("#111111","#A6A6A6","#555555")[k],lty=c(1,1,2)[k],lwd=c(1.3,.9,.9)[k]))
    txt(c("Average","Repeats","Nominal")[k],x+.06,.035,just="left")
  }
}
dir.create(outdir, recursive=TRUE)
cairo_pdf(file.path(outdir,"figureS4_compact.pdf"),width=10.5/2.54,height=8.4/2.54,
  family="sans",pointsize=8,bg="white")
draw(); dev.off()
png(file.path(outdir,"figureS4_compact.png"),width=10.5,height=8.4,units="cm",
  res=300,type="cairo",bg="white")
draw(); dev.off()
stopifnot(identical(input_hashes,unname(tools::sha256sum(input_paths))))
writeLines(c("Figure S4 rendered from saved coverage results",
  "width_cm=10.5", "height_cm=8.4", "ticks_and_legend_pt=8", "axis_and_scenario_titles_pt=9",
  "row_headings_pt=8.5", "square_panels=true", "panels=6", "repeat_curves_per_panel=50",
  "levels_per_curve=99", "original_data_preparation_and_mean_calculation_preserved=true",
  "input_hashes_match_original_figure_metadata=true", "inputs_unchanged=true",
  "estimator_fitting=false", "rng_consumed=false",
  paste0("mean_recalculation_max_abs_difference=",format(mean_max_abs_difference,scientific=TRUE)),
  paste0("input_sha256=",input_hashes)),file.path(outdir,"validation.txt"))

