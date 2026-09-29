#!/usr/bin/env Rscript
# Reviewer-requested display-only Figure 4 candidate. Original data preparation
# copied verbatim below; no fits, raw-data recomputation or random-number calls.
library(grid)
args <- commandArgs(TRUE)
if (length(args) != 3L) stop("Arguments: global-null-summary-dir coverage-summary-dir new-output-dir")
global_null_path <- file.path(args[1], "seed_calibration_curves.csv")
coverage_path <- file.path(args[2], "coverage_by_seed.csv")
outdir <- args[3]
if (dir.exists(outdir)) stop("Output directory exists; choose a new candidate directory")
input_paths <- c(global_null_path, coverage_path)
input_hashes <- unname(tools::sha256sum(input_paths))
stopifnot(identical(input_hashes, c(
 "1679dd6db36806c4c97b900ac3541bc7c63d2d41811f1cccd8a6e7a4bafeef44",
 "f7e73e34f82ddd8bb5efc188f1e0783cbc391412d1dee26d1ac7ca1aaad2a2aa")))
global_null_all <- utils::read.csv(
  global_null_path, stringsAsFactors = FALSE
)
coverage_all <- utils::read.csv(coverage_path, stringsAsFactors = FALSE)
global_null_required <- c(
  "scenario", "n0", "n1", "seed", "group", "nominal_level",
  "empirical_coverage"
)
coverage_required <- c(
  "family", "scenario", "n0", "n1", "representation", "seed",
  "nominal_mass", "group", "coverage"
)
if (!all(global_null_required %in% names(global_null_all)) ||
    !all(coverage_required %in% names(coverage_all))) {
  stop("Calibration summaries do not have the required columns")
}

global_null <- global_null_all[
  global_null_all$scenario %in% c("global_shift", "null") &
    global_null_all$group == "all",
  global_null_required,
  drop = FALSE
]
names(global_null)[names(global_null) == "nominal_level"] <- "nominal_mass"
names(global_null)[names(global_null) == "empirical_coverage"] <- "coverage"

location_dispersion <- coverage_all[
  coverage_all$family == "20d" &
    coverage_all$scenario %in% c(
      "latent_location_shift", "latent_dispersion"
    ) &
    coverage_all$representation == "raw" &
    coverage_all$group == "all_pooled",
  c("scenario", "n0", "n1", "seed", "nominal_mass", "coverage"),
  drop = FALSE
]

seed_data <- rbind(
  global_null[c(
    "scenario", "n0", "n1", "seed", "nominal_mass", "coverage"
  )],
  location_dispersion
)
# The two independently written CSVs can differ at machine precision even
# though both grids are 0.01, ..., 0.99. Restore their documented grid exactly.
seed_data$nominal_mass <- round(seed_data$nominal_mass, digits = 2L)
scenario_levels <- c(
  "global_shift", "latent_location_shift", "latent_dispersion", "null"
)
expected_sizes <- data.frame(
  n0 = c(5000L, 9000L),
  n1 = c(5000L, 1000L),
  stringsAsFactors = FALSE
)
expected_nominal <- seq(0.01, 0.99, by = 0.01)
expected_rows <- length(scenario_levels) * nrow(expected_sizes) * 50L * 99L
if (nrow(seed_data) != expected_rows ||
    !setequal(unique(seed_data$scenario), scenario_levels) ||
    !identical(sort(unique(seed_data$seed)), 1:50) ||
    max(abs(sort(unique(seed_data$nominal_mass)) - expected_nominal)) > 1e-12 ||
    any(!is.finite(seed_data$coverage)) ||
    any(seed_data$coverage < 0 | seed_data$coverage > 1)) {
  stop("Combined 20D calibration input failed structural validation")
}

cell_counts <- stats::aggregate(
  seed ~ scenario + n0 + n1 + nominal_mass,
  data = seed_data,
  FUN = function(x) length(unique(x))
)
if (nrow(cell_counts) != length(scenario_levels) * 2L * 99L ||
    any(cell_counts$seed != 50L)) {
  stop("Every panel and nominal level must contain exactly 50 seeds")
}
observed_sizes <- unique(seed_data[c("n0", "n1")])
if (nrow(observed_sizes) != 2L ||
    nrow(merge(observed_sizes, expected_sizes, by = c("n0", "n1"))) != 2L) {
  stop("Unexpected 20D sample-size settings")
}

mean_data <- stats::aggregate(
  coverage ~ scenario + n0 + n1 + nominal_mass,
  data = seed_data,
  FUN = mean
)
scenario_labels <- c(
  global_shift = "Global Shift",
  latent_location_shift = "Location Shift",
  latent_dispersion = "Dispersion",
  null = "Null"
)

# Use explicit physical typography, avoiding base layout's implicit cex scaling.
txt <- function(label, x, y, size = 8, bold = FALSE, ...) {
  grid.text(label, x, y, gp = gpar(fontfamily = "sans", fontsize = size,
    fontface = if (bold) "bold" else "plain", col = "#222222"), ...)
}
draw <- function() {
  grid.newpage()
  lefts <- c(.11, .333, .556, .779)
  pw <- .191
  ph <- pw * 13.5 / 8.4 # Equal physical x/y scales; panels remain square.
  bottoms <- c(.565, .19)
  ticks <- seq(0, 1, .25)
  txt(expression(bold("Balanced: ")~n[0]*" = "*n[1]*" = 5000"), .54, .967, 8.5)
  txt(expression(bold("Unbalanced: ")~n[0] == 9000*","~n[1] == 1000), .54, .529, 8.5)
  for (j in 1:4) txt(unname(scenario_labels[j]), lefts[j]+pw/2, .904, 9, TRUE)
  for (r in 1:2) for (j in 1:4) {
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
    grid.lines(cell_mean$nominal_mass, cell_mean$coverage, gp=gpar(col="#111111",lwd=1.3))
    grid.rect(gp=gpar(fill=NA,col="#4D4D4D",lwd=.6))
    popViewport()
    if (j == 1) for (v in ticks) {
      yp <- bottoms[r]+v*ph
      txt(sprintf("%.2f",v),lefts[j]-.013,yp,just="right")
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
    x <- c(.25,.43,.62)[k]
    grid.lines(c(x,x+.05),c(.035,.035),gp=gpar(
      col=c("#111111","#A6A6A6","#555555")[k],lty=c(1,1,2)[k],lwd=c(1.3,.9,.9)[k]))
    txt(c("Average","Repeats","Nominal")[k],x+.06,.035,just="left")
  }
}
dir.create(outdir, recursive=TRUE)
cairo_pdf(file.path(outdir,"figure4_compact.pdf"),width=13.5/2.54,height=8.4/2.54,
  family="sans",pointsize=8,bg="white")
draw(); dev.off()
png(file.path(outdir,"figure4_compact.png"),width=13.5,height=8.4,units="cm",
  res=300,type="cairo",bg="white")
draw(); dev.off()
stopifnot(identical(input_hashes,unname(tools::sha256sum(input_paths))))
writeLines(c("Figure 4 reviewer-requested display-only candidate",
  "width_cm=13.5", "height_cm=8.4", "ticks_and_legend_pt=8", "axis_and_scenario_titles_pt=9",
  "row_headings_pt=8.5", "square_panels=true", "panels=8", "repeat_curves_per_panel=50",
  "levels_per_curve=99", "original_data_preparation_and_mean_calculation_preserved=true",
  "input_hashes_match_original_figure_metadata=true", "inputs_unchanged=true",
  "estimator_fitting=false", "rng_consumed=false",
  paste0("input_sha256=",input_hashes)),file.path(outdir,"validation.txt"))

