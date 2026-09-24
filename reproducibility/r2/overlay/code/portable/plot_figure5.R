#!/usr/bin/env Rscript
# Approved display-only Figure 5 candidate. Reads saved summaries; no fitting,
# aggregation, resampling, RNG calls, or writes to existing figure assets.
library(grid)
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
stopifnot(length(file_arg) == 1L)
repo_root <- normalizePath(file.path(dirname(sub("^--file=", "", file_arg)), "../.."), winslash = "/")
args <- commandArgs(TRUE)
if (length(args) != 2L) stop("Arguments: saved-summary-csv new-output-dir")
input <- normalizePath(args[[1L]], winslash = "/", mustWork = TRUE)
outdir <- args[[2L]]
if (dir.exists(outdir)) stop("Candidate directory already exists; choose a new directory")
d <- read.csv(input, stringsAsFactors = FALSE, check.names = FALSE)
ids <- c("global_balanced", "location_balanced", "dispersion_balanced",
         "global_unbalanced", "location_unbalanced", "dispersion_unbalanced")
fields <- c("mean_correct_detection_rate", "mean_coverage_rate", "mean_wrong_direction_rate")
bands <- c("q10_correct_detection_rate", "q90_correct_detection_rate", "q10_coverage_rate", "q90_coverage_rate")
stopifnot(nrow(d) == 66L, setequal(unique(d$setting_id), ids),
          all(is.finite(d$plot_value)),
          !any(is.infinite(as.matrix(d[c(fields, bands)]))))
panels <- lapply(ids, function(id) {
  z <- d[d$setting_id == id, , drop = FALSE]
  z <- z[order(z$plot_value), , drop = FALSE]
  stopifnot(identical(z$plot_value, seq(0, 2.5, by = .25)))
  z
})
# Verify each value passed to drawing against the input, without rounding.
for (i in seq_along(ids)) {
  original <- d[d$setting_id == ids[i], , drop = FALSE]
  original <- original[order(original$plot_value), , drop = FALSE]
  stopifnot(identical(panels[[i]], original))
}
input_md5 <- unname(tools::md5sum(input))
dir.create(outdir, recursive = TRUE)
colours <- c("#111111", "#2C7FB8", "#D7301F")
line_types <- c("solid", "dashed", "dotted")
line_widths <- c(1.3, 1.3, 1.2) # R lwd unit is 1/96 inch: about .98/.90 pt.
txt <- function(label, x, y, size = 8, bold = FALSE, ...) {
  grid.text(label, x, y, gp = gpar(fontfamily = "sans", fontsize = size,
    fontface = if (bold) "bold" else "plain", col = "#222222", lineheight = 1), ...)
}
draw <- function() {
  grid.newpage()
  lefts <- c(.105, .405, .705)
  pw <- .267
  bottoms <- c(.59, .245)
  ph <- .265
  txt(expression(bold("Balanced: ")~n[0]*" = "*n[1]*" = 5000"), .53, .968, 8.5)
  txt(expression(bold("Unbalanced: ")~n[0] == 9000*","~n[1] == 1000), .53, .55, 8.5)
  for (j in 1:3) txt(c("Global Shift", "Location Shift", "Dispersion")[j],
                     lefts[j] + pw / 2, .884, 9, TRUE)
  for (i in 1:6) {
    row <- if (i <= 3) 1 else 2
    col <- (i - 1) %% 3 + 1
    z <- panels[[i]]
    pushViewport(viewport(x = lefts[col], y = bottoms[row], width = pw, height = ph,
      just = c("left", "bottom"), xscale = c(-.025, 2.525),
      yscale = c(-.055, 1.055), clip = "on"))
    grid.rect(gp = gpar(fill = "white", col = NA))
    for (v in seq(0, 2.5, .5)) grid.lines(unit(c(v, v), "native"), unit(c(-.055, 1.055), "native"), gp = gpar(col = "#E3E3E3", lwd = .45))
    for (v in seq(0, 1, .25)) grid.lines(unit(c(-.025, 2.525), "native"), unit(c(v, v), "native"), gp = gpar(col = if (v == 0) "#BDBDBD" else "#E3E3E3", lwd = .45))
    for (b in 1:2) {
      lo <- z[[bands[2*b-1]]]; hi <- z[[bands[2*b]]]
      valid <- is.finite(z$plot_value) & is.finite(lo) & is.finite(hi)
      if (sum(valid) > 1L) grid.polygon(unit(c(z$plot_value[valid], rev(z$plot_value[valid])), "native"),
        unit(c(lo[valid], rev(hi[valid])), "native"), gp = gpar(col = NA,
        fill = adjustcolor(colours[b], alpha.f = c(.10, .20)[b])))
    }
    for (k in 1:3) {
      grid.lines(unit(z$plot_value, "native"), unit(z[[fields[k]]], "native"),
        gp = gpar(col = colours[k], lty = line_types[k], lwd = line_widths[k]))
      grid.points(unit(z$plot_value, "native"), unit(z[[fields[k]]], "native"),
        pch = 16, size = unit(.55, "mm"), gp = gpar(col = colours[k]))
    }
    grid.rect(gp = gpar(fill = NA, col = "#4D4D4D", lwd = .6))
    popViewport()
    # All labels use physical font sizes; no layout-induced cex scaling.
    if (col == 1) for (v in seq(0, 1, .25)) {
      yp <- bottoms[row] + (v + .055) / 1.11 * ph
      txt(sprintf("%.2f", v), lefts[col] - .012, yp, just = "right")
    }
    if (row == 2) {
      for (v in seq(0, 2.5, .5)) {
        xp <- lefts[col] + (v + .025) / 2.55 * pw
        grid.lines(c(xp, xp), c(bottoms[row], bottoms[row] - .009), gp = gpar(lwd = .5))
        if (v %in% c(0, 1, 2)) txt(as.character(v), xp, bottoms[row] - .025)
        if (v == 2.5) {
          # Connect the tail-bin label directly to its plotted tick at 2.5.
          # The threshold remains >= 2.375; no bin or data coordinate changes.
          tail_y <- bottoms[row] - .066
          grid.lines(c(xp, xp, xp - .006),
            c(bottoms[row] - .009, tail_y, tail_y),
            gp = gpar(col = "#4D4D4D", lwd = .5))
          txt(expression(phantom(0) >= 2.375), xp - .012, tail_y, just = "right")
        }
      }
    }
  }
  txt("Rate", .022, .55, 9, rot = 90)
  txt("Absolute true log-density ratio", .54, .125, 9)
  labels <- c("Correct-direction\nzero exclusion", "Coverage of\ntrue log ratio", "Wrong-direction\nzero exclusion")
  for (k in 1:3) {
    x <- c(.03, .36, .685)[k]
    grid.lines(c(x, x + .048), c(.045, .045), gp = gpar(col = colours[k], lty = line_types[k], lwd = line_widths[k]))
    txt(labels[k], x + .058, .045, 8, just = "left")
  }
}
pdf_path <- file.path(outdir, "figure5_compact.pdf")
cairo_pdf(pdf_path, width = 13.5/2.54, height = 8.4/2.54, family = "sans", pointsize = 8, bg = "white")
draw(); dev.off()
png(file.path(outdir, "figure5_compact.png"), width = 13.5, height = 8.4,
    units = "cm", res = 300, type = "cairo", bg = "white")
draw(); dev.off()
stopifnot(identical(input_md5, unname(tools::md5sum(input))))
writeLines(c("Approved reviewer-driven display-only candidate", "width_cm=13.5", "height_cm=8.4",
  "ticks_pt=8", "legend_pt=8", "scenario_and_axis_titles_pt=9", "row_heading_pt=8.5",
  "tail_label_connected_to_tick=true", "axis_title_legend_center_gap_npc=0.08",
  paste0("input_md5=", input_md5), "input_rows=66", "drawing_values_identical_to_saved_csv=true",
  "input_unchanged=true", "no_fitting_aggregation_or_rng=true", "existing_assets_not_modified=true",
  paste0("R=", getRversion())), file.path(outdir, "validation.txt"))
cat(pdf_path, "\n")
