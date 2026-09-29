#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_path <- if (length(file_arg)) {
  normalizePath(sub("^--file=", "", file_arg[[1L]]), winslash = "/")
} else {
  normalizePath("scripts/figures/plot_eight_panel_equal_size_preview.R", winslash = "/")
}
repo_root <- normalizePath(file.path(dirname(script_path), "..", ".."), winslash = "/")
outdir <- file.path(repo_root, "output", "figures", "eight-panel-equal-size-preview")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

main_source <- file.path(
  repo_root, "output", "figures", "figure3-compact-preview", "plotted_data.rds"
)
supplement_dir <- file.path(
  repo_root, "output", "figures", "supplement-eight-panel-final-preview"
)

specifications <- list(
  list(id = "figure3", path = main_source, main = TRUE),
  list(id = "s2_global_shift_balanced", main = FALSE),
  list(id = "s2_local_dispersion_balanced", main = FALSE),
  list(id = "s3_global_shift_unbalanced", main = FALSE),
  list(id = "s3_local_shift_unbalanced", main = FALSE),
  list(id = "s3_local_dispersion_unbalanced", main = FALSE),
  list(id = "s5_latent_location_balanced", main = FALSE)
)
for (i in seq_along(specifications)) {
  if (!isTRUE(specifications[[i]]$main)) {
    specifications[[i]]$path <- file.path(
      supplement_dir, paste0(specifications[[i]]$id, "_plotted_data.rds")
    )
  }
}
stopifnot(all(vapply(specifications, function(x) file.exists(x$path), logical(1))))

normalise_bundle <- function(specification) {
  source_bundle <- readRDS(specification$path)
  if (isTRUE(specification$main)) {
    list(
      id = specification$id,
      data = source_bundle$plot_data,
      xlim = source_bundle$axis_limits$x,
      ylim = source_bundle$axis_limits$y,
      colours = source_bundle$palette_colours,
      positions = source_bundle$palette_locations,
      limits = source_bundle$limits,
      axes = c("x1", "x2"),
      xlab = expression(x[1]),
      ylab = expression(x[2]),
      main = TRUE,
      source_path = specification$path
    )
  } else {
    list(
      id = specification$id,
      data = source_bundle$data,
      xlim = source_bundle$xlim,
      ylim = source_bundle$ylim,
      colours = source_bundle$colours,
      positions = source_bundle$positions,
      limits = source_bundle$limits,
      axes = source_bundle$axes,
      xlab = source_bundle$xlab,
      ylab = source_bundle$ylab,
      main = FALSE,
      source_path = specification$path
    )
  }
}

bundles <- lapply(specifications, normalise_bundle)
reference <- bundles[[which(vapply(
  bundles, function(x) identical(x$id, "s2_global_shift_balanced"), logical(1)
))]]
target_aspect <- diff(reference$xlim) / diff(reference$ylim)
# The reference S2 global-shift panel measures 2.47 cm wide at 13.5 cm output.
# Its height follows from coord_equal() and the reference coordinate aspect.
target_panel_width_cm <- 2.47
target_panel_height_cm <- target_panel_width_cm / target_aspect
output_width_cm <- 13.5
output_height_cm <- 8.7

expand_limits_to_aspect <- function(xlim, ylim, aspect) {
  x_span <- diff(xlim)
  y_span <- diff(ylim)
  if (x_span / y_span < aspect) {
    x_half_span <- aspect * y_span / 2
    xlim <- mean(xlim) + c(-x_half_span, x_half_span)
  } else if (x_span / y_span > aspect) {
    y_half_span <- x_span / aspect / 2
    ylim <- mean(ylim) + c(-y_half_span, y_half_span)
  }
  list(x = xlim, y = ylim)
}

base_theme <- ggplot2::theme_gray(base_size = 9, base_family = "sans") +
  ggplot2::theme(
    panel.grid.minor = ggplot2::element_blank(),
    panel.border = ggplot2::element_rect(
      fill = NA, colour = "#4D4D4D", linewidth = 0.25
    ),
    panel.background = ggplot2::element_rect(fill = "#EBEBEB", colour = NA),
    strip.background = ggplot2::element_rect(
      fill = "#D9D9D9", colour = "#4D4D4D", linewidth = 0.25
    ),
    strip.text = ggplot2::element_text(
      size = 9, face = "bold", margin = ggplot2::margin(3, 2, 3, 2)
    ),
    axis.title = ggplot2::element_text(size = 9),
    axis.text = ggplot2::element_text(size = 8, colour = "#333333"),
    legend.title = ggplot2::element_text(size = 9),
    legend.text = ggplot2::element_text(size = 8),
    plot.margin = ggplot2::margin(3, 3, 3, 3),
    legend.position = "right"
  )

method_labels_main <- c(
  "Truth" = "Truth", "DRT (AdaBoost)" = "DRT\n(AdaBoost)",
  "KLIEP" = "KLIEP", "uLSIF" = "uLSIF", "GB" = "GB",
  "Bayesian additive trees" = "Bayesian\nadditive trees",
  "Lower (2.5%)" = "Lower (2.5%)", "Upper (97.5%)" = "Upper (97.5%)"
)
method_labels_supplement <- method_labels_main
method_labels_supplement[c("Lower (2.5%)", "Upper (97.5%)")] <-
  c("Lower\n(2.5%)", "Upper\n(97.5%)")

render_bundle <- function(bundle) {
  expanded <- expand_limits_to_aspect(bundle$xlim, bundle$ylim, target_aspect)
  stopifnot(
    expanded$x[[1L]] <= bundle$xlim[[1L]],
    expanded$x[[2L]] >= bundle$xlim[[2L]],
    expanded$y[[1L]] <= bundle$ylim[[1L]],
    expanded$y[[2L]] >= bundle$ylim[[2L]],
    abs(diff(expanded$x) / diff(expanded$y) - target_aspect) < 1e-12
  )
  labels <- if (bundle$main) method_labels_main else method_labels_supplement

  figure <- ggplot2::ggplot(
    bundle$data,
    ggplot2::aes(
      x = .data[[bundle$axes[[1L]]]],
      y = .data[[bundle$axes[[2L]]]],
      colour = log_weight
    )
  ) +
    ggplot2::geom_point(size = 0.7, alpha = 0.75, stroke = 0) +
    ggplot2::scale_colour_gradientn(
      colours = bundle$colours,
      values = bundle$positions,
      limits = bundle$limits,
      oob = scales::squish,
      name = "log(density\nratio)",
      guide = ggplot2::guide_colourbar(
        barwidth = grid::unit(0.35, "cm"),
        barheight = grid::unit(2.6, "cm"),
        title.position = "top",
        title.hjust = 0.5
      )
    ) +
    ggplot2::facet_wrap(
      ~method, nrow = 2L, ncol = 4L,
      labeller = ggplot2::as_labeller(labels)
    ) +
    ggplot2::coord_equal(
      xlim = expanded$x, ylim = expanded$y, expand = FALSE, clip = "on"
    ) +
    ggplot2::labs(x = bundle$xlab, y = bundle$ylab) +
    base_theme

  figure_grob <- ggplot2::ggplotGrob(figure)
  panel_cells <- grepl("^panel", figure_grob$layout$name)
  panel_columns <- unique(figure_grob$layout$l[panel_cells])
  panel_rows <- unique(figure_grob$layout$t[panel_cells])
  stopifnot(length(panel_columns) == 4L, length(panel_rows) == 2L)
  figure_grob$widths[panel_columns] <- grid::unit(target_panel_width_cm, "cm")
  figure_grob$heights[panel_rows] <- grid::unit(target_panel_height_cm, "cm")
  stopifnot(
    all(abs(grid::convertWidth(
      figure_grob$widths[panel_columns], "cm", valueOnly = TRUE
    ) - target_panel_width_cm) < 1e-12),
    all(abs(grid::convertHeight(
      figure_grob$heights[panel_rows], "cm", valueOnly = TRUE
    ) - target_panel_height_cm) < 1e-12)
  )

  pdf_path <- file.path(outdir, paste0(bundle$id, ".pdf"))
  png_path <- file.path(outdir, paste0(bundle$id, ".png"))
  validation_path <- file.path(outdir, paste0(bundle$id, "_validation.txt"))
  ggplot2::ggsave(
    png_path, figure_grob, width = output_width_cm, height = output_height_cm,
    units = "cm",
    dpi = 300, bg = "white"
  )
  ggplot2::ggsave(
    pdf_path, figure_grob, width = output_width_cm, height = output_height_cm,
    units = "cm",
    device = grDevices::cairo_pdf, bg = "white"
  )

  source_hash <- tolower(unname(tools::sha256sum(bundle$source_path)))
  writeLines(c(
    paste0("figure=", bundle$id),
    paste0("width_cm=", output_width_cm),
    paste0("height_cm=", output_height_cm),
    "ticks_and_colorbar_pt=8",
    "headings_and_axis_titles_pt=9",
    "point_size=0.7",
    "point_alpha=0.75",
    "estimator_fitting=false",
    "coord_equal=true",
    sprintf("target_panel_aspect=%.9f", target_aspect),
    sprintf("target_panel_width_cm=%.6f", target_panel_width_cm),
    sprintf("target_panel_height_cm=%.6f", target_panel_height_cm),
    paste0("original_xlim=", paste(bundle$xlim, collapse = ",")),
    paste0("original_ylim=", paste(bundle$ylim, collapse = ",")),
    paste0("expanded_xlim=", paste(expanded$x, collapse = ",")),
    paste0("expanded_ylim=", paste(expanded$y, collapse = ",")),
    paste0("source_plotted_data_sha256=", source_hash)
  ), validation_path)

  list(
    id = bundle$id,
    png = png_path,
    pdf = pdf_path,
    original_xlim = bundle$xlim,
    original_ylim = bundle$ylim,
    expanded_xlim = expanded$x,
    expanded_ylim = expanded$y
  )
}

results <- lapply(bundles, render_bundle)
stopifnot(length(list.files(outdir, pattern = "[.]pdf$")) == length(bundles))
stopifnot(length(list.files(outdir, pattern = "[.]png$")) == length(bundles))
cat(sprintf("Saved %d equal-size eight-panel figures to %s\n", length(results), outdir))
