get_arg_value <- function(args, name, default = NULL) {
  prefix <- paste0("--", name, "=")
  matches <- args[startsWith(args, prefix)]
  if (!length(matches)) {
    return(default)
  }
  sub(prefix, "", matches[[1]], fixed = TRUE)
}

args <- commandArgs(trailingOnly = TRUE)
png_output <- get_arg_value(
  args,
  "png-output",
  file.path("output", "figures", "figure2_r2_preview.png")
)
pdf_output <- get_arg_value(
  args,
  "pdf-output",
  file.path("output", "pdf", "figure2_r2_preview.pdf")
)

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1L) {
  stop("Cannot identify the script path")
}
script_path <- normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE)
repo_root <- normalizePath(file.path(dirname(script_path), "..", ".."), mustWork = TRUE)
model_path <- file.path(
  repo_root,
  "scripts",
  "coverage",
  "models",
  "section41_2d_models.R"
)
if (!file.exists(model_path)) {
  stop("Missing two-dimensional model file: ", model_path)
}

required_packages <- c("ggplot2", "patchwork", "mvtnorm")
missing_packages <- required_packages[!vapply(
  required_packages,
  requireNamespace,
  logical(1),
  quietly = TRUE
)]
if (length(missing_packages)) {
  stop("Missing required packages: ", paste(missing_packages, collapse = ", "))
}

source(model_path, local = TRUE)

scenario_spec <- data.frame(
  scenario = c("global_shift", "local_shift", "local_dispersion"),
  title = c("(a) Global shift", "(b) Local shift", "(c) Local dispersion"),
  stringsAsFactors = FALSE
)
n0 <- 5000L
n1 <- 5000L
data_seed <- 1L
n_grid_per_dim <- 100L

RNGkind("Mersenne-Twister", "Inversion", "Rejection")
scenario_data <- vector("list", nrow(scenario_spec))
for (index_scenario in seq_len(nrow(scenario_spec))) {
  set.seed(data_seed)
  generated <- simulation_2d(
    n0,
    n1,
    scenario_spec$scenario[[index_scenario]],
    n_grid_per_dim
  )
  if (!is.matrix(generated$data) || !identical(dim(generated$data), c(n0 + n1, 2L))) {
    stop("Unexpected generated-data dimensions for ", scenario_spec$scenario[[index_scenario]])
  }
  if (length(generated$group_labels) != n0 + n1) {
    stop("Unexpected group-label length for ", scenario_spec$scenario[[index_scenario]])
  }
  scenario_data[[index_scenario]] <- data.frame(
    x1 = generated$data[, 1L],
    x2 = generated$data[, 2L],
    group = factor(
      generated$group_labels,
      levels = c(0L, 1L),
      labels = c("Sample 0", "Sample 1")
    ),
    scenario = scenario_spec$scenario[[index_scenario]],
    stringsAsFactors = FALSE
  )
}

plot_data <- do.call(rbind, scenario_data)
plot_data <- plot_data[sample.int(nrow(plot_data)), , drop = FALSE]
row.names(plot_data) <- NULL

if (nrow(plot_data) != 3L * (n0 + n1) || any(!is.finite(plot_data$x1)) ||
    any(!is.finite(plot_data$x2))) {
  stop("Generated Figure 2 data failed validation")
}
cell_counts <- table(plot_data$scenario, plot_data$group)
if (!all(cell_counts == 5000L)) {
  stop("Each scenario/group cell must contain 5000 observations")
}

palette <- c("Sample 0" = "#2C7FB8", "Sample 1" = "#D95F02")
make_panel <- function(index_scenario) {
  scenario_name <- scenario_spec$scenario[[index_scenario]]
  ggplot2::ggplot(
    plot_data[plot_data$scenario == scenario_name, , drop = FALSE],
    ggplot2::aes(x = x1, y = x2, color = group)
  ) +
    ggplot2::geom_point(size = 1.10, alpha = 0.36, stroke = 0) +
    ggplot2::scale_color_manual(values = palette, drop = FALSE) +
    ggplot2::coord_equal(expand = TRUE) +
    ggplot2::labs(
      title = scenario_spec$title[[index_scenario]],
      x = expression(x[1]),
      y = expression(x[2]),
      color = NULL
    ) +
    ggplot2::guides(
      color = ggplot2::guide_legend(
        override.aes = list(alpha = 1, size = 3.2)
      )
    ) +
    ggplot2::theme_minimal(base_family = "sans", base_size = 15) +
    ggplot2::theme(
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major = ggplot2::element_line(color = "#E6E6E6", linewidth = 0.35),
      panel.border = ggplot2::element_rect(
        color = "#333333",
        fill = NA,
        linewidth = 0.55
      ),
      plot.title = ggplot2::element_text(
        face = "bold",
        size = 16,
        hjust = 0.5,
        margin = ggplot2::margin(b = 7)
      ),
      axis.title = ggplot2::element_text(size = 15),
      axis.text = ggplot2::element_text(size = 12.5, color = "#222222"),
      axis.ticks = ggplot2::element_line(color = "#333333", linewidth = 0.4),
      legend.text = ggplot2::element_text(size = 13),
      legend.key.width = grid::unit(1.35, "cm"),
      legend.spacing.x = grid::unit(0.25, "cm"),
      plot.margin = ggplot2::margin(8, 8, 4, 8)
    )
}

panels <- lapply(seq_len(nrow(scenario_spec)), make_panel)
combined_plot <- patchwork::wrap_plots(panels, nrow = 1L, guides = "collect") &
  ggplot2::theme(legend.position = "bottom")

dir.create(dirname(png_output), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(pdf_output), recursive = TRUE, showWarnings = FALSE)
png_output <- normalizePath(png_output, mustWork = FALSE)
pdf_output <- normalizePath(pdf_output, mustWork = FALSE)

png_type <- if (capabilities("cairo")) "cairo" else getOption("bitmapType")
grDevices::png(
  png_output,
  width = 3600,
  height = 1350,
  res = 300,
  pointsize = 15,
  bg = "white",
  type = png_type
)
print(combined_plot)
grDevices::dev.off()

grDevices::cairo_pdf(
  pdf_output,
  width = 12,
  height = 4.5,
  family = "sans",
  pointsize = 15,
  bg = "white"
)
print(combined_plot)
grDevices::dev.off()

cat("Created Figure 2 preview\n")
cat("  PNG: ", png_output, "\n", sep = "")
cat("  PDF: ", pdf_output, "\n", sep = "")
cat("  Scenarios: ", paste(scenario_spec$scenario, collapse = ", "), "\n", sep = "")
cat("  Observations: ", nrow(plot_data), " (5000 per scenario/group cell)\n", sep = "")
cat("  RNG: Mersenne-Twister/Inversion/Rejection; data seed 1 per scenario\n")
