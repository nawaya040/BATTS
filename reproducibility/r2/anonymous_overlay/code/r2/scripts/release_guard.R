# Verify an extracted review release without a Git checkout.
review_verify_release <- function(project_root) {
  root <- normalizePath(file.path(project_root, "../.."), winslash = "/", mustWork = TRUE)
  id <- readLines(file.path(root, "RELEASE_ID.txt"), warn = FALSE)
  if(length(id) != 1L || !grepl("^[0-9a-f]{40}$", id)) stop("Invalid anonymous release ID")
  manifest <- read.csv(file.path(root, "SOURCE_MANIFEST.csv"), stringsAsFactors = FALSE)
  if(!all(c("release_path", "release_sha256", "release_bytes") %in% names(manifest)) ||
     anyDuplicated(manifest$release_path)) stop("Invalid release manifest")
  # Source and release identity are verified on every invocation; full input
  # integrity is separately checked with code/verify_bundle.py.
  selected <- manifest[startsWith(manifest$release_path, "code/") |
                         startsWith(manifest$release_path, "reference/r1/code/") |
                         manifest$release_path == "RELEASE_ID.txt", , drop = FALSE]
  stopifnot(nrow(selected) > 50L)
  for(i in seq_len(nrow(selected))) {
    rel <- selected$release_path[[i]]
    if(grepl("(^/|^[A-Za-z]:|(^|/)\\.\\.(/|$))", rel)) stop("Unsafe manifest path")
    path <- file.path(root, rel)
    if(!file.exists(path) || file.info(path)$size != selected$release_bytes[[i]] ||
       unname(tools::sha256sum(path)) != selected$release_sha256[[i]])
      stop("Release source mismatch: ", rel)
  }
  id
}
