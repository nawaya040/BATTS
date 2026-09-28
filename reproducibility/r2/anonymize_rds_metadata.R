# Adapt only seven historical result metadata objects in a NEW candidate.
args <- commandArgs(TRUE)
stopifnot(length(args) == 3L)
source_root <- normalizePath(args[[1L]], winslash="/", mustWork=TRUE)
target_root <- normalizePath(args[[2L]], winslash="/", mustWork=TRUE)
stopifnot(source_root != target_root, !file.exists(args[[3L]]))
files <- list.files(file.path(source_root,"results/r2/figure_details/boosting"), pattern="[.]rds$",full.names=TRUE)
stopifnot(length(files)==7L)
records <- list()
for(f in files) {
  original <- readRDS(f); adapted <- original
  stopifnot(is.list(original$metadata))
  adapted$metadata$commit_hash <- NULL
  adapted$metadata$source_hashes <- NULL
  adapted$metadata$r_library <- "ANONYMOUS_ORIGINAL_LIBRARY"
  adapted$metadata$batts_install_hashes <- NULL
  names(adapted$metadata$package_versions) <- sub("BATTS", "original_method", names(adapted$metadata$package_versions), fixed=TRUE)
  adapted$metadata$source_record <- "PRIOR_BOOSTING_RUN"
  adapted$metadata$identity_note <- "Original identity fields are retained privately; scientific fields are unchanged."
  stopifnot(identical(original[names(original)!="metadata"], adapted[names(adapted)!="metadata"]))
  shared <- setdiff(intersect(names(original$metadata),names(adapted$metadata)),c("r_library","package_versions"))
  stopifnot(identical(original$metadata[shared],adapted$metadata[shared]),
            identical(unname(original$metadata$package_versions),unname(adapted$metadata$package_versions)))
  rel <- substring(f,nchar(source_root)+2L); output <- file.path(target_root,rel)
  stopifnot(file.exists(output)); saveRDS(adapted,output)
  stopifnot(identical(readRDS(output),adapted))
  records[[length(records)+1L]] <- data.frame(file=rel, scientific_content_identical=TRUE)
}
# Check every remaining serialized input byte for byte.
all <- list.files(source_root,pattern="[.]rds$",recursive=TRUE,full.names=TRUE)
for(f in setdiff(all,files)) {
  target <- file.path(target_root,substring(f,nchar(source_root)+2L))
  stopifnot(identical(readBin(f,"raw",n=file.info(f)$size),readBin(target,"raw",n=file.info(target)$size)))
}
write.csv(do.call(rbind,records),args[[3L]],row.names=FALSE)
cat("PASS:",length(files),"metadata-only adaptations and",length(all)-length(files),"byte-identical RDS copies\n")
