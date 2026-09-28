# Read-only audit of RDS/RData contents, including character attributes.
args <- commandArgs(TRUE)
stopifnot(length(args) == 2L, !file.exists(args[[2L]]))
root <- normalizePath(args[[1L]], winslash = "/", mustWork = TRUE)
pattern <- "BATTS|nawaya|Awaya|2508[.]03059|[A-Za-z]:[/\\\\]Users[/\\\\]|/Users/|/home/|(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])"
findings <- list()
walk <- function(x, file, field) {
  if (is.environment(x) || is.function(x) || is.language(x) || isS4(x)) {
    findings[[length(findings) + 1L]] <<- data.frame(file, field, value = paste("REQUIRES_INSPECTION", typeof(x)))
    return(invisible(NULL))
  }
  if (is.character(x)) {
    hit <- which(!is.na(x) & grepl(pattern, x, ignore.case = TRUE, perl = TRUE))
    for (i in hit) findings[[length(findings) + 1L]] <<- data.frame(file, field = paste0(field,"[",i,"]"), value = x[[i]])
  }
  if (is.list(x)) for (i in seq_along(x)) walk(x[[i]], file, paste0(field, "$", if(is.null(names(x))) i else names(x)[[i]]))
  a <- attributes(x)
  if (length(a)) for (n in names(a)) walk(a[[n]], file, paste0(field, "@", n))
}
files <- list.files(root, pattern = "[.](rds|rda|rdata)$", full.names = TRUE, recursive = TRUE, ignore.case = TRUE)
for (file in files) {
  rel <- substring(file, nchar(root) + 2L)
  if (grepl("[.]rds$", file, ignore.case = TRUE)) walk(readRDS(file), rel, "root") else {
    env <- new.env(parent = emptyenv()); load(file, envir = env)
    for (n in ls(env, all.names = TRUE)) walk(env[[n]], rel, n)
  }
}
result <- if(length(findings)) do.call(rbind,findings) else data.frame(file=character(),field=character(),value=character())
write.csv(result, args[[2L]], row.names = FALSE)
cat("Audited",length(files),"serialized files;",nrow(result),"candidate fields\n")
