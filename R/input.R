# Input handling for batts_boost(), batts_bart() and the predict() methods.
# A double matrix with 0/1 group labels passes through unchanged, so those
# fits stay identical to boots() and batts().

.as_data_matrix = function(x, name){
  if(is.data.frame(x)){
    bad = names(x)[!vapply(x, is.numeric, logical(1))]
    if(length(bad) > 0L){
      stop(sprintf("%s has non-numeric columns: %s", name,
                   paste(bad, collapse = ", ")), call. = FALSE)
    }
    x = as.matrix(x)
  }else if(is.numeric(x) && is.null(dim(x))){
    x = matrix(x, ncol = 1L)
  }
  if(!is.matrix(x) || !is.numeric(x)){
    stop(sprintf("%s must be a numeric matrix, data frame or vector", name),
         call. = FALSE)
  }
  if(nrow(x) < 1L || ncol(x) < 1L){
    stop(sprintf("%s has no rows or no columns", name), call. = FALSE)
  }
  if(any(!is.finite(x))){
    stop(sprintf("%s must not contain missing or infinite values", name),
         call. = FALSE)
  }
  if(storage.mode(x) != "double") storage.mode(x) = "double"
  x
}

# Map two-valued labels to 0/1. The first level (factor levels, otherwise the
# smaller value in sort order) becomes group 0, the numerator p.
.as_group = function(group, n){
  if(!is.null(dim(group)) || length(group) != n){
    stop("group must have one label per row of x", call. = FALSE)
  }
  if(anyNA(group)) stop("group must not contain missing values", call. = FALSE)
  if(is.numeric(group) && all(group %in% c(0, 1))){
    labels = group
    levels = c("0", "1")
  }else if(is.factor(group)){
    group = droplevels(group)
    levels = levels(group)
    labels = as.integer(group) - 1L
  }else if(is.numeric(group) || is.logical(group) || is.character(group)){
    values = sort(unique(group), method = "radix")
    levels = as.character(values)
    labels = match(group, values) - 1L
  }else{
    stop("group must be a numeric, logical, character or factor vector",
         call. = FALSE)
  }
  if(length(levels) != 2L){
    stop(sprintf("group must have exactly two distinct values, not %d",
                 length(levels)), call. = FALSE)
  }
  list(labels = labels, levels = levels)
}

# Build the training matrix and 0/1 labels from either (x, group) or (x, y).
.fit_input = function(x, group, y){
  if(is.null(group) == is.null(y)){
    stop("supply exactly one of group (labels for the rows of x) and y (a second sample)",
         call. = FALSE)
  }
  x = .as_data_matrix(x, "x")
  if(is.null(y)){
    g = .as_group(group, nrow(x))
  }else{
    y = .as_data_matrix(y, "y")
    if(ncol(x) != ncol(y)){
      stop("x and y must have the same number of columns", call. = FALSE)
    }
    if(!is.null(colnames(x)) && !is.null(colnames(y)) &&
       !identical(colnames(x), colnames(y))){
      stop("x and y must have the same column names in the same order",
           call. = FALSE)
    }
    g = list(labels = rep(0:1, c(nrow(x), nrow(y))), levels = c("x", "y"))
    x = rbind(x, y)
  }
  list(x = x, group = g$labels, levels = g$levels,
       variable_names = colnames(x))
}

.check_constant_columns = function(x, margin_scale){
  if(margin_scale < 0) return(invisible(NULL))
  constant = which(apply(x, 2L, function(v) max(v) == min(v)))
  if(length(constant) > 0L){
    nm = if(is.null(colnames(x))) as.character(constant)
         else sprintf("'%s'", colnames(x)[constant])
    stop(sprintf("constant column%s %s; remove constant columns before fitting",
                 if(length(constant) > 1L) "s" else "",
                 paste(nm, collapse = ", ")), call. = FALSE)
  }
  invisible(NULL)
}

# Turn newdata into a numeric matrix whose columns follow the training matrix.
.as_newdata = function(newdata, object){
  d = object$data_info$d
  vn = object$variable_names
  if(is.numeric(newdata) && is.null(dim(newdata))){
    if(d == 1L){
      newdata = matrix(newdata, ncol = 1L)
    }else if(length(newdata) == d){
      newdata = matrix(newdata, nrow = 1L)
    }else{
      stop(sprintf("a vector newdata must be one point (%d values) for this fit with %d variables; use a matrix or data frame for several points",
                   d, d), call. = FALSE)
    }
  }
  nm = if(is.data.frame(newdata)) names(newdata) else colnames(newdata)
  if(!is.null(vn) && !is.null(nm)){
    missing_vars = setdiff(vn, nm)
    if(length(missing_vars) > 0L){
      stop(sprintf("newdata lacks the variable%s %s",
                   if(length(missing_vars) > 1L) "s" else "",
                   paste(missing_vars, collapse = ", ")), call. = FALSE)
    }
    if(!identical(nm, vn)){
      newdata = if(is.data.frame(newdata)) newdata[vn] else newdata[, vn, drop = FALSE]
    }
  }
  newdata = .as_data_matrix(newdata, "newdata")
  if(ncol(newdata) != d){
    stop(sprintf("newdata must have %d column%s, as the training data had",
                 d, if(d > 1L) "s" else ""), call. = FALSE)
  }
  newdata
}

# Apply the outside rule. Returns the points to evaluate and, for "na", the
# rows that lie inside the fitted domain.
.handle_outside = function(newdata, object, outside){
  domain = object$data_info$min_max_values
  lo = matrix(domain[, 1], nrow(newdata), ncol(newdata), byrow = TRUE)
  hi = matrix(domain[, 2], nrow(newdata), ncol(newdata), byrow = TRUE)
  out = newdata < lo | newdata > hi
  if(!any(out)) return(list(newdata = newdata, keep = NULL))
  switch(outside,
    error = {
      where = which(out, arr.ind = TRUE)
      where = where[order(where[, 1], where[, 2]), , drop = FALSE]
      vn = object$variable_names
      shown = seq_len(min(nrow(where), 3L))
      lines = vapply(shown, function(k){
        i = where[k, 1]; j = where[k, 2]
        sprintf("  row %d, %s: %s is outside [%s, %s]", i,
                if(is.null(vn)) sprintf("column %d", j) else sprintf("variable '%s'", vn[j]),
                format(newdata[i, j], digits = 6),
                format(domain[j, 1], digits = 6), format(domain[j, 2], digits = 6))
      }, character(1))
      n_rows = length(unique(where[, 1]))
      stop(sprintf("%d of %d points lie outside the fitted domain:\n%s%s\nUse outside = \"na\" to return NA for them or outside = \"clamp\" to move them to the boundary.",
                   n_rows, nrow(newdata), paste(lines, collapse = "\n"),
                   if(nrow(where) > 3L) "\n  ..." else ""), call. = FALSE)
    },
    clamp = list(newdata = pmin(pmax(newdata, lo), hi), keep = NULL),
    na = list(newdata = newdata, keep = rowSums(out) == 0))
}

# Balancing weights at new points: a vector for boosting, a points-by-draws
# matrix for BART. Rows outside the domain are NA when outside = "na".
.eval_new = function(object, newdata, outside, bayes){
  newdata = .handle_outside(.as_newdata(newdata, object), object, outside)
  evaluate = function(points){
    res = .eval_weights(object, points, is_Bayes = bayes)
    if(bayes) res$balancing_weight_BART
    else as.numeric(res$balancing_weight_boosting)
  }
  keep = newdata$keep
  if(is.null(keep)) return(evaluate(newdata$newdata))
  n = nrow(newdata$newdata)
  if(bayes){
    w = matrix(NA_real_, n, ncol(object$balance_weight_BART_data))
    if(any(keep)) w[keep, ] = evaluate(newdata$newdata[keep, , drop = FALSE])
  }else{
    w = rep(NA_real_, n)
    if(any(keep)) w[keep] = evaluate(newdata$newdata[keep, , drop = FALSE])
  }
  w
}

# Lines describing which group is the numerator, for the print methods.
.orientation_lines = function(levels){
  if(is.null(levels)){
    return(c("  estimand:     balancing weight w = sqrt(p/q), group 0 in the numerator",
             "                2 * log(w) is the log-density ratio"))
  }
  c("  estimand:     balancing weight w = sqrt(p/q)",
    sprintf("                p: group \"%s\" (numerator), q: group \"%s\"",
            levels[1], levels[2]),
    "                2 * log(w) is the log-density ratio")
}

.count_line = function(n0, n1, levels){
  lab = if(is.null(levels)) c("group 0", "group 1")
        else sprintf("group \"%s\"", levels)
  sprintf("  observations: %d (%s: %d, %s: %d)", n0 + n1, lab[1], n0, lab[2], n1)
}
