# Compare one source package at a time in separate R processes.
# Usage: Rscript ... baseline|anonymous library-dir NEW-output.rds
args <- commandArgs(TRUE)
stopifnot(length(args)==3L,args[[1L]] %in% c("baseline","anonymous"),!file.exists(args[[3L]]))
lib <- normalizePath(args[[2L]],winslash="/",mustWork=TRUE)
.libPaths(c(lib,.libPaths()))
pkg <- if(args[[1L]]=="baseline") "BATTS" else "ReviewPkg"
stopifnot(identical(normalizePath(find.package(pkg,lib.loc=lib),winslash="/"),paste0(lib,"/",pkg)))
loadNamespace(pkg,lib.loc=lib)
boost <- getExportedValue(pkg,if(pkg=="BATTS") "boots" else "fit_boosting_model")
bayes <- getExportedValue(pkg,if(pkg=="BATTS") "batts" else "fit_bayesian_model")
evaluate <- getExportedValue(pkg,if(pkg=="BATTS") "eval_balance_weight" else "evaluate_density_ratio")
RNGkind("Mersenne-Twister","Inversion","Rejection")
results <- list()
for(d in c(2L,20L)) for(seed in c(13L,29L)) {
  set.seed(seed+1000L)
  x <- matrix(rnorm(120L*d),ncol=d);x[1:80,1] <- x[1:80,1]+.5
  labels <- c(rep(0L,80L),rep(1L,40L))
  newx <- .8*x[1:15,,drop=FALSE] + .2*matrix(colMeans(x),nrow=15,ncol=d,byrow=TRUE)
  for(gradient in c(FALSE,TRUE)) {
    set.seed(seed)
    fit <- boost(x,labels,num_trees_max=12,K_CV=3,max_resol=3,n_bins=8,use_gradient=gradient,quiet=TRUE)
    rng_fit <- .Random.seed
    pred <- evaluate(fit,newx,is_Bayes=FALSE)
    results[[paste(d,seed,gradient,sep="_")]] <- list(fit=fit,prediction=pred,rng_after_fit=rng_fit,rng_after_prediction=.Random.seed)
  }
  set.seed(seed)
  fit <- bayes(x,labels,num_trees=5,size_burnin=6,size_backfitting=10,n_bins=8,output_BART_ensembles=TRUE,quiet=TRUE)
  rng_fit <- .Random.seed
  pred <- evaluate(fit,newx,is_Bayes=TRUE)
  results[[paste(d,seed,"bayes",sep="_")]] <- list(fit=fit,prediction=pred,rng_after_fit=rng_fit,rng_after_prediction=.Random.seed)
}
saveRDS(results,args[[3L]])
cat("Completed",length(results),"fit/prediction/RNG comparisons for",pkg,"\n")
