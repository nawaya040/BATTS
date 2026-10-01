library(BATTS)
expect_error <- function(expr, pattern) {
  message <- tryCatch({force(expr); NA_character_}, error=conditionMessage)
  stopifnot(!is.na(message), grepl(pattern,message,fixed=TRUE))
}
set.seed(414)
x <- cbind(c(rnorm(12),rnorm(12,0.5)),rnorm(24))
g <- rep(0:1,each=12)
for (bad in list(0,-1,1.5,NA_real_,NaN,Inf,numeric(),c(1,2),TRUE,'1',2^31,matrix(1))) {
  rng <- .Random.seed
  expect_error(batts(x,g,size_backfitting=3,thin=bad), 'thin must be a positive integer')
  stopifnot(identical(rng,.Random.seed))
}
stopifnot(!any(c('alpha_cutpoint','update_lambda','lambda_prior_parameters') %in% names(formals(batts))),
          !'alpha_cutpoint' %in% names(formals(boots)))
expect_error(batts(x,g,size_backfitting=3,update_lambda=FALSE),'unused argument')
expect_error(boots(x,g,alpha_cutpoint=1),'unused argument')
for (thin in c(1L,2L)) {
  set.seed(415)
  fit <- batts(x,g,num_trees=3,size_burnin=2,size_backfitting=4,thin=thin,
               output_BART_ensembles=TRUE,quiet=TRUE)
  stopifnot(ncol(fit$balance_weight_BART_data)==4L,
            all(is.finite(fit$balance_weight_BART_data)),
            all(fit$lambda_store==15))
  for (bayes in c(FALSE,TRUE)) {
    for (bad in list(x[,1,drop=FALSE],cbind(x,x[,1]),x[0,,drop=FALSE],
                     x[,1],data.frame(x),matrix(NA_real_,2,2),matrix(Inf,2,2),matrix('a',2,2))) {
      rng <- .Random.seed
      expect_error(eval_balance_weight(fit,bad,is_Bayes=bayes),'eval_points must be')
      stopifnot(identical(rng,.Random.seed))
    }
    prediction <- eval_balance_weight(fit,x,is_Bayes=bayes)
    if (bayes) stopifnot(isTRUE(all.equal(prediction$balancing_weight_BART,fit$balance_weight_BART_data)))
  }
}
# The internal native entry point also rejects zero thinning before fitting.
expect_error(BATTS:::run_adaboost(x,g,3L,0L,0.01,c(.25,.5,.75),rep(1L,24),
  1,1e-100,FALSE,2L,4L,0L,c(1/3,1/3,1/3),5,1,1,.95,2,FALSE,TRUE),
  'thin must be a positive integer')
cat('U14/U17 input checks and removed API tests passed.\n')
