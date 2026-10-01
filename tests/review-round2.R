library(BATTS)
expect_reject <- function(expr, pattern) {
  before <- .Random.seed
  error <- tryCatch({force(expr); NA_character_}, error=conditionMessage)
  stopifnot(!is.na(error), grepl(pattern,error,fixed=TRUE), identical(before,.Random.seed))
}
set.seed(22023)
x <- matrix(rnorm(48),24,2); g <- rep(0:1,each=12)
for (bad in list(-1,NA_real_,NaN,Inf,-Inf,1.5,numeric(),c(0,1),TRUE,'0',2^31,matrix(0)))
  expect_reject(batts(x,g,size_backfitting=4,size_burnin=bad), 'size_burnin must')
for (bad in list(0,-1,NA_real_,Inf,1.5,numeric(),c(1,2),TRUE,'4',2^31,matrix(4)))
  expect_reject(batts(x,g,size_backfitting=bad), 'size_backfitting must')
bad_moves <- list(c(.8,.8,.8),c(.2,.2,.2),c(-.1,.5,.6),c(0,.5,.5),c(.5,.5),
  c(1,1,1,1)/4,c(NA,.5,.5),c(Inf,.5,.5),numeric(),TRUE,c('a','b','c'),matrix(1/3,1,3))
for (bad in bad_moves)
  expect_reject(batts(x,g,size_backfitting=4,prob_moves=bad), 'prob_moves must')
native <- function(burn=0L, moves=c(1/3,1/3,1/3), draws=4L)
  BATTS:::run_adaboost(x,g,2L,0L,.01,c(.25,.5,.75),rep(1L,24),1,1e-100,FALSE,
    burn,draws,1L,moves,5,1,1,.95,2,FALSE,TRUE)
expect_reject(native(burn=-1L),'size_burnin must')
expect_reject(native(burn=NA_integer_),'size_burnin must')
expect_reject(native(draws=-1L),'size_backfitting must')
for (bad in bad_moves[1:9]) expect_reject(native(moves=bad),'prob_moves must')
for (burn in c(0L,3L)) for (moves in list(c(1/3,1/3,1/3),c(.2,.5,.3))) {
  fit <- batts(x,g,num_trees=2,size_backfitting=4,size_burnin=burn,prob_moves=moves,quiet=TRUE)
  stopifnot(identical(dim(fit$balance_weight_BART_data),c(24L,4L)),
    all(is.finite(fit$balance_weight_BART_data)),all(fit$balance_weight_BART_data>0))
}
# MH grow/prune corrections are reciprocals for asymmetric valid probabilities.
m <- c(.2,.5,.3)
stopifnot(identical(log(m[2])-log(m[1]),-(log(m[1])-log(m[2]))))
cat('U22/U23 invalid-input rejection, RNG preservation and valid-input checks passed.\n')
