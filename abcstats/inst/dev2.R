## Development of Rcpp functions

library(Rcpp)
library(devtools)
library(roxygen2)

aa <- "abcstats"

load_all(aa)


x <- 2 * matrix(rbinom(100, 1, 0.5), 10, 10) - 1
image(x)
image(ising_chain(x, 0.3, 200))
