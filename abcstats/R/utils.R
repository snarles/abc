#' Loads all elements of a list into the global environment
#' 
#' @param ll List with named elements
#' @return No return: side effect is to load elements into environment, 
#' similar to attach() but into environment rather than search path
zattach <- function(ll) {
  for (i in 1:length(ll)) {
    assign(names(ll)[i], ll[[i]], envir=globalenv())
  }
}
