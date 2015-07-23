#' Elementwise list sum
#' @param l1 List
#' @param l2 List
list_sum <- function(l1, l2) {
  for (i in 1:length(l1)) l1[[i]] <- l1[[i]] + l2[[i]]
  l1
}

#' Elementwise list concatenation
#' @param l1 List
#' @param l2 List
list_rbind <- function(l1, l2) {
  for (i in 1:length(l1)) l1[[i]] <- rbind(l1[[i]], l2[[i]])
  l1
}


#' Optimizes the network using sgd
#' @param Wbs Weights and biases of DNN
#' @param x_tr Training points, dimension n x p_x
#' @param y_tr Training targets, dimension n x p_y
#' @param x_te Test points, dimension n x p_x
#' @param y_te Test targets, dimension n x p_y
#' @param epoch Number of epochs
#' @param reg Weight regularization
#' @param minibatch Minibatch size
#' @param mc.cores MC cores for parallel
#' @param alpha0 Initial step size
#' @return Results
#' @export
dnn_sgd_par <- function(Wbs, x_tr, y_tr, x_te, y_te, epoch = 100,
  reg = 0.0, minibatch = 25, alpha0 = 0.1, mc.cores = 4) {
  depth <- length(Wbs)/2
  num_batch_per_epoch <- floor(dim(x_tr)[1]/minibatch)
  splits <- as.list(numeric(num_batch_per_epoch))
  for (k in 1:num_batch_per_epoch) {
    x <- x_tr[(k - 1) * minibatch + (1:minibatch), , drop = FALSE]
    y <- y_tr[(k - 1) * minibatch + (1:minibatch), , drop = FALSE]
    splits[[k]] <- list(x = x, y = y)
  }
  grad_split <- function(k) {
    x <- splits[[k]]$x
    y <- splits[[k]]$y
    gr <- dnn_grad(Wbs, x, y)
    gr
  }
  num_par_per_epoch <- num_batch_per_epoch/mc.cores
  n_par <- dim(x_tr)[1]/num_par_per_epoch
  train_costs <- numeric(epoch)
  test_costs <- numeric(epoch)
  alphas <- alpha0 * 1/(1:epoch)
  for (i in 1:epoch) {
    for (ii in 1:num_par_per_epoch) {
      inds <- (ii-1) * mc.cores + 1:mc.cores
      res <- parallel::mclapply(inds, grad_split, mc.cores = mc.cores)
      gr <- Reduce(list_sum, res)
      for (j in 1:depth) {
        Wbs[[j]] <- Wbs[[j]] - alphas[i]*(gr[[j]] + reg * Wbs[[j]])/n_par
        Wbs[[j + depth]] <- Wbs[[j + depth]] - alphas[i]*gr[[j + depth]]/n_par
      }
    }
    p_tr <- dnn_predict(Wbs, x_tr)
    train_costs[i] <- sqrt(f2(p_tr - y_tr)/dim(y_tr)[1])
    p_te <- dnn_predict(Wbs, x_te)
    test_costs[i] <- sqrt(f2(p_te - y_te)/dim(y_te)[1])
  }
  list(Wbs = Wbs, train_costs = train_costs, test_costs = test_costs, p_tr = p_tr, p_te = p_te)
}
 