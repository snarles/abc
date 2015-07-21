#' Initialize DNN
#' 
#' Create weight matrices for deep neural network
#' @param dims Size of each layer
#' @param sigma0 Standard deviation of initial weights
#' @param rseed Random seed
#' @return A list containing weight matrices
#' @export
dnn_init_slow <- function(dims, sigma0 = 0.1, rseed = 315) {
  set.seed(rseed)
  Ws <- as.list(numeric(length(dims) - 1))
  for (i in 1:length(Ws)) {
    # first row of weights are bias term
    Ws[[i]] <- rbind(0, sigma0 * randn(dims[i], dims[i + 1]))
  }
  Ws
}

#' Predict given DNN and tanh transfer function
#' @param Ws Weights of DNN
#' @param x New points, dimension n x p
#' @return An n x ? matrix of predictions
#' @export
dnn_predict_slow <- function(Ws, x) {
  y <- x
  for (i in 1:(length(Ws) - 1)) {
    y <- tanh(t(t(y %*% Ws[[i]][-1, , drop = FALSE]) + Ws[[i]][1, ]))
  }
  y <- t(t(y %*% Ws[[length(Ws)]][-1, , drop = FALSE]) + Ws[[length(Ws)]][1, ])
  y
}

#' Compute gradient for DNN
#' @param Ws Weights of DNN
#' @param x Training points, dimension n x p_x
#' @param y Training targets, dimension n x p_y
#' @return A list of gradients
dnn_grad_slow <- function(Ws, x, y) {
  Hs <- as.list(numeric(length(Ws)))
  for (i in 1:length(Ws)) {
    if (i == 1) {  
      Hs[[i]] <- tanh(t(t(x %*% Ws[[i]][-1, , drop = FALSE]) + Ws[[i]][1, ]))
    } else {
      Hs[[i]] <- tanh(t(t(Hs[[i - 1]] %*% Ws[[i]][-1, , drop = FALSE]) + Ws[[i]][1, ]))
    }
  }
  delta <- t(t(Hs[[length(hs)]] - y))
  dWs <- as.list(numeric(length(Ws)))
  for (i in length(Ws):2) {
    dW <- t(Hs[[i - 1]]) %*% delta
    db <- colSums(delta)
    dWs[[i]] <- rbind(db, dW)
    delta <- (delta %*% t(Ws[[i]][-1, , drop = FALSE])) * (1 - Hs[[i - 1]]^2)
  }
  dW <- t(x) %*% delta
  db <- colSums(delta)
  dWs[[1]] <- rbind(db, dW)
  dWs
}

#' Computes the loss
#' @param Ws Weights of DNN
#' @param x Training points, dimension n x p_x
#' @param y Training targets, dimension n x p_y
#' @return Objective function, sqrt(sum(y - hat(y))^2/n)
dnn_loss_slow <- function(Ws, x, y) {
  yh <- dnn_predict_slow(Ws, x)
  sqrt(sum((y - yh)^2)/dim(x)[1])
}

dnn_sgd_slow <- function(Ws, x_tr, y_tr, x_te, y_te, epoch = 100,
                         reg = 0.0, minibatch = 100, alpha0 = 0.1) {
  num_batch_per_epoch <- floor(dim(x_tr)[1]/minibatch)
  train_costs <- numeric(epoch)
  test_costs <- numeric(epoch)
  alphas <- alpha0 * 1/(1:epoch)
  for (i in 1:epoch) {
    for (k in 1:num_batch_per_epoch) {
      x <- x_tr[(k - 1) * minibatch + (1:minibatch), , drop = FALSE]
      y <- y_tr[(k - 1) * minibatch + (1:minibatch), , drop = FALSE]
      gr <- dnn_grad_slow(Ws, x, y)
      for (ii in 1:length(Ws)) {
        W <- Ws[[ii]][-1, , drop = FALSE]
        dW <- -alphas[i] * (gr[[ii]][-1, , drop = FALSE] + reg * W)/minibatch
        db <- -alphas[i] * (gr[[ii]][1, ])/minibatch
        Ws[[ii]] <- Ws[[ii]] + rbind(db, dW)
      }
    }
    (train_costs[i] <- dnn_loss_slow(Ws, x_tr, y_tr))
    (test_costs[i] <- dnn_loss_slow(Ws, x_te, y_te))
  }
  list(Ws = Ws, train_costs = train_costs, test_costs = test_costs)
}

