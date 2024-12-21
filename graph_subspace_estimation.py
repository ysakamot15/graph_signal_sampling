import numpy as np
from sklearn.decomposition import PCA
import graph_signal_utils as utils


def subspace_estimate_with_pca(X, k):
    pca = PCA(k, svd_solver='full')
    pca.fit(X)
    return pca.components_.T

def subspace_estimate_with_gft(X, adj, k):
    gft = utils.GraphFourierTransformer(adj)
    Xf = gft.transform(X)
    s = np.sqrt(np.sum((Xf * Xf), axis=0))
    remain = np.argsort(-s)[:k]
    return gft.U[:, remain]

def proximal_gradient_method(grad, prox, eta, x_init, max_iter=10000, tol=1e-5):
    x = np.copy(x_init)
    for _ in range(max_iter):
        z = x - eta * grad(x)
        x_new = prox(z)
        print(np.linalg.norm(x - x_new))
        if np.linalg.norm(x - x_new) <= tol:
            break
        x = np.copy(x_new)
    return x_new

def trace_prox(Y, lam):
    U, S, V = np.linalg.svd(Y)
    return U @ np.diag(np.maximum(0.0, S - lam)) @ V[:S.shape[0], :]

def subspace_estimate_with_regularization(Y, T, alpha, beta):
    Hess = 2.0 * (np.eye(T.shape[0]) + alpha * T)
    grad = lambda W: -2.0 * Y + W @ Hess
    eta = 1.0/np.max(np.linalg.eigvalsh(Hess))
    prox = lambda Y: trace_prox(Y, beta * eta)
    W_init = np.copy(Y)
    W_hat = proximal_gradient_method(grad, prox, eta, W_init)
    return W_hat

def subspace_estimate_with_vertex_regularization(X, adj, k, alpha, beta):
    L = utils.compute_graph_laplacian(adj)
    W = subspace_estimate_with_regularization(X, L, alpha, beta)
    Uk, _, _ = np.linalg.svd(W.T)
    return Uk[:, :k], W

def subspace_estimate_with_frequency_regularization(X, adj, k, alpha, beta):
    gft = utils.GraphFourierTransformer(adj)
    X_ = gft.transform(X)
    P = np.eye(X.shape[1]) - (np.eye(X.shape[1], k=-k) +
                              np.eye(X.shape[1], k= X.shape[1] - k))
    W = subspace_estimate_with_regularization(X_, P @ P.T, alpha, beta)
    Uk, _, _ = np.linalg.svd(W.T)
    return  gft.U @ Uk[:, :k], W
