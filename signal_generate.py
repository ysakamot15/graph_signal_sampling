import numpy as np
import graph_signal_utils as utils
from sklearn.cluster import KMeans


# グラフ周波数が低い方からK個までに帯域制限されたグラフ信号を生成
# グラフ信号と部分空間の基底を返す
def generate_band_limiation_signal(A, n, K=10):
    L = utils.compute_graph_laplacian(A)
    _, U = np.linalg.eigh(L)
    D = np.random.randn(n, K)
    A = U[:, :K]
    X = D @ A.T
    return X, A

def generate_piecewise_constant_signal(V, n, K=10):
    N = V.shape[0]
    kmeans = KMeans(n_clusters=K).fit(V)
    A = np.zeros((N, K))
    for k in range(K):
        A[kmeans.labels_ == k, k] = 1.0
    D = np.random.randn(n, K)
    X = D @ A.T
    return X, A

def generate_pgs_signal(A, n, K=10):
    L = utils.compute_graph_laplacian(A)
    _, U = np.linalg.eigh(L)
    D = np.random.randn(n, K)
    Ik = np.tile(np.eye(K), U.shape[0]//K)
    X = D @ Ik @ U.T
    return X, U @ Ik.T

def add_noise_to_graph_signal(X, noise_std):
    X_ = X + np.random.randn(X.shape[0], X.shape[1]) * noise_std
    return X_ - np.mean(X_, axis=0)
