import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
import argparse

class GraphFourierTransformer:
    def __init__(self, A):
        self.A = np.copy(A)
        self.L = compute_graph_laplacian(A)
        self.lam, self.U = np.linalg.eigh(self.L)

    def transform(self, X):
        return X @ self.U
    
    def inv_transform(self, Xf):
        return Xf @ self.U.T

def subspace_estimate_with_pca(X, k=0.9):
    pca = PCA(k, svd_solver='full')
    pca.fit(X)
    print(pca.components_.T.shape)
    return pca.components_.T

def subspace_estimate_with_gft(X, A, k):
    gft = GraphFourierTransformer(A)
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
    L = compute_graph_laplacian(adj)
    W = subspace_estimate_with_regularization(X, L, alpha, beta)
    print("rank",np.linalg.matrix_rank(W))

    Uk, _, _ = np.linalg.svd(W.T)
    return Uk[:, :k], W


def subspace_estimate_with_frequency_regularization(X, adj, k, alpha, beta):
    gft = GraphFourierTransformer(adj)
    X_ = gft.transform(X)
    P = np.eye(X.shape[1]) - np.eye(X.shape[1], k=-1)
    W = subspace_estimate_with_regularization(X_, P @ P.T, alpha, beta)
    print("rank",np.linalg.matrix_rank(W))
    Uk, _, _ = np.linalg.svd(W.T)
    return  gft.U @ Uk[:, :k]



# 組み合わせグラフラプラシアンを返す
def compute_graph_laplacian(A):
    D = np.diag(np.sum(A, axis=1))
    return D - A

# 2次元平面上ランダムに頂点を生成し、その近傍をつないだグラフを生成
# 平面上の座標と隣接行列を返す
def generate_random_graph(N, adj_num=4):
    V = np.random.randn(N, 2)
    dist = distance.cdist(V, V, metric='euclidean')
    connect = np.zeros((N, N))
    for i in range(N):
        sorted = np.argsort(dist[i, :])[1:adj_num + 1]
        connect[i, sorted] = 1
        connect[sorted, i] = 1
    dist[connect == 0] = np.inf
    adj = np.exp(-dist) - np.eye(N)
    return V, adj


def generate_piecewise_constant_signal(V, n, K=4):
    X = np.zeros((n, V.shape[0]))
    X[:, np.where(V[:, 0] > 0.5)[0]] = np.random.randn(n, 1)
    X[:, np.where(np.logical_and(V[:, 0] >= -0.5, V[:, 0] <= 0.5))[0]] = np.random.randn(n, 1)
    X[:, np.where(V[:, 0] < -0.5)[0]] = np.random.randn(n, 1)
    X += np.random.randn(X.shape[0], X.shape[1]) * 1
    return X - np.mean(X, axis=0)


def generate_pgs_signal(A, K=10):
    L = compute_graph_laplacian(A)
    _, U = np.linalg.eigh(L)
    d = np.random.randn(K)
    d = np.tile(d, U.shape[0]//K)
    print(d)
    x = d @ U.T
    return x, U

# グラフ周波数が低い方からK個までに帯域制限されたグラフ信号を生成
# グラフ信号と部分空間の基底を返す
def generate_band_limiation_signal(A, n, K=10):
    L = compute_graph_laplacian(A)
    _, U = np.linalg.eigh(L)
    d = np.random.randn(n, K)
    # d *= np.exp(-np.arange(0, K)/4.0)[None, :]
    X = d @ U[:, :K].T
    X += np.random.randn(X.shape[0], X.shape[1]) * 1
    return X - np.mean(X, axis=0)

# ランダムに部分集合を選択する行列を返す
def compute_random_sampling_matrix(N, M):
    tau = np.random.choice(range(N), size=M, replace=False)
    T = np.zeros((M, N))
    for i, t in enumerate(tau):
        T[i, t] = 1
    return T

# 指定されたサンプリング方法を用いてM個の点をサンプリング
# サンプリングされた信号とサンプリング作用素を返す
def vertex_sampling(x, M, samplinrg_operator_generator):
    St = samplinrg_operator_generator(x.shape[0], M)
    c = St @ x
    return c, St

# グラフの辺を描画
def draw_graph_edge(X, adj, ax):
    lines = []
    for i in range(X.shape[0]):
        for j in range(i + 1, X.shape[0]):
            if(adj[i, j] > 0):
                lines.append([X[i, :], X[j, :]])
    lc = LineCollection(lines, linewidths=0.5)
    ax.add_collection(lc)


def graph_sampling_and_recover(V, x, N, M, adj, A):
    # グラフと原信号を描画
    _, ax = plt.subplots()
    draw_graph_edge(V, adj, ax)
    plt.scatter(V[:, 0], V[:, 1], c=x, cmap='cool', vmax=1.0, vmin=-1.0)
    plt.colorbar()
    plt.title("original graph N={}".format(N))
    plt.savefig("origin_graph.png")
    plt.close()
 

    # ランダムに頂点をサンプリング
    c, St = vertex_sampling(x, M, compute_random_sampling_matrix)

    # サンプリング後のグラフ信号を描画
    _, ax = plt.subplots()
    draw_graph_edge(V, adj, ax)
    plt.scatter(V[:, 0], V[:, 1], c='white', linewidths=0.3, edgecolors='gray')
    plt.scatter(V[np.where(St==1)[1], 0], V[np.where(St==1)[1], 1],
                c=x[np.where(St==1)[1]], cmap='cool',  vmax=1.0, vmin=-1.0)
    plt.colorbar()
    plt.title("sampled graph N={} M={}".format(N, M))
    plt.savefig("sampled_graph.png")
    plt.close()

    # 補正行列の計算
    H = np.linalg.pinv(St @ A)

    # 復元
    x_hat = A @ H @ c

    # 復元後の信号と原信号の誤差計算
    err = np.linalg.norm(x_hat - x)
    print(err)

    # 復元後のグラフ信号を描画
    _, ax = plt.subplots()
    draw_graph_edge(V, adj, ax)
    plt.scatter(V[:, 0], V[:, 1], c=x_hat, cmap='cool', vmax=1.0, vmin=-1.0)
    plt.colorbar()
    plt.title("recovered graph M={} error={:.3e}".format(M, err))
    plt.savefig("recovered_graph.png")
    plt.close()

    L = compute_graph_laplacian(adj)
    _, U = np.linalg.eigh(L)
    x_ = U.T @ x_hat
    plt.plot(x_, '.-')
    plt.savefig("spec.png")
    plt.close()
    print(x_)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("M", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("seed", type=int)
    args = parser.parse_args()

    N = args.N #グラフの頂点数
    K = args.K #部分空間の次元
    M = args.M #サンプリング点数
    n = args.n #観測する信号数
    np.random.seed(args.seed)

    # グラフの生成
    V, adj = generate_random_graph(N)

    # 帯域制限されたグラフ信号の生成
    X = generate_band_limiation_signal(adj, n, K)
    # X = generate_piecewise_constant_signal(V, n, K)

    # A = subspace_estimate_with_pca(X, k=K)
    A = subspace_estimate_with_gft(X, adj, k=K)
    # A, W= subspace_estimate_with_vertex_regularization(X, adj, K, 1, 28)
    # A = subspace_estimate_with_frequency_regularization(X, adj, K, 0.1, 28)
    # print(np.max(np.abs(A2- A)))
    # # x, _ = generate_pgs_signal(adj, K)
    


    x = X[3, :]
    # w = W[0, :]
    graph_sampling_and_recover(V, x, N, M, adj, A)


if __name__ == '__main__':
    main()
