import numpy as np
from scipy.spatial import distance
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

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

# 組み合わせグラフラプラシアンを返す
def compute_graph_laplacian(A):
    D = np.diag(np.sum(A, axis=1))
    return D - A

def compute_principal_angle(A1, A2):
    #直交化
    UQ , _ = np.linalg.qr(A1)
    VQ , _ = np.linalg.qr(A2)

    #特異値分解でcos(theta)と係数a, bを計算
    _, D, _ = np.linalg.svd(UQ.T @ VQ)
   
    return np.sum(D)

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

class GraphFourierTransformer:
    def __init__(self, A):
        self.A = np.copy(A)
        self.L = compute_graph_laplacian(A)
        self.lam, self.U = np.linalg.eigh(self.L)

    def transform(self, X):
        return X @ self.U
    
    def inv_transform(self, Xf):
        return Xf @ self.U.T
