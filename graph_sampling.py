import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
from sklearn import linear_model

class GraphFourierTransformer:
    def __init__(self, A):
        self.A = np.copy(A)
        self.L = compute_graph_laplacian(A)
        self.lam, self.U = np.linalg.eigh(self.L)

    def transform(self, X):
        return X @ self.U
    
    def inv_transform(self, Xf):
        return Xf @ self.U.T

def compute_graph_laplacian(A):
    D = np.diag(np.sum(A, axis=1))
    return D - A

def make_random_sampler(n, k):
    tau = np.random.choice(range(n), size=k, replace=False)
    T = np.zeros((k, n))
    for i, t in enumerate(tau):
        T[i, t] = 1
    return T

def vertex_sampling(x, k, samplinrg_operator_maker):
    St = samplinrg_operator_maker(x.shape[0], k)
    c = St @ x
    return c, St

def make_correction_transformation(St, A):
    return np.linalg.pinv(St @ A)

def make_random_graph(N):
    X = np.random.randn(N, 2)
    dist = distance.cdist(X, X, metric='euclidean')
    k = 4
    Adj = np.zeros((N, N))
    for i in range(N):
        sorted = np.argsort(dist[i, :])[1:k + 1]
        Adj[i, sorted] = 1
        Adj[sorted, i] = 1

    dist[Adj == 0] = np.inf
    A = np.exp(-dist) - np.eye(N)
    return X, A

def make_band_limiation_signals(A, m = 1):
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    l, U = np.linalg.eigh(L)
   # l[:120] = 0
    Lam = np.diag(l)
    k = 10
    d = np.random.randn(m, k)
    d /= np.sqrt(np.diag(d @ d.T))[:, None]
    x_ = d @ U[:, :k].T + np.random.randn(A.shape[0])*0.1 #U @ Lam @ U.T @ d
    return x_, U

def subspace_estimate_with_pca(X, k=0.9):
    pca = PCA(k, svd_solver='full')
    pca.fit(X)
    return pca.components_.T

def subspace_estimate_with_gft(X, A, thresh_percent = 0.7):
    gft = GraphFourierTransformer(A)
    Xf = gft.transform(X)
    s = np.sqrt(np.sum((Xf * Xf), axis=0))
    s_average = np.average(s)
    return gft.U[:, s > s_average * thresh_percent]

def subspace_estimate_with_group_lasso(X, A, alpha = 0.00297):
    gft = GraphFourierTransformer(A)
    clf = linear_model.MultiTaskLasso(alpha=alpha, fit_intercept = False)
    clf.fit(gft.U, X.T)
    s = np.sum(np.abs(clf.coef_), axis=0)
    return gft.U[:, s > 1e-15]

np.random.seed(1)

X, A = make_random_graph(150)

x, U = make_band_limiation_signals(A, 5)


fig, ax = plt.subplots()
lines = []
for i in range(X.shape[0]):
    for j in range(i + 1, X.shape[0]):
        if(A[i, j] > 0):
            lines.append([X[i, :], X[j, :]])
lc = LineCollection(lines, linewidths=0.5)
ax.add_collection(lc)
plt.scatter(X[:, 0], X[:, 1], c=x[0, :], cmap='cool', vmin=-0.3, vmax=0.3)
plt.colorbar()

plt.savefig("test.png")
plt.close()



# UU = subspace_estimate_with_gft(x, A)
UU = subspace_estimate_with_group_lasso(x, A)

m = UU.shape[1]
c, St = vertex_sampling(x[0, :], m, make_random_sampler)
H = make_correction_transformation(St, UU)
x_ = UU @ H @ c


fig, ax = plt.subplots()
lines = []
for i in range(X.shape[0]):
    for j in range(i + 1, X.shape[0]):
        if(A[i, j] > 0):
            lines.append([X[i, :], X[j, :]])
lc = LineCollection(lines, linewidths=0.5)
ax.add_collection(lc)
plt.scatter(X[:, 0], X[:, 1], c=x_, cmap='cool', vmin=-0.3, vmax=0.3)
plt.colorbar()

plt.savefig("test2.png")

