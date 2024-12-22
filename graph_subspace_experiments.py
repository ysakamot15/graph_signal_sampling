import mlflow
import hydra
import numpy as np
import matplotlib.pyplot as plt
import graph_signal_utils as utils
from graph_sampling_experiments import graph_sampling_and_recover
from signal_generate import (generate_band_limiation_signal,
                             generate_piecewise_constant_signal,
                             generate_pgs_signal,
                             add_noise_to_graph_signal)
from graph_subspace_estimation import(subspace_estimate_with_gft,
                                      subspace_estimate_with_pca,
                                      subspace_estimate_with_vertex_regularization,
                                      subspace_estimate_with_frequency_regularization)


@hydra.main(config_name="subspace_estimation_config",
            version_base=None, config_path="./config/")
def main(cfg):
    mlflow.set_experiment(cfg.experiment_name)
    mlflow.log_params(cfg)
    np.random.seed(cfg.seed)
    N = cfg.graph_params.number_of_graph_vertex # グラフの頂点数
    graph_adj_num = cfg.graph_params.number_of_adjacent # 接続する隣接数
    n = cfg.observation_size # 観測する信号数

    # グラフの作成
    V, adj = utils.generate_random_graph(N, graph_adj_num)

    # グラフ信号の生成
    if cfg.signal_params.signal_kind == 'band_limitation':
        K = cfg.signal_params.band_limitation_signal_params.number_of_remain
        noise_std = cfg.signal_params.band_limitation_signal_params.noise_std
        X, A = generate_band_limiation_signal(adj, n, K)
    elif cfg.signal_params.signal_kind == 'piecewise_constant':
        K = cfg.signal_params.piecewise_constant_signal_params.number_of_cluster
        noise_std = cfg.signal_params.piecewise_constant_signal_params.noise_std
        X, A = generate_piecewise_constant_signal(V, n, K)
    elif cfg.signal_params.signal_kind == 'pgs':
        K = cfg.signal_params.pgs_signal_params.period
        noise_std = cfg.signal_params.pgs_signal_params.noise_std
        X, A = generate_pgs_signal(adj, n, K)

    # グラフ信号にノイズを加える
    X_with_noise = add_noise_to_graph_signal(X, noise_std)
    x_noise = X_with_noise[0, :]
    _, ax = plt.subplots()
    utils.draw_graph_edge(V, adj, ax)
    plt.scatter(V[:, 0], V[:, 1], c=x_noise, cmap='cool', vmax=1.0, vmin=-1.0)
    plt.colorbar()
    plt.title("noise graph N={}".format(N))
    plt.savefig("noise_graph.png")
    mlflow.log_artifact("noise_graph.png", "images")
    plt.close()
 
    x_noise_ = utils.GraphFourierTransformer(adj).transform(x_noise)
    plt.plot(x_noise_, '.-')
    plt.savefig("noise_spectrum.png")
    mlflow.log_artifact("noise_spectrum.png", "images")
    plt.close()

    # 部分空間の推定
    if cfg.method_params.use_method == 'pca':
        A_pred = subspace_estimate_with_pca(X_with_noise, k=K)

    elif cfg.method_params.use_method == 'gft':
        A_pred = subspace_estimate_with_gft(X_with_noise, adj, k=K)

    elif cfg.method_params.use_method == 'vertex_regularization':
        alpha = cfg.method_params.vertex_regularization_params.alpha
        beta = cfg.method_params.vertex_regularization_params.beta
        A_pred, _ = subspace_estimate_with_vertex_regularization(
            X_with_noise, adj, K, alpha, beta)
    elif cfg.method_params.use_method == 'frequency_regularization':
        alpha = cfg.method_params.frequency_regularization_params.alpha
        beta = cfg.method_params.frequency_regularization_params.beta
        A_pred, _ = subspace_estimate_with_frequency_regularization(
            X_with_noise, adj, K, alpha, beta)

    # 正準角を求める
    principal_angle = utils.compute_principal_angle(A, A_pred)
    print("principal angle", principal_angle)
    mlflow.log_metric('principal angle', principal_angle)

    # 実験のためノイズを加えていない信号をサンプリングし、
    # 推定した部分空間の基底で復元
    x = X[0, :]
    M = cfg.sampling_size # サンプリングする頂点数
    graph_sampling_and_recover(V, x, N, M, adj, A_pred)

if __name__ == '__main__':
    main()
