import mlflow
import hydra
import matplotlib.pyplot as plt
import numpy as np
import graph_signal_utils as utils
from signal_generate import generate_band_limiation_signal


def graph_sampling_and_recover(V, x, N, M, adj, A):
    # グラフと原信号を描画
    _, ax = plt.subplots()
    utils.draw_graph_edge(V, adj, ax)
    plt.scatter(V[:, 0], V[:, 1], c=x, cmap='cool', vmax=1.0, vmin=-1.0)
    plt.colorbar()
    plt.title("original graph N={}".format(N))
    plt.savefig("origin_graph.png")
    mlflow.log_artifact("origin_graph.png", "images")
    plt.close()
 
    # 原信号のスペクトラムの表示
    gft = utils.GraphFourierTransformer(adj)
    x_ = gft.transform(x)
    plt.plot(x_, '.-')
    plt.savefig("original_spectrum.png")
    mlflow.log_artifact("original_spectrum.png", "images")
    plt.close()

    # ランダムに頂点をサンプリング
    c, St = utils.vertex_sampling(x, M, utils.compute_random_sampling_matrix)

    # サンプリング後のグラフ信号を描画
    _, ax = plt.subplots()
    utils.draw_graph_edge(V, adj, ax)
    plt.scatter(V[:, 0], V[:, 1], c='white', linewidths=0.3, edgecolors='gray')
    plt.scatter(V[np.where(St==1)[1], 0], V[np.where(St==1)[1], 1],
                c=x[np.where(St==1)[1]], cmap='cool',  vmax=1.0, vmin=-1.0)
    plt.colorbar()
    plt.title("sampled graph N={} M={}".format(N, M))
    plt.savefig("sampled_graph.png")
    mlflow.log_artifact("sampled_graph.png", "images")
    plt.close()

    # 補正行列の計算
    H = np.linalg.pinv(St @ A)

    # 復元
    x_hat = A @ H @ c

    # 復元後の信号と原信号の誤差計算
    err = np.linalg.norm(x_hat - x)
    print(err)
    mlflow.log_metric('recover error', err)

    # 復元後のグラフ信号を描画
    _, ax = plt.subplots()
    utils.draw_graph_edge(V, adj, ax)
    plt.scatter(V[:, 0], V[:, 1], c=x_hat, cmap='cool', vmax=1.0, vmin=-1.0)
    plt.colorbar()
    plt.title("recovered graph M={} error={:.3e}".format(M, err))
    plt.savefig("recovered_graph.png")
    mlflow.log_artifact("recovered_graph.png", "images")
    plt.close()

    # 復元信号のスペクトラムの表示
    x_hat_ = gft.transform(x_hat)
    plt.plot(x_hat_, '.-')
    plt.savefig("recovered_spectrum.png")
    mlflow.log_artifact("recovered_spectrum.png", "images")
    plt.close()

@hydra.main(config_name="graph_sampling_config",
            version_base=None, config_path="./config/")
def main(cfg):
    mlflow.set_experiment(cfg.experiment_name)
    N = cfg.N #グラフの頂点数
    K = cfg.K #部分空間の次元
    M = cfg.M #サンプリング点数
    np.random.seed(cfg.seed)

    # グラフの生成
    V, adj = utils.generate_random_graph(N)

    # 帯域制限されたグラフ信号の生成
    x, A = generate_band_limiation_signal(adj, 1, K)

    # サンプリングと復元
    graph_sampling_and_recover(V, x.squeeze(), N, M, adj, A)

if __name__ == '__main__':
    main()