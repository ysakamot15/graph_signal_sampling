# Graph Signal Sampling and Recover

## Overview

This repository is intended to experiment with sampling and recover in graph signal processing.


## Getting Started

1. Clone the repository:

    ```bash
    git clone https://github.com/ysakamot15/graph_signal_sampling.git
    ```

2. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. If you want to run subspace estimation and sampling/recover experiments with default settings, execute the following command:

    ```bash
    python3 graph_subspace_experiments.py
    ```

4. To change the configuration, edit `config/subspace_estimation_config.yaml` directly or edit the configuration you wish to change on the command line as in the following example.

    ```bash
    python3 graph_subspace_experiments.py method_params.use_method=frequency_regularization signal_params.signal_kind=pgs
    ```
