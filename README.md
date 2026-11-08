# nesy-suffix-prediction-dfa

**Neurosymbolic suffix prediction using Deterministic Finite Automata (DFA).**

*Axel Mezini, Elena Umili, Ivan Donadello, Fabrizio Maria Maggi, Matteo Mancanelli, Fabio Patrizi*

This repository contains the source code accompanying the paper: [Neuro-Symbolic Predictive Process Monitoring](https://authors.elsevier.com/c/1nGKE15hGZkPXv). 
The implementation provides the experimental framework used in the paper, including model execution, DFA-based symbolic components, and evaluation pipelines.

> **Replication.** This repository is a replication of the paper. The code is taken from the authors' original repository, [axelmezini/nesy-suffix-prediction-dfa](https://github.com/axelmezini/nesy-suffix-prediction-dfa), with full credit to them. It adds the configuration we used and the results we obtained (see [Replication Setup and Results](#replication-setup-and-results)).

---

## Overview

This project implements a neurosymbolic approach to suffix prediction that integrates neural sequence modeling with symbolic reasoning via deterministic finite automata (DFA).

The repository is intended primarily for:

- Reproducing the experiments reported in the paper  
- Running the provided suffix prediction pipeline  
- Extending the framework for further research  

For theoretical details, methodology, and evaluation discussion, please refer to the paper.

---

## Project Structure

```
Seminar-Paper-Replication-Code/
│
├── src/                       # Core implementation
│   ├── runner.py              # Main entry point
│   └── plot_paper_figures.py  # Reproduces Figs. 4 and 5 from the results
├── environment.yaml           # Conda environment (Python dependencies)
├── config_example.yaml        # Configuration template (our replication setup)
├── data/                      # Datasets
├── results/                   # Results
├── README.md
├── CITATION.bib
├── .gitignore
└── LICENSE
```

---

## Prerequisites

- Python 3.12 (installed by the conda environment)
- CUDA
- [Mona](https://www.brics.dk/mona/)

---

## Setup

1. Clone the Repository
    ```bash
    git clone https://github.com/ioannisdrossas/Seminar-Paper-Replication-Code.git
    cd Seminar-Paper-Replication-Code
    ```

2. Install dependencies
    ```bash
    conda config --set channel_priority flexible
    conda env create -f environment.yaml
    ```

3. Create configuration file
    ```bash
    cp config_example.yaml config.yaml
    ```
    Edit `config.yaml` to set output directories (`root_path: './'` points to the repository root, which must contain `data/`; results are written to `<root_path>/results/<dataset>/`)

---

## Running the Code

To reproduce a specific experiment:
1. Ensure MONA is correctly installed.
2. Ensure the correct dataset is available in the `data/` directory.
3. Set experiment parameters in `config.yaml` to match the paper setup. For exact parameter settings, refer to the experimental section of the paper.
4. Activate the environment:
    ```bash
    conda activate env_sp
    ```
5. Run from the repository root:
    ```bash
    python src/runner.py
    ```

Results will be written to the configured output directory.

---

## Replication Setup and Results

`config_example.yaml` holds the setup of our replication. Copying it to `config.yaml` (Setup, step 3) and running `python src/runner.py` reproduces our experiments. It differs from the paper's setup as follows:

| Parameter | Paper | Ours |
|---|---|---|
| `architectures` | LSTM, Transformer | LSTM |
| `nr_runs` | 15 | 5 |
| `gll_samples` (Monte Carlo continuations per trace in GLL) | 10 | 5 |

All other parameters (datasets, noise levels, alpha values, training hyperparameters) are the same as in the paper. `gll_samples` is the only addition to the original code: it sets the number of Monte Carlo samples GLL draws per trace and defaults to 10 (the paper's value) if it is missing from `config.yaml`.

Our results are in `results/<dataset>/<timestamp>_noise<N>_alpha<A>/`. Each folder holds `results.csv` (train and test metrics for every run, loss, prefix length and sampling strategy), the plots `test_satisfiability.png` and `test_similarity_scaled.png`, and the trained model weights `run<i>/r<i>_lstm_<loss>.pt`. `results/paper_results.csv` holds the original authors' results.

The baseline does not depend on alpha, so we trained it once per dataset and noise level and reused its results for every alpha; its weights (`*_baseline.pt`) are therefore saved only in the first alpha folder of each batch of experiments, while the code in this repository retrains it for every alpha with the same seeds.

To redraw Figs. 4 and 5 of the paper (LSTM, GLL with alpha = 0.75, LLL with alpha = 0.25) from our results into `results/paper_figures/`:
```bash
python src/plot_paper_figures.py
```
Use `python src/plot_paper_figures.py --csv results/paper_results.csv --out-dir <folder>` to draw them from the original results instead.

---

## Citation

If you use this code in your research, please cite:
```bibtex
@article{MEZINI2026102762,
    title = {Neuro-Symbolic Predictive Process Monitoring},
    journal = {Information Systems},
    volume = {141},
    pages = {102762},
    year = {2026},
    issn = {0306-4379},
    doi = {https://doi.org/10.1016/j.is.2026.102762},
    url = {https://www.sciencedirect.com/science/article/pii/S0306437926000761},
    author = {Axel Mezini and Elena Umili and Ivan Donadello and Fabrizio Maria Maggi and Matteo Mancanelli and Fabio Patrizi},
    keywords = {Suffix prediction, Neuro-Symbolic AI, Deep learning with logical knowledge, Linear Temporal Logic, Differentiable automata},
    abstract = {This paper addresses the problem of suffix prediction in Business Process Management (BPM) by proposing a Neuro-Symbolic Predictive Process Monitoring (PPM) approach that integrates data-driven learning with temporal logic-based prior knowledge. While recent approaches leverage deep learning models for suffix prediction, they often fail to satisfy even basic logical constraints due to the lack of explicit integration of domain knowledge during training. We propose a novel method to incorporate Linear Temporal Logic over finite traces (LTLf ) into the training process of autoregressive sequence predictors. Our approach introduces a differentiable logical loss function, defined using a soft approximation of LTLf semantics and the Gumbel-Softmax trick, which can be combined with standard predictive losses. This ensures that the model learns to generate suffixes that are both accurate and logically consistent. Experimental evaluation on three real-world datasets shows that our method improves suffix prediction accuracy and compliance with temporal constraints. We also introduce two variants of the logic loss (local and global) and demonstrate their effectiveness under noisy and realistic settings. While developed in the context of BPM, our framework is applicable to any symbolic sequence generation task and contributes to advancing Neuro-Symbolic AI.}
}
```


## License

This project is licensed under the MIT License.  
See the `LICENSE` file for details.
