import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASETS = {'sepsis': 'Sepsis', 'BPIC_2013_closed': 'BPIC 2013', 'BPIC_2020_travel': 'BPIC 2020'}
LOSSES = ['baseline', 'GLL', 'LLL']
ALPHAS = {'GLL': 0.75, 'LLL': 0.25}
STRATEGIES = ['temperature', 'greedy']
COLORS = {'baseline': '#1b9e77', 'GLL': '#d95f02', 'LLL': '#1f77b4'}
METRICS = {
    'satisfiability': ('test_satisfiability', 'SAT', 'fig4_lstm_satisfiability'),
    'similarity': ('test_similarity_scaled', 'Similarity', 'fig5_lstm_similarity'),
}


def load_results(results_dir, csv_path=None):
    if csv_path is not None:
        df = pd.read_csv(csv_path)
    else:
        files = sorted(Path(results_dir).glob('*/*/results.csv'))
        if not files:
            raise FileNotFoundError(f'No results.csv files found under {results_dir}')
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df[df['architecture'] == 'LSTM']


def select_configurations(df):
    keep = df['loss'] == 'baseline'
    for loss, alpha in ALPHAS.items():
        keep |= (df['loss'] == loss) & np.isclose(df['alpha'], alpha)
    return df[keep]


def summarize(df, metric):
    keys = ['dataset', 'sampling_strategy', 'loss', 'noise']
    per_prefix = df.groupby(keys + ['prefix_length'])[metric].mean().reset_index()
    return per_prefix.groupby(keys)[metric].agg(mean='mean', std=lambda x: np.std(x, ddof=0))


def summarize_runs(df, metric):
    keys = ['dataset', 'sampling_strategy', 'loss', 'noise']
    per_run = df.groupby(keys + ['run_id'])[metric].mean().reset_index()
    return per_run.groupby(keys)[metric].agg(mean='mean', std=lambda x: np.std(x, ddof=1), runs='count')


def plot_metric(summary, ylabel, out_path):
    present = summary.index.get_level_values('dataset')
    datasets = {dataset: title for dataset, title in DATASETS.items() if dataset in present}
    fig = plt.figure(figsize=(9, 11.5 * len(datasets) / len(DATASETS)))
    subfigs = fig.subfigures(len(datasets), 1, hspace=0.04, squeeze=False).ravel()
    width = 0.27

    for subfig, (dataset, title) in zip(subfigs, datasets.items()):
        subfig.suptitle(rf'$\mathbf{{{title.replace(" ", r"\ ")}}}$', fontsize=17, y=1.0)
        axes = subfig.subplots(1, 2)

        for ax, strategy in zip(axes, STRATEGIES):
            noise_levels = sorted(summary.loc[dataset, strategy].index.get_level_values('noise').unique())
            x = np.arange(len(noise_levels))

            for i, loss in enumerate(LOSSES):
                stats = summary.loc[(dataset, strategy, loss)].reindex(noise_levels)
                ax.bar(x + (i - 1) * width, stats['mean'], width, yerr=stats['std'], capsize=3,
                       color=COLORS[loss], edgecolor='black', linewidth=0.8, label=loss,
                       error_kw={'elinewidth': 1, 'capthick': 1}, zorder=2)

            ax.set_xticks(x, noise_levels)
            ax.set_ylim(0.4, 1.02)
            ax.set_yticks(np.arange(0.4, 1.01, 0.1))
            ax.yaxis.grid(True, linestyle='--', color='lightgrey', zorder=0)
            ax.set_axisbelow(True)
            ax.set_xlabel('Noise level')
            ax.set_ylabel(ylabel)
            ax.legend(loc='lower left', fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(out_path.with_suffix(f'.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Reproduce Figs. 4 and 5 of the paper for the LSTM results')
    parser.add_argument('--results-dir', default='results', help='folder with <dataset>/<experiment>/results.csv')
    parser.add_argument('--csv', default=None, help='single results CSV to use instead of --results-dir')
    parser.add_argument('--out-dir', default='results/paper_figures', help='output folder for the figures')
    args = parser.parse_args()

    plt.rcParams['mathtext.fontset'] = 'cm'
    df = select_configurations(load_results(args.results_dir, args.csv))

    for metric, ylabel, filename in METRICS.values():
        summary = summarize(df, metric)
        plot_metric(summary, ylabel, Path(args.out_dir) / filename)
        summary.round(4).to_csv(Path(args.out_dir) / f'{filename}.csv')
        summarize_runs(df, metric).round(4).to_csv(Path(args.out_dir) / f'{filename}_runs.csv')


if __name__ == '__main__':
    main()
