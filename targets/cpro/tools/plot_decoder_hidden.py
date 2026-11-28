"""Plot measured post-decoder hidden state comparisons from the native 40-layer prefill."""
import argparse
import csv
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))
from qwen35_moe_dflash.common.accuracy import ERROR_RATIO_LIMIT, THRESHOLDS


PAIRS = (('cpro', 'cpu', 'CPro / CPU'), ('310p', 'cpu', '310P / CPU'), ('cpro', '310p', 'CPro / 310P'))
LAYERS = range(40)


def plot(matrices, directory):
    curves = {metric: {label: {} for _, _, label in PAIRS} for metric in ('cosine', 'error_ratio')}
    scopes = set()
    with matrices.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            match = re.fullmatch(r'hidden_(\d+)', row['operator'])
            metric = row['metric']
            if not match or metric not in curves or not re.fullmatch(r'valid[1-9]\d*', row['scope']):
                continue
            layer = int(match[1])
            scopes.add(row['scope'])
            for actual, reference, label in PAIRS:
                if row['actual'] != actual:
                    continue
                points = curves[metric][label]
                value = float(row[reference])
                if layer in points or not math.isfinite(value):
                    raise ValueError(f'{matrices}: duplicate or nonfinite {metric} for L{layer} {label}')
                points[layer] = value
    for series in curves.values():
        for label, points in series.items():
            if set(points) != set(LAYERS) or len(scopes) != 1:
                raise ValueError(f'{matrices}: expected hidden_0 through hidden_39 at one valid-token scope for {label}')

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    directory.mkdir(parents=True, exist_ok=True)
    valid = next(iter(scopes)).removeprefix('valid')
    for metric, series in curves.items():
        cosine = metric == 'cosine'
        threshold = THRESHOLDS['cosine']['value'] if cosine else ERROR_RATIO_LIMIT
        fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
        for (_, _, label), color, linestyle in zip(PAIRS, ('#005f83', '#bf5a24', '#635594'), ('-', '--', '-.')):
            points = series[label]
            ax.plot(LAYERS, [points[i] for i in LAYERS], label=label, color=color, linestyle=linestyle,
                    marker='o', markersize=3, linewidth=1.8)
        primary = series['CPro / CPU']
        first = next((i for i in LAYERS if (primary[i] <= threshold if cosine else primary[i] > threshold)), None)
        crossing = 'no threshold crossing' if first is None else f'first threshold crossing at L{first}'
        changes = [(primary[i-1] - primary[i] if cosine else primary[i] - primary[i-1], i) for i in range(1, 40)]
        delta, layer = max(changes, key=lambda p: p[0])
        jump = ('largest drop' if cosine else 'largest rise') + f' L{layer-1} to L{layer}'
        if delta <= 0:
            jump = 'no worsening between adjacent layers'
        status = f'CPro / CPU: {crossing} ({"≤" if cosine else ">"} {threshold:g}), {jump}'
        if first is not None:
            ax.scatter(first, primary[first], s=65, facecolors='white', edgecolors='#005f83', zorder=4)
        ax.set_title(f'40-layer prefill hidden states  |  {metric}\n{status}', loc='left', fontsize=12)
        ax.set_xlabel(f'Decoder layer (post-residual output, first {valid} tokens)')
        ax.set_ylabel('Cosine similarity' if cosine else 'Elements outside tolerance')
        ax.set_xticks([0, *range(4, 40, 4), 39])
        ax.set_xlim(-0.6, 39.6)
        if not cosine:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis='y', color='#d4d9df', linewidth=0.7)
        ax.legend(loc='best', frameon=False)
        output = directory / f'hidden_{metric}.png'
        fig.savefig(output, dpi=175)
        plt.close(fig)
        print(f'{metric}: {status} → {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('matrices', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    plot(args.matrices, args.output_dir or args.matrices.parent)
