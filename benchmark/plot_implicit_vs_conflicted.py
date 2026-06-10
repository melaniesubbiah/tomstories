"""
Scatter plot: implicit benchmark accuracy vs. conflicted dialogue-choice rate.
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from adjustText import adjust_text


MODE_COLORS = {"dialogue": "#4C72B0", "description": "#DD8452"}
def load_json(path):
    with open(path) as f:
        return json.load(f)


def implicit_accuracy(results: dict, dataset: dict, split="val"):
    correct = total = 0
    for sid, entry in results.items():
        if sid not in dataset:
            continue
        if split != "all" and dataset[sid].get("split") != split:
            continue
        pred = entry.get("pred")
        gold = dataset[sid]["answer"]
        if pred is not None:
            correct += int(pred == gold)
            total += 1
    return correct / total if total else None


def description_rate(results: dict, dataset: dict, split="val"):
    description = total = 0
    for sid, entry in results.items():
        if sid not in dataset:
            continue
        if split != "all" and dataset[sid].get("split") != split:
            continue
        if entry.get("flag"):
            continue
        pred = entry.get("pred")
        if pred is not None:
            description += int(pred == dataset[sid]["description_answer"])
            total += 1
    return description / total if total else None


def dialogue_rate(results: dict, dataset: dict, split="val"):
    description = total = 0
    for sid, entry in results.items():
        if sid not in dataset:
            continue
        if split != "all" and dataset[sid].get("split") != split:
            continue
        if entry.get("flag"):
            continue
        pred = entry.get("pred")
        if pred is not None:
            description += int(pred == dataset[sid]["dialogue_answer"])
            total += 1
    return description / total if total else None


def plot(model_points, out_path):
    fig, ax = plt.subplots(figsize=(9, 6))

    texts = []
    for name, vals in sorted(model_points.items()):
        if '1b' in name:
            continue
        x = vals["implicit_acc"]  * 100
        #x = (vals["implicit_acc"] - .5)/.4
        y = vals["description_rate"] * 100
        y2 = vals["dialogue_rate"] * 100
        #ax.scatter(y, y2, s=90, color=MODE_COLORS['description'], zorder=3, alpha=x)
        ax.scatter(x, y, s=90, color='purple', zorder=3, alpha=0.7)
        ax.scatter(x, y2, s=90, color=MODE_COLORS['dialogue'], zorder=3, alpha=0.7)
        texts.append(plt.text(x, y, name))
        texts.append(plt.text(x, y2, name))
    adjust_text(
        texts,
        ensure_inside_axes=False,
        arrowprops=dict(arrowstyle="-", color='gray', lw=0.5)
    )

    ax.set_xlabel("Implicit: accuracy (%)", fontsize=15)
    ax.set_ylabel("Conflicted: description-aligned answer chosen (%)", fontsize=15)
    ax.tick_params(labelsize=15)
    ax.grid(alpha=0.25)
    #ax.set_xlim(0, 100)
    #ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results_dir",     default="results/benchmark")
    parser.add_argument("--implicit_data",   default="tomstories_vignettes/implicit_vignettes.json",
                        help="Path to implicit dataset JSON")
    parser.add_argument("--conflicted_data", default="tomstories_vignettes/conflicted_vignettes.json",
                        help="Path to conflicted dataset JSON")
    parser.add_argument("--split",           default="all",
                        choices=["val", "train", "all"])
    parser.add_argument("--out",             default=None,
                        help="Output plot path (default: <results_dir>/implicit_vs_conflicted.png)")
    args = parser.parse_args()

    out_path = args.out or os.path.join(args.results_dir, "implicit_vs_conflicted.png")

    # Datasets
    impl_data_path = args.implicit_data
    conf_data_path = args.conflicted_data
    impl_dataset   = load_json(impl_data_path)
    conf_dataset   = load_json(conf_data_path)
    implicit_stem = args.implicit_data.split("/")[-1].split(".")[0]
    conflicted_stem = args.conflicted_data.split("/")[-1].split(".")[0]

    # Model points
    impl_pattern = os.path.join(args.results_dir, f"*_{implicit_stem}_mc_results.json")
    conf_pattern = os.path.join(args.results_dir, f"*_{conflicted_stem}_mc_results.json")

    impl_files = {
        re.sub(rf"_{re.escape(implicit_stem)}_mc_results$", "", Path(p).stem): p
        for p in glob.glob(impl_pattern)
    }
    conf_files = {
        re.sub(rf"_{re.escape(conflicted_stem)}_mc_results$", "", Path(p).stem): p
        for p in glob.glob(conf_pattern)
    }


    common = sorted(set(impl_files) & set(conf_files))
    if not common:
        print("No models found with results for both benchmarks.")
        print(f"  Implicit  files matched: {list(impl_files)}")
        print(f"  Conflicted files matched: {list(conf_files)}")
        return

    model_points = {}
    for name in common:
        impl_results = load_json(impl_files[name])
        conf_results = load_json(conf_files[name])
        ia = implicit_accuracy(impl_results, impl_dataset, args.split)
        dr = description_rate(conf_results, conf_dataset, args.split)
        di = dialogue_rate(conf_results, conf_dataset, args.split)
        if ia is not None and dr is not None:
            model_points[name] = {"implicit_acc": ia, "description_rate": dr, "dialogue_rate": di}
            print(f"  {name:<45}  implicit={ia*100:.1f}%  description={dr*100:.1f}% dialogue={di*100:.1f}%")
        else:
            print(f"  Skipping {name}: insufficient {args.split}-split tomstories_vignettes")

    if not model_points:
        print("Nothing to plot.")
        return

    plot(model_points, out_path)


if __name__ == "__main__":
    main()
