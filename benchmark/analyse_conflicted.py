"""
Analyse benchmark results on the conflicted vignette dataset.
"""

import argparse
import glob
import json
import os
import re

import numpy as np


TYPES = ["emotion", "wound", "fear", "motivation"]


def load_results(results_dir, models=None):
    pattern = os.path.join(results_dir, "*_conflicted_vignettes_mc_results.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No *_mc_results.json files found in {results_dir}")

    out = {}
    for path in paths:
        name = re.sub(r"_conflicted_vignettes_mc_results\.json$", "", os.path.basename(path))
        if models and name not in models:
            continue
        with open(path) as f:
            out[name] = json.load(f)
    return out


def score_model(model_results, dataset, split="all"):
    def empty():
        return {"dialogue": 0, "description": 0, "distractor": 0, "no_parse": 0, "total": 0}

    overall  = empty()
    by_type  = {t: empty() for t in TYPES}
    by_split = {"train": empty(), "val": empty()}

    for sid, entry in model_results.items():
        if sid not in dataset:
            continue
        d = dataset[sid]

        if split != "all" and d.get("split") != split:
            continue

        typ = d["type"]
        sp  = d.get("split", "train")

        pred = entry.get("pred")
        if pred is None:
            outcome = "no_parse"
        elif pred == d["dialogue_answer"]:
            outcome = "dialogue"
        elif pred == d["description_answer"]:
            outcome = "description"
        else:
            outcome = "distractor"

        for bucket in (overall, by_type[typ], by_split[sp]):
            bucket[outcome] += 1
            bucket["total"] += 1

    return overall, by_type, by_split


def pct(n, total):
    return f"{n / total * 100:5.1f}%" if total > 0 else "   n/a"


def print_results(all_scores, split="all"):
    col_w = 28
    header = (
        f"{'Model':<{col_w}}  {'Dialogue':>9}  {'Description':>12}  {'Distractor':>10}  {'No parse':>9}  {'N':>5}"
    )
    sep = "=" * len(header)

    print(f"\n{'─'*len(header)}")
    print(f"  Split: {split}  |  Modality bias analysis")
    print(sep)
    print(header)
    print(sep)

    for model_name, (overall, by_type, by_split) in all_scores.items():
        ov = overall
        n  = ov["total"]
        print(
            f"{model_name:<{col_w}}"
            f"  {pct(ov['dialogue'],    n):>9}"
            f"  {pct(ov['description'], n):>12}"
            f"  {pct(ov['distractor'],  n):>10}"
            f"  {pct(ov['no_parse'],    n):>9}"
            f"  {n:>5}"
        )

    print(sep)

    # Per-type breakdown
    print(f"\n{'─'*len(header)}")
    print("  Per-type breakdown")
    print(sep)
    type_header = (
        f"{'Model / Type':<{col_w}}  {'Dialogue':>9}  {'Description':>12}  {'Distractor':>10}  {'No parse':>9}  {'N':>5}"
    )
    print(type_header)
    print(sep)

    for model_name, (overall, by_type, by_split) in all_scores.items():
        print(f"{model_name}")
        for typ in TYPES:
            b = by_type[typ]
            n = b["total"]
            print(
                f"  {typ:<{col_w-2}}"
                f"  {pct(b['dialogue'],    n):>9}"
                f"  {pct(b['description'], n):>12}"
                f"  {pct(b['distractor'],  n):>10}"
                f"  {pct(b['no_parse'],    n):>9}"
                f"  {n:>5}"
            )
    print(sep)


def print_latex_table(all_scores):
    # Determine model order from MODELS registry if available, else sort
    try:
        from benchmark import MODELS
        model_order = [m for m in MODELS if any(m in k for k in all_scores)]
        ordered = [k for k in model_order if k in all_scores]
        ordered += sorted(k for k in all_scores if k not in ordered)
    except Exception:
        ordered = sorted(all_scores)

    # Find per-column maxima for bolding
    def vals(key):
        return [all_scores[m][0][key] / all_scores[m][0]["total"] * 100
                for m in ordered if all_scores[m][0]["total"] > 0]
    best = {k: max(vals(k)) for k in ("dialogue", "description", "distractor", "no_parse")}

    def fmt(n, total, best_val):
        if total == 0:
            return "--"
        v = n / total * 100
        s = f"{v:5.1f}"
        return r"\textbf{" + s.strip() + "}" if abs(v - best_val) < 0.05 else s.strip()

    print()
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{Dialogue} & \textbf{Description} & \textbf{Distractor} & \textbf{No Parse}  \\")
    print(r"\midrule")
    for name in ordered:
        overall, _, _ = all_scores[name]
        n = overall["total"]
        cells = [
            name,
            fmt(overall["dialogue"],    n, best["dialogue"]),
            fmt(overall["description"], n, best["description"]),
            fmt(overall["distractor"],  n, best["distractor"]),
            fmt(overall["no_parse"],    n, best["no_parse"]),
        ]
        print("  " + " & ".join(cells) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tomstories_vignettes",        default="tomstories_vignettes/conflicted_vignettes.json")
    parser.add_argument("--results_dir", default="results/benchmark")
    parser.add_argument("--models",      nargs="+", default=None,
                        help="Restrict to these model names (default: all mc result files)")
    parser.add_argument("--split",       choices=["train", "val", "all"], default="all")
    args = parser.parse_args()

    with open(args.tomstories_vignettes) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} conflicted samples from {args.tomstories_vignettes}")

    all_model_results = load_results(args.results_dir, models=args.models)
    if not all_model_results:
        print("No matching result files found.")
        return

    all_scores = {}
    for model_name, model_results in all_model_results.items():
        # Only keep sample IDs that are in the conflicted dataset
        relevant = {sid: r for sid, r in model_results.items() if sid in dataset}
        if not relevant:
            print(f"  {model_name}: no overlapping sample IDs with conflicted dataset — skipping")
            continue
        all_scores[model_name] = score_model(relevant, dataset, split=args.split)

    if all_scores:
        print_results(all_scores, split=args.split)
        print_latex_table(all_scores)
    else:
        print("No scoreable results found.")


if __name__ == "__main__":
    main()
