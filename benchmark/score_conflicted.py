"""
Score benchmark results on the conflicted vignettes, where dialogue and narration point to
different answers, to measure which modality each model follows.

Outputs:
  <results_dir>/conflicted_scores.json   dialogue / description / distractor / no-parse counts
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, for `common`
from common.data import load_json, load_vignettes, save_json
from common.models import MODELS
from common.prompts import TYPES


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
        out[name] = load_json(path)
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


def save_scores(all_scores, path, split="all"):
    """
    Write the raw outcome counts per model so tables can be rebuilt elsewhere.

    Counts (not percentages) are saved at every breakdown the terminal summary
    shows — overall, per question type, and per split — so any table or plot of
    modality bias can be derived from this file alone.
    """
    out = {
        "split": split,
        "models": {
            name: {"overall": overall, "by_type": by_type, "by_split": by_split}
            for name, (overall, by_type, by_split) in all_scores.items()
        },
    }
    save_json(path, out, indent=2)
    print(f"\nScores → {path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tomstories_vignettes",        default="tomstories_vignettes/conflicted_vignettes.json")
    parser.add_argument("--results_dir", default="results/benchmark")
    parser.add_argument("--models",      nargs="+", default=None,
                        help="Restrict to these model names (default: every model with "
                             "conflicted mc result files in --results_dir)")
    parser.add_argument("--split",       choices=["train", "val", "all"], default="all")
    args = parser.parse_args()

    dataset = load_vignettes(args.tomstories_vignettes)
    print(f"Loaded {len(dataset)} conflicted samples from {args.tomstories_vignettes}")

    all_model_results = load_results(args.results_dir, models=args.models)
    if not all_model_results:
        print("No matching result files found.")
        return

    # Registry order first, then any result files not in the registry
    ordered  = [m for m in MODELS if m in all_model_results]
    ordered += sorted(m for m in all_model_results if m not in ordered)

    all_scores = {}
    for model_name in ordered:
        # Only keep sample IDs that are in the conflicted dataset
        relevant = {sid: r for sid, r in all_model_results[model_name].items()
                    if sid in dataset}
        if not relevant:
            print(f"  {model_name}: no overlapping sample IDs with conflicted dataset — skipping")
            continue
        all_scores[model_name] = score_model(relevant, dataset, split=args.split)

    if not all_scores:
        print("No scoreable results found.")
        return

    print_results(all_scores, split=args.split)
    save_scores(all_scores, os.path.join(args.results_dir, "conflicted_scores.json"),
                split=args.split)


if __name__ == "__main__":
    main()
