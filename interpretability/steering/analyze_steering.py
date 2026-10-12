"""
Score steered vignettes with the NRC emotion-intensity and VAD lexicons, then test the
paired differences between conditions for each model and across models.

Outputs:
  <output_dir>/lexicon_stats_per_model.csv   per model, comparison and metric
  <output_dir>/lexicon_stats_summary.csv     the same, aggregated across models
  <output_dir>/lexicon_text_scores.csv       per-vignette scores, with --save_text_scores
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

EMOTIONS = ["anger", "disgust", "fear", "sadness", "joy", "trust", "anticipation", "surprise"]
VAD_DIMS = ["valence", "arousal", "dominance"]
METRIC_GROUPS = {
    "emotion_rate":      [f"{e}_rate" for e in EMOTIONS] + ["coverage"],
    "emotion_intensity": [f"{e}_intensity" for e in EMOTIONS],
    "emotion_share":     [f"{e}_share" for e in EMOTIONS],
    "vad":               [f"{d}_full" for d in VAD_DIMS],
    "length":            ["n_tokens"],
}
FILE_RE = re.compile(r"^(?P<model>.+)_steered_vignettes_(?P<cond>implicit|explicit|none)"
                     r"(?P<tag>_x[0-9.]+)?\.json$")
TOKEN_RE = re.compile(r"\b\w+\b")


# ── Lexicons ───────────────────────────────────────────────────────────────────

def _read_lexicon(path, expected_cols):
    df = pd.read_csv(path, sep="\t")
    if "term" not in df.columns:                      # file without a header row
        df = pd.read_csv(path, sep="\t", header=None, names=expected_cols)
    return df


def load_eil(path):
    """{word: [(emotion, intensity), ...]} from the NRC Emotion Intensity Lexicon."""
    df = _read_lexicon(path, ["term", "emotion", "score"])
    lex = defaultdict(list)
    for term, emo, score in df[["term", "emotion", "score"]].itertuples(index=False):
        lex[str(term).lower()].append((emo, float(score)))
    return dict(lex)


def load_vad(path):
    """{word: (valence, arousal, dominance)} from the NRC VAD Lexicon."""
    df = _read_lexicon(path, ["term"] + VAD_DIMS)
    return {str(t).lower(): (float(v), float(a), float(d))
            for t, v, a, d in df[["term"] + VAD_DIMS].itertuples(index=False)}


# ── Per-text scores ────────────────────────────────────────────────────────────

def score_text(text, eil, vad=None):
    tokens = TOKEN_RE.findall((text or "").lower())
    n = len(tokens)
    sums, counts, matched = defaultdict(float), defaultdict(int), 0
    for w in tokens:
        entries = eil.get(w)
        if entries:
            matched += 1
            for emo, s in entries:
                sums[emo] += s
                counts[emo] += 1
    out = {}
    for emo in EMOTIONS:
        # share: intensity per lexicon word of any emotion (current measure)
        out[f"{emo}_share"] = sums[emo] / matched if matched else np.nan
        # rate: intensity per 100 tokens (amount of this emotion; recommended primary)
        out[f"{emo}_rate"] = 100 * sums[emo] / n if n else np.nan
        # intensity: mean intensity of this emotion's own words (NaN if none present)
        out[f"{emo}_intensity"] = sums[emo] / counts[emo] if counts[emo] else np.nan
        out[f"{emo}_count"] = counts[emo]
    out["coverage"] = matched / n if n else np.nan
    out["n_tokens"] = n
    if vad is not None:
        vals = np.array([vad[w] for w in tokens if w in vad], dtype=float).reshape(-1, 3)
        for j, dim in enumerate(VAD_DIMS):
            col = vals[:, j]
            has = len(col) > 0
            out[f"{dim}_full"] = col.mean() if has else np.nan
            out[f"{dim}_neg"]  = np.where(col < 0, col, 0).mean() if has else np.nan
            out[f"{dim}_pos"]  = np.where(col > 0, col, 0).mean() if has else np.nan
    return out


# ── Loading vignettes ──────────────────────────────────────────────────────────

def load_vignettes(path):
    with open(path) as f:
        data = json.load(f)
    vign = data.get("vignettes", data) if isinstance(data, dict) else {}
    return {k: v["output"] for k, v in vign.items() if isinstance(v, dict) and "output" in v}


def discover(vignettes_dir, models=None):
    found, baselines = defaultdict(dict), {}
    for p in sorted(Path(vignettes_dir).glob("*_steered_vignettes_*.json")):
        m = FILE_RE.match(p.name)
        if not m or (models and m["model"] not in models):
            continue
        if m["cond"] == "none":
            baselines[m["model"]] = p
        else:
            found[(m["model"], m["tag"] or "")][m["cond"]] = p
    for (model, tag), conds in found.items():
        if model in baselines:
            conds["none"] = baselines[model]
    return dict(found)


# ── Statistics ─────────────────────────────────────────────────────────────────

def paired_tests(a, b):
    """Paired stats on aligned arrays; pairs with NaN on either side are dropped."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    res = {"n": int(ok.sum()), "mean_a": np.nan, "mean_b": np.nan, "mean_diff": np.nan,
           "d_z": np.nan, "t_p": np.nan, "wilcoxon_p": np.nan}
    if len(a) < 2:
        return res
    d = a - b
    res.update(mean_a=a.mean(), mean_b=b.mean(), mean_diff=d.mean())
    sd = d.std(ddof=1)
    res["d_z"] = d.mean() / sd if sd > 0 else np.nan
    res["t_p"] = stats.ttest_rel(a, b).pvalue if sd > 0 else np.nan
    try:
        res["wilcoxon_p"] = stats.wilcoxon(a, b).pvalue
    except ValueError:                                   # e.g. all differences are zero
        pass
    return res


def fdr(pvals):
    p = np.asarray(pvals, float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out


def compare(scores_a, scores_b, groups):
    """scores_x: DataFrame indexed by key. Returns rows with per-group FDR."""
    keys = scores_a.index.intersection(scores_b.index)
    rows = []
    for group, metrics in groups.items():
        g_rows = []
        for metric in metrics:
            if metric not in scores_a.columns:
                continue
            r = paired_tests(scores_a.loc[keys, metric], scores_b.loc[keys, metric])
            g_rows.append({"group": group, "metric": metric, **r})
        for col in ("t_p", "wilcoxon_p"):
            for r, q in zip(g_rows, fdr([r[col] for r in g_rows])):
                r[col.replace("_p", "_q")] = q
        rows += g_rows
    return rows, len(keys)


def summarize_across_models(df, alpha):
    out = []
    for (tag, comp, metric), g in df.groupby(["scale", "comparison", "metric"], sort=False):
        g = g.dropna(subset=["mean_diff"])
        if g.empty:
            continue
        diffs = g["mean_diff"].to_numpy()
        try:
            p = stats.wilcoxon(diffs).pvalue if len(diffs) >= 2 and np.any(diffs != 0) else np.nan
        except ValueError:
            p = np.nan
        out.append({
            "scale": tag, "comparison": comp, "group": g["group"].iloc[0], "metric": metric,
            "n_models": len(g),
            "mean_d_z": g["d_z"].mean(),
            "n_pos": int((g["mean_diff"] > 0).sum()),
            "n_sig_pos": int(((g["t_q"] < alpha) & (g["mean_diff"] > 0)).sum()),
            "n_sig_neg": int(((g["t_q"] < alpha) & (g["mean_diff"] < 0)).sum()),
            "across_models_wilcoxon_p": p,
        })
    return pd.DataFrame(out)


# ── Main ───────────────────────────────────────────────────────────────────────

def score_file(path, eil, vad, cache):
    if path not in cache:
        texts = load_vignettes(path)
        cache[path] = pd.DataFrame({k: score_text(t, eil, vad) for k, t in texts.items()}).T
    return cache[path]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vignettes_dir", type=Path, default=Path("results/steered_vignettes"))
    p.add_argument("--models", nargs="+", default=None,
                   help="Restrict to these model names (default: every model with steered "
                        "vignettes in --vignettes_dir)")
    p.add_argument("--pair", nargs=2, type=Path, metavar=("A_FILE", "B_FILE"),
                   help="Compare two specific files (A - B) instead of scanning --vignettes_dir")
    p.add_argument("--eil", type=Path, required=True, help="NRC Emotion Intensity Lexicon (.txt)")
    p.add_argument("--vad", type=Path, default=None, help="NRC VAD Lexicon (.txt); adds VAD metrics")
    p.add_argument("--vs_baseline", action="store_true",
                   help="Also compare implicit vs none and explicit vs none (unsteered)")
    p.add_argument("--alpha", type=float, default=0.05, help="FDR level for 'significant' counts")
    p.add_argument("--save_text_scores", action="store_true")
    p.add_argument("--output_dir", type=Path, default=Path("results/lexicon_stats"))
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    eil = load_eil(args.eil)
    vad = load_vad(args.vad) if args.vad else None
    groups = {g: m for g, m in METRIC_GROUPS.items() if g != "vad" or vad is not None}
    print(f"Emotion lexicon: {len(eil)} words" + (f"; VAD lexicon: {len(vad)} words" if vad else ""))

    comparisons = [("implicit", "explicit")]
    if args.vs_baseline:
        comparisons += [("implicit", "none"), ("explicit", "none")]

    if args.pair:
        runs = {("pair", ""): {"A": args.pair[0], "B": args.pair[1]}}
        comparisons = [("A", "B")]
    else:
        runs = discover(args.vignettes_dir, args.models)
        if not runs:
            sys.exit(f"No *_steered_vignettes_*.json files found in {args.vignettes_dir}")

    cache, rows, text_rows = {}, [], []
    for (model, tag), paths in sorted(runs.items()):
        for a, b in comparisons:
            if a not in paths or b not in paths:
                continue
            sa = score_file(paths[a], eil, vad, cache)
            sb = score_file(paths[b], eil, vad, cache)
            comp_rows, n_pairs = compare(sa, sb, groups)
            if n_pairs < len(sa) or n_pairs < len(sb):
                print(f"  [note] {model}{tag} {a} vs {b}: {n_pairs} paired vignettes "
                      f"({len(sa)} / {len(sb)} in the files)")
            for r in comp_rows:
                rows.append({"model": model, "scale": tag or "x1", "comparison": f"{a} - {b}", **r})
        if args.save_text_scores:
            for cond, path in paths.items():
                s = score_file(path, eil, vad, cache)
                text_rows.append(s.assign(model=model, scale=tag or "x1", condition=cond,
                                          key=s.index))

    if not rows:
        sys.exit("No comparable condition pairs found (need both files of a comparison).")
    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "lexicon_stats_per_model.csv", index=False)

    # ── Print per-model tables (notebook-style) ──
    pd.set_option("display.width", 200)
    for (model, tag, comp), g in df.groupby(["model", "scale", "comparison"], sort=False):
        print(f"\n=== {model} [{tag}]  {comp}  (n={int(g['n'].max())}) ===")
        show = g[["metric", "mean_a", "mean_b", "d_z", "t_p", "t_q", "wilcoxon_p", "wilcoxon_q"]].copy()
        show["sig"] = np.where(show["t_q"] < args.alpha, "*", "")
        print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    summary = summarize_across_models(df, args.alpha)
    summary.to_csv(args.output_dir / "lexicon_stats_summary.csv", index=False)
    if df["model"].nunique() > 1:
        print("\n=== Across models (per-model mean differences; significance = t-test, "
              f"BH q < {args.alpha}) ===")
        print(summary[["scale", "comparison", "metric", "n_models", "mean_d_z", "n_pos",
                       "n_sig_pos", "n_sig_neg", "across_models_wilcoxon_p"]]
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if text_rows:
        pd.concat(text_rows).to_csv(args.output_dir / "lexicon_text_scores.csv", index=False)
    print(f"\nResults → {args.output_dir}/")


if __name__ == "__main__":
    main()