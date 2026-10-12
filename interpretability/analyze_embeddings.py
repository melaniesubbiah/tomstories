"""
Compare implicit and explicit representations at the layer linear_probes.py chose: per
(type, style) centroids, the implicit-minus-explicit direction, and how parallel those
directions are across question types.

Outputs:
  <output_dir>/<model>_layer<N>_factorization.pt   centroids and the implicit direction
  <output_dir>/<model>_summary.json                per-layer mean pairwise cosine and norm CV
  <output_dir>/<model>_factorization_acts_all.pt   cached activations, when extracting
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, for `common`
from common.data import load_best_layer, load_json
from common.hf import clamp_layers, extract_activations, load_model, supports_system_role
from common.models import MODELS, add_model_args, selected_models
from common.prompts import TYPES


def compute_centroids(acts, labels):
    centroids = {}
    for typ in TYPES:
        for style in ("implicit", "explicit"):
            idx = [i for i, (t, s) in enumerate(labels) if t == typ and s == style]
            if idx:
                centroids[(typ, style)] = acts[idx].mean(dim=0)
    return centroids


def load_probe_activations(modelname, probe_acts_path, implicit_data, explicit_data):
    saved = torch.load(os.path.join(probe_acts_path, f"{modelname}_probe_acts.pt"),
                       map_location="cpu", weights_only=True)
    layer_acts = saved["acts"]
    sids       = saved["sids"]

    n_impl, n_expl = len(implicit_data), len(explicit_data)
    if len(sids) != n_impl + n_expl:
        raise ValueError(
            f"sid count mismatch: probe cache has {len(sids)} samples but "
            f"implicit_data+explicit_data have {n_impl + n_expl}. Make sure "
            f"--implicit_data/--explicit_data point at the same files used "
            f"when the probe script built {probe_acts_path.name}."
        )

    impl_sids, expl_sids = sids[:n_impl], sids[n_impl:]
    # fail loudly instead of silently mislabeling on a key collision
    if impl_sids != list(implicit_data.keys()) or expl_sids != list(explicit_data.keys()):
        raise ValueError(
            "sid ordering doesn't match implicit_data/explicit_data key order — "
            "the probe cache may have been built from different or reordered JSON files."
        )

    labels = [(implicit_data[sid]["type"], "implicit") for sid in impl_sids] + \
             [(explicit_data[sid]["type"], "explicit") for sid in expl_sids]

    return layer_acts, labels, sids

def summary_statistics(centroids, layer_idx, out_dir, verbose=True):
    types_present = [t for t in TYPES
                     if (t, "implicit") in centroids and (t, "explicit") in centroids]

    diffs   = {t: (centroids[(t, "implicit")] - centroids[(t, "explicit")]).numpy()
               for t in types_present}
    norms = {t: float(np.linalg.norm(diffs[t])) for t in types_present}
    v_impl  = np.mean(list(diffs.values()), axis=0)       # (hidden,) — saved for steering

    # Pairwise cosine between the per-type implicit-minus-explicit directions
    pair_cos = {}
    for i, a in enumerate(types_present):
        for b in types_present[i + 1:]:
            pair_cos[(a, b)] = float(diffs[a] @ diffs[b] /
                                     (norms[a] * norms[b] + 1e-9))

    if verbose:
        print(f"\n[Layer {layer_idx}] Summary statistics")
        print(f"  {'type':<12} {'||d_t||':>10}")
        for t in types_present:
            print(f"  {t:<12} {norms[t]:>10.4f}")
        print(f"  {'pair':<24} {'cos(d_t, d_t_prime)':>20}")
        for (a, b), c in pair_cos.items():
            print(f"  {a + ' / ' + b:<24} {c:>20.4f}")

    # Type-averaged summary stats for cross-model table
    summary = {
        "mean_pairwise_cos": (float(np.mean(list(pair_cos.values())))
                              if pair_cos else float("nan")),
        "norm_cv":           float(np.std( [norms[t]   for t in types_present]) /
                                   (np.mean([norms[t]  for t in types_present]) + 1e-9)),
    }

    return v_impl, summary


def recompute_summaries_from_centroids(out_dir, model_names=None):
    """
    Rebuild <model>_summary.json from the per-layer <model>_layer<N>_factorization.pt
    files already in out_dir. The summary statistics only need the saved centroids,
    so no activations or models are loaded.
    """
    pat = re.compile(r"^(?P<model>.+)_layer(?P<layer>\d+)_factorization\.pt$")
    by_model = {}
    for f in out_dir.glob("*_layer*_factorization.pt"):
        m = pat.match(f.name)
        if m:
            by_model.setdefault(m["model"], {})[int(m["layer"])] = f

    if model_names:
        missing = [m for m in model_names if m not in by_model]
        for m in missing:
            print(f"{m}: no *_factorization.pt files in {out_dir}, skipping")
        model_names = [m for m in model_names if m in by_model]
    else:
        model_names = sorted(by_model)

    for model_slug in model_names:
        layer_summaries = {}
        for li, path in sorted(by_model[model_slug].items()):
            saved = torch.load(path, map_location="cpu", weights_only=True)
            # keys were saved as f"{type}_{style}"
            centroids = {tuple(k.rsplit("_", 1)): v for k, v in saved["centroids"].items()}
            _, layer_summaries[li] = summary_statistics(centroids, li, out_dir, verbose=False)

        summary_path = out_dir / f"{model_slug}_summary.json"
        with open(summary_path, "w") as f:
            json.dump({"model": model_slug, "layers": layer_summaries}, f, indent=2)
        print(f"{model_slug}: {len(layer_summaries)} layers → {summary_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(parser)
    parser.add_argument("--implicit_data",   default="tomstories_vignettes/implicit_vignettes.json")
    parser.add_argument("--explicit_data",   default="tomstories_vignettes/explicit_vignettes.json")
    parser.add_argument("--layers",          nargs="+", type=int, default=None,
                        help="Override the layers to analyse (default: the layer "
                             "linear_probes.py saved for the model)")
    parser.add_argument("--batch_size",      type=int, default=4)
    parser.add_argument("--output_dir",      default="results/embeddings")
    parser.add_argument("--skip_extraction", action="store_true",
                        help="Load saved activations instead of re-extracting")
    parser.add_argument("--recompute_summaries", action="store_true",
                        help="Rebuild *_summary.json from the saved *_layer<N>_factorization.pt "
                             "files in output_dir (no activations/models needed), then exit. "
                             "Uses --models/--all if given, otherwise every model found")
    parser.add_argument("--print_summary",   action="store_true",
                        help="Read all saved *_summary.json files in output_dir and print "
                             "the cross-model summary, then exit")
    parser.add_argument("--probe_acts_path", type=Path, default=None,
                        help="Load activations cached by the probe script instead of "
                             "re-running the model")
    parser.add_argument("--probe_results_dir", type=Path, default=Path("results/probes"),
                        help="Directory holding <model>_best_layers.json from "
                             "linear_probes.py, which decides each model's layer")
    args = parser.parse_args()

    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.recompute_summaries:
        names = list(MODELS) if args.all else args.models
        recompute_summaries_from_centroids(out_dir, names)
        return

    if args.print_summary:
        _print_cross_model_table(out_dir, args.probe_results_dir)
        return

    model_names = selected_models(args, parser, extra_hint=", or --print_summary")

    # ── Load tomstories_vignettes ──────────────────────────────────────────────────────────
    implicit_data = load_json(args.implicit_data)
    explicit_data = load_json(args.explicit_data)

    impl_samples = [(k, v, (v["type"], "implicit")) for k, v in implicit_data.items()]
    expl_samples = [(k, v, (v["type"], "explicit")) for k, v in explicit_data.items()]
    all_pairs    = impl_samples + expl_samples
    sids         = [k for k, _, _ in all_pairs]
    samples_flat = [v for _, v, _ in all_pairs]
    labels       = [l for _, _, l in all_pairs]

    print(f"Implicit: {len(impl_samples)}  Explicit: {len(expl_samples)}")
    for typ in TYPES:
        ni = sum(1 for _, _, (t, s) in all_pairs if t == typ and s == "implicit")
        ne = sum(1 for _, _, (t, s) in all_pairs if t == typ and s == "explicit")
        print(f"  {typ:<12}  impl={ni}  expl={ne}")

    # ── Extract / load activations ─────────────────────────────────────────
    for modelname in model_names:
        model_kwargs = MODELS[modelname]
        model_slug = modelname.split("/")[-1]
        act_path = out_dir / f"{model_slug}_factorization_acts_all.pt"

        # Layers come from linear_probes.py unless --layers overrides them
        if args.layers is not None:
            requested = sorted(set(args.layers))
        else:
            requested = [load_best_layer(args.probe_results_dir, model_slug)["layer"]]
            print(f"\n{model_slug}: analysing layer {requested[0]} "
                  f"(from {model_slug}_best_layers.json)")

        if args.probe_acts_path is not None:
            probe_path = args.probe_acts_path
            if not probe_path.exists():
                raise FileNotFoundError(f"--probe_acts_path set but not found: {probe_path}")
            print(f"Loading probe-script activations from {probe_path}")
            layer_acts, labels, sids = load_probe_activations(
                modelname, probe_path, implicit_data, explicit_data
            )
            layers_to_use = [li for li in requested if li in layer_acts]
            print(f"  Loaded {len(layer_acts)} layers, {len(sids)} samples; "
                  f"analysing {len(layers_to_use)}")
        elif args.skip_extraction:
            if not act_path.exists():
                raise FileNotFoundError(f"--skip_extraction set but cache not found: {act_path}")
            print(f"Loading activations from {act_path}")
            saved      = torch.load(act_path, map_location="cpu", weights_only=True)
            layer_acts = saved["acts"]
            labels     = saved["labels"]
            layers_to_use = [li for li in requested if li in layer_acts]
        else:
            print(f"Loading {modelname} …")
            model, tokenizer = load_model(model_kwargs)
            sys_ok = supports_system_role(tokenizer)
            if not sys_ok:
                print("  Note: model does not support system role — merging into user message")

            device     = next(model.parameters()).device
            num_layers = model.config.num_hidden_layers
            layers_to_use = clamp_layers(requested, num_layers)
            print(f"Layers: {layers_to_use}  (model has {num_layers} layers, "
                  f"hidden={model.config.hidden_size})")

            print("\nExtracting activations …")
            layer_acts = extract_activations(
                model, tokenizer, samples_flat, layers_to_use, args.batch_size, device,
                system_supported=sys_ok,
            )
            torch.save({"acts": layer_acts, "labels": labels, "sids": sids}, act_path)
            print(f"  Saved → {act_path}")

        # ── Per-layer analysis ─────────────────────────────────────────────────
        layer_summaries = {}   # {layer_idx: summary_dict}
        for li in layers_to_use:
            print(f"\n{'='*60}\nLayer {li}\n{'='*60}")
            acts      = layer_acts[li]                            # (N_total, hidden)
            centroids = compute_centroids(acts, labels)

            v_implicit, summary = summary_statistics(centroids, li, out_dir)
            layer_summaries[li] = summary

            # Save vectors for downstream use
            save_path = out_dir / f"{model_slug}_layer{li}_factorization.pt"
            torch.save({
                "centroids":  {f"{t}_{s}": v for (t, s), v in centroids.items()},
                "v_implicit": torch.tensor(v_implicit, dtype=torch.float32),
            }, save_path)
            print(f"  Saved vectors → {save_path.name}")

        print(f"\nAll outputs saved to {out_dir}/")

        # ── Save and print summary table ───────────────────────────────────────
        summary_path = out_dir / f"{model_slug}_summary.json"
        with open(summary_path, "w") as f:
            json.dump({"model": modelname, "layers": layer_summaries}, f, indent=2)
        print(f"Summary stats → {summary_path}")

        _print_run_table(model_slug, layer_summaries)


def _print_run_table(model_slug, layer_summaries):
    """Terminal table for one model run (one row per analysed layer)."""
    header = f"{'Layer':>6}  {'mean cos(d_t, d_t2)':>20}  {'CV(||d_t||)':>12}"
    print(f"\n{'=' * len(header)}")
    print(f"  Factorization summary: {model_slug}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for li, s in sorted(layer_summaries.items()):
        print(f"{li:>6}  {s['mean_pairwise_cos']:>20.3f}  {s['norm_cv']:>12.3f}")
    print("=" * len(header))


def _fmt(x, nd=3):
    return "--" if x is None else f"{x:.{nd}f}"


def _print_cross_model_table(out_dir, probe_results_dir):
    """
    Read all *_summary.json files in out_dir and print one row per model at the
    layer linear_probes.py chose for it, plus an average row for
    mean_pairwise_cos / norm_cv.
    """
    import glob
    paths = sorted(glob.glob(str(out_dir / "*_summary.json")))
    if not paths:
        print(f"No *_summary.json files found in {out_dir}")
        return

    rows, notes = [], []
    cos_vals, cv_vals = [], []
    for path in paths:
        data = load_json(path)
        model = data["model"].split("/")[-1]
        ls = {int(li): s for li, s in data["layers"].items()}
        if not ls:
            continue

        try:
            selected = load_best_layer(probe_results_dir, model)
        except (FileNotFoundError, ValueError) as e:
            notes.append(f"{model}: skipped, {str(e).splitlines()[0]}")
            continue
        li = selected["layer"]
        if li not in ls:
            notes.append(f"{model}: skipped, chosen layer {li} not in factorization "
                         f"summary; rerun with --probe_acts_path to cover all layers")
            continue

        s = ls[li]
        if "mean_pairwise_cos" not in s:
            notes.append(f"{model}: skipped, summary has no mean_pairwise_cos "
                         f"(written by an older version; rerun the analysis)")
            continue
        cos_vals.append(s["mean_pairwise_cos"])
        cv_vals.append(s["norm_cv"])
        rows.append([model, str(li), _fmt(selected["selection"].get("acc")),
                     _fmt(s["mean_pairwise_cos"]), _fmt(s["norm_cv"])])

    header = (f"{'Model':<20}  {'Layer':>5}  {'ProbeAcc':>8}  "
              f"{'mean cos(d_t, d_t2)':>20}  {'CV(||d_t||)':>12}")
    print("\n" + "=" * len(header))
    print("  Cross-model factorization at each model's selected layer")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for model, li, acc, cos, cv in rows:
        print(f"{model:<20}  {li:>5}  {acc:>8}  {cos:>20}  {cv:>12}")
    if rows:
        print("-" * len(header))
        print(f"{'Average':<20}  {'':>5}  {'':>8}  "
              f"{np.mean(cos_vals):>20.3f}  {np.mean(cv_vals):>12.3f}")
    print("=" * len(header))
    if notes:
        print("\nNotes:")
        for n in notes:
            print(f"  {n}")


if __name__ == "__main__":
    main()