"""
Fit logistic-regression probes on last-token activations at every layer and report held-out
accuracy, over all samples and over implicit/explicit samples separately. Also chooses the
layer the other interpretability scripts work at, so run this one first.

Outputs:
  <output_dir>/<model>_probe_results.json   per-layer accuracy per probe and question type
  <output_dir>/<model>_best_layers.json     the chosen layer and each probe's peak layer
  <output_dir>/<model>_probe_acts.pt        cached all-layer activations
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, for `common`
from common.data import load_json, save_json
from common.hf import (extract_activations, free_cuda, load_model, supports_system_role)
from common.models import MODELS, add_model_args, selected_models
from common.prompts import TYPES

# The probes this script always fits, in table/report order
BEST_KEYS = [
    ("choice",          "choice index (0–3)"),
    ("choice_implicit", "choice — implicit"),
    ("choice_explicit", "choice — explicit"),
]
PROBE_KEYS = [k for k, _ in BEST_KEYS]

# Per-type val accuracies derived from each probe (only the style probes get them)
TYPE_KEYS_BY_PROBE = {
    "choice":          [],
    "choice_implicit": [f"choice_implicit_{t}" for t in TYPES],
    "choice_explicit": [f"choice_explicit_{t}" for t in TYPES],
}
ALL_KEYS = PROBE_KEYS + [tk for k in PROBE_KEYS for tk in TYPE_KEYS_BY_PROBE[k]]


def fit_probe(X_train, y_train, X_val, y_val):
    """Returns (val accuracy, val predictions); predictions are None if no data."""
    if len(X_train) == 0 or len(X_val) == 0:
        return float("nan"), None
    if len(set(y_train)) < 2:
        preds = np.full(len(y_val), list(set(y_train))[0])
        return float(np.mean(preds == y_val)), preds

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train)
    X_va   = scaler.transform(X_val)

    n_classes = len(set(y_train))
    solver = "liblinear" if n_classes == 2 else "lbfgs"
    clf = LogisticRegression(
        max_iter=500, C=0.1, random_state=42,
        solver=solver,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(X_tr, y_train)
    preds = clf.predict(X_va)
    return float(np.mean(preds == y_val)), preds


def chance_accuracy(labels):
    counts = np.bincount(labels)
    return counts.max() / counts.sum()


# ── Layer selection ───────────────────────────────────────────────────────────
# Owned by this script: the chosen layer is saved to <model>_best_layers.json and
# every downstream script reads it from there.

def best_layer(accs, layers):
    """Earliest layer achieving the max accuracy, ignoring NaNs; None if unavailable."""
    if not accs:
        return None
    arr = np.asarray(accs, dtype=float)
    if np.all(np.isnan(arr)):
        return None
    i = int(np.nanargmax(arr))   # first occurrence on ties -> earliest layer
    return {"layer": int(layers[i]), "acc": float(arr[i])}


def _smooth(arr, window):
    """Centered moving average that shrinks at the edges and ignores NaNs."""
    if window <= 1:
        return arr.copy()
    half = window // 2
    out  = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        seg = arr[max(0, i - half): i + half + 1]
        if not np.all(np.isnan(seg)):
            out[i] = np.nanmean(seg)
    return out


def plateau_layer(accs, layers, frac=0.9, window=3, hold=2):
    """
    Find where an S-shaped probe-accuracy curve has finished rising.

    1. Smooth the curve (centered moving average of `window` layers).
    2. top  = max of the smoothed curve (the plateau level);
       base = min of the smoothed curve up to that peak.
    3. Return the earliest layer (at or before the peak) where the smoothed
       curve reaches base + frac * (top - base) and stays at or above that
       for `hold` consecutive layers.

    Small late increases barely move the threshold, so they don't drag the
    choice later; mid-rise layers fall short of the threshold, so they aren't
    picked either. Returns None if the curve is missing or too short.
    """
    if not accs or len(accs) < 3:
        return None
    raw = np.asarray(accs, dtype=float)
    sm  = _smooth(raw, window)
    if np.all(np.isnan(sm)):
        return None

    peak_i = int(np.nanargmax(sm))
    top    = float(sm[peak_i])
    base   = float(np.nanmin(sm[: peak_i + 1]))
    thresh = base + frac * (top - base)

    chosen = peak_i   # fallback: the peak always clears the threshold
    for i in range(peak_i + 1):
        seg = sm[i: i + hold]
        seg = seg[~np.isnan(seg)]
        if len(seg) and np.all(seg >= thresh):
            chosen = i
            break

    return {
        "layer":      int(layers[chosen]),
        "acc":        float(raw[chosen]),
        "acc_smooth": float(sm[chosen]),
        "base":       base,
        "top":        top,
        "thresh":     thresh,
        "peak_layer": int(layers[peak_i]),
    }


def save_best_layers(path, modelname, results, layers, args):
    """
    Write the layer every downstream script should use, plus how it was picked.

    "layer" is the plateau onset of the --probe_key curve: the layer where that
    curve has finished rising. "peak" additionally records the earliest
    maximum-accuracy layer for each probe and question type, so a different
    selection rule can be applied later without refitting anything.
    """
    plat = plateau_layer(results.get(args.probe_key), layers, args.plateau_frac,
                         args.smooth_window, args.plateau_hold)
    if plat is None:
        raise ValueError(f"no usable '{args.probe_key}' curve to pick a layer from; "
                         f"fit that probe first (got {len(layers)} layers)")

    type_keys = [tk for k in PROBE_KEYS for tk in TYPE_KEYS_BY_PROBE[k]]
    best = {k: best_layer(results.get(k), layers) for k in PROBE_KEYS + type_keys}

    record = {
        "model":     modelname,
        "layer":     plat["layer"],
        "n_layers":  len(layers),
        "probe_key": args.probe_key,
        "selection": {
            "rule":          "plateau",
            "plateau_frac":  args.plateau_frac,
            "smooth_window": args.smooth_window,
            "plateau_hold":  args.plateau_hold,
            **{k: plat[k] for k in ("acc", "acc_smooth", "base", "top", "thresh",
                                    "peak_layer")},
        },
        "peak": best,
    }
    save_json(path, record, indent=2)
    return record


def print_best_layers_table(best_by_model):
    """Terminal table: one row per model, chosen layer and each probe's peak."""
    labels = [label for _, label in BEST_KEYS]
    header = (f"{'Model':<20}  {'Layer':>7}  {'Acc':>6}"
              + "".join(f"  {lab[:20]:>22}" for lab in labels))
    print("\n" + "=" * len(header))
    print("  Chosen layer (probe-accuracy plateau) and peak accuracy per probe")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for model, record in best_by_model.items():
        cells = []
        for key, _ in BEST_KEYS:
            b = record["peak"].get(key)
            cells.append(f"{b['acc']:.3f} @ L{b['layer']}" if b else "--")
        print(f"{model:<20}  {record['layer']:>7}  {record['selection']['acc']:>6.3f}"
              + "".join(f"  {c:>22}" for c in cells))
    print("=" * len(header))
    print("Layer = where the probe curve finishes rising; downstream scripts read this "
          "from <model>_best_layers.json.")


def cached_keys(existing, requested):
    """
    Requested settings whose results are already in the loaded results JSON:
    present, a list, and one value per saved layer.
    """
    layers = existing.get("layers")
    if not isinstance(layers, list):
        return set()
    return {k for k in requested
            if isinstance(existing.get(k), list) and len(existing[k]) == len(layers)}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(parser)
    parser.add_argument("--implicit_data",   default="tomstories_vignettes/implicit_vignettes.json")
    parser.add_argument("--explicit_data",   default="tomstories_vignettes/explicit_vignettes.json")
    parser.add_argument("--batch_size",      type=int, default=4)
    parser.add_argument("--output_dir",      default="results/probes")
    parser.add_argument("--skip_extraction", action="store_true",
                        help="Load cached activations instead of re-running the model")
    parser.add_argument("--skip_fitting", action="store_true",
                        help="Only extract/cache activations; skip probe fitting")
    parser.add_argument("--skip_probes", action="store_true",
                        help="Load saved probe results JSON and skip extraction + fitting")
    parser.add_argument("--overwrite", action="store_true",
                        help="Refit all three probes even if their results are "
                             "already in the results JSON")
    parser.add_argument("--probe_key", default="choice", choices=PROBE_KEYS,
                        help="Probe curve used to pick the layer saved for downstream scripts")
    parser.add_argument("--plateau_frac", type=float, default=0.9,
                        help="Fraction of the total probe-accuracy rise a layer must reach "
                             "to count as the plateau onset")
    parser.add_argument("--smooth_window", type=int, default=3,
                        help="Moving-average window (in layers) applied before plateau detection")
    parser.add_argument("--plateau_hold", type=int, default=2,
                        help="Layers the curve must stay above the threshold to count")
    args = parser.parse_args()

    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load datasets ──────────────────────────────────────────────────────
    implicit_data = load_json(args.implicit_data)
    explicit_data = load_json(args.explicit_data)

    # Tag each sample with style, then flatten
    records = []
    sids    = []
    for sid, sample in implicit_data.items():
        records.append({**sample, "style": "implicit"})
        sids.append(sid)
    for sid, sample in explicit_data.items():
        records.append({**sample, "style": "explicit"})
        sids.append(sid)

    splits     = np.array([r["split"]  for r in records])
    styles_raw = np.array([r["style"]  for r in records])

    # Choice probe: 0-based index of the correct answer in answer_options (0–3).
    y_choice = np.array([r["answer_options"].index(r["answer"]) for r in records])
    types_raw = np.array([r["type"] for r in records])

    train_mask = splits == "train"
    val_mask   = splits == "val"
    impl_mask  = styles_raw == "implicit"
    expl_mask  = styles_raw == "explicit"

    # (train mask, val mask) for each probe setting
    probe_masks = {
        "choice":          (train_mask,             val_mask),
        "choice_implicit": (train_mask & impl_mask, val_mask & impl_mask),
        "choice_explicit": (train_mask & expl_mask, val_mask & expl_mask),
    }

    requested = ALL_KEYS

    # Val-set size behind every curve (saved alongside the results)
    n_val = {}
    for k in PROBE_KEYS:
        va = probe_masks[k][1]
        n_val[k] = int(va.sum())
        for t in TYPES:
            if f"{k}_{t}" in TYPE_KEYS_BY_PROBE[k]:
                n_val[f"{k}_{t}"] = int((va & (types_raw == t)).sum())

    print(f"Train: {train_mask.sum()}  Val: {val_mask.sum()}")

    model_names = selected_models(args, parser)

    best_by_model = {}
    for modelname in model_names:
        model_kwargs = MODELS[modelname]
        model_slug = modelname.split("/")[-1]
        json_path = out_dir / f"{model_slug}_probe_results.json"
        act_path  = out_dir / f"{model_slug}_probe_acts.pt"
        chance_choice = 0.25

        existing = load_json(json_path) if json_path.exists() else {}
        # --overwrite refits the requested settings but keeps any others in the JSON
        have = set() if args.overwrite else cached_keys(existing, requested)

        if args.skip_probes:
            # ── Load saved results only ────────────────────────────────────────
            if not json_path.exists():
                raise FileNotFoundError(f"--skip_probes set but results not found: {json_path}")
            print(f"\nLoading probe results from {json_path}")
            results = load_json(json_path)
            layers  = results["layers"]

        elif set(requested) <= have and not args.skip_fitting:
            # ── Everything requested is already cached: no model, no fitting ───
            print(f"\n{modelname}: all probe results (incl. per-type) already in "
                  f"{json_path.name}; skipping extraction and fitting")
            results = existing
            layers  = existing["layers"]

        else:
            # ── Load or extract activations ────────────────────────────────────
            if args.skip_extraction:
                if not act_path.exists():
                    raise FileNotFoundError(f"--skip_extraction set but cache not found: {act_path}")
                print(f"\nLoading activations from {act_path}")
                saved      = torch.load(act_path, map_location="cpu", weights_only=True)
                layer_acts = saved["acts"]
                print(f"  Loaded {len(layer_acts)} layers")
            else:
                print(f"\nLoading {modelname} …")
                model, tokenizer = load_model(model_kwargs)
                sys_ok = supports_system_role(tokenizer)
                if not sys_ok:
                    print("  Note: model does not support system role — merging into user message")

                device = next(model.parameters()).device
                print(f"  {model.config.num_hidden_layers} layers, hidden={model.config.hidden_size}")

                print("\nExtracting activations at all layers …")
                layer_acts = extract_activations(
                    model, tokenizer, records, range(model.config.num_hidden_layers),
                    args.batch_size, device, system_supported=sys_ok,
                    desc="extracting activations",
                )
                torch.save({"acts": layer_acts, "sids": sids}, act_path)
                print(f"  Saved → {act_path}")

                del model
                free_cuda()

            layers = sorted(layer_acts.keys())

            if args.skip_fitting:
                continue

            # Cached results only count if they were computed on the same layers
            if existing and existing.get("layers") != layers:
                print(f"  Cached results in {json_path.name} cover different layers; ignoring them")
                existing, have = {}, set()

            # A probe is refit if its own curve or any of its per-type curves is missing
            to_fit = [k for k in PROBE_KEYS
                      if not ({k, *TYPE_KEYS_BY_PROBE[k]} <= have)]
            for k in PROBE_KEYS:
                if k not in to_fit:
                    print(f"  '{k}': already in {json_path.name}, reusing")

            # ── Fit only the missing probes at every layer ─────────────────────
            new = {}
            for k in to_fit:
                new[k] = []
                for tk in TYPE_KEYS_BY_PROBE[k]:
                    new[tk] = []
            if to_fit:
                n_type = sum(len(TYPE_KEYS_BY_PROBE[k]) for k in to_fit)
                extra = f" (+ {n_type} per-type val curves)" if n_type else ""
                print(f"\nFitting {', '.join(to_fit)}{extra} across {len(layers)} layers …")
                for li in tqdm(layers, desc="probing layers"):
                    acts = layer_acts[li].numpy()
                    for k in to_fit:
                        tr, va = probe_masks[k]
                        acc, preds = fit_probe(acts[tr], y_choice[tr], acts[va], y_choice[va])
                        new[k].append(acc)
                        # Per-type accuracy of the same probe on the val items of each type
                        val_types, val_y = types_raw[va], y_choice[va]
                        for t in TYPES:
                            tk = f"{k}_{t}"
                            if tk not in new:
                                continue
                            m = val_types == t
                            new[tk].append(float(np.mean(preds[m] == val_y[m]))
                                           if preds is not None and m.any() else float("nan"))

            # Merge with whatever was already saved (keeps anything not refit this run)
            results = {**existing, "layers": layers, **new, "n_val": n_val}
            if to_fit:
                with open(json_path, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"\nRaw results → {json_path}")

        if args.skip_fitting:
            continue

        # ── Layer selection + summary ──────────────────────────────────────────
        best_path = out_dir / f"{model_slug}_best_layers.json"
        record    = save_best_layers(best_path, modelname, results, layers, args)
        best      = record["peak"]
        sel       = record["selection"]

        print(f"\nChosen layer: {record['layer']} of {record['n_layers']} "
              f"('{args.probe_key}' curve rises {sel['base']:.3f} → {sel['top']:.3f}, "
              f"threshold {sel['thresh']:.3f}, max at layer {sel['peak_layer']})")

        print("\nPeak probe accuracies:")
        for key, label in BEST_KEYS:
            if best[key] is not None:
                print(f"  {label:<32}  peak={best[key]['acc']:.3f} at layer "
                      f"{best[key]['layer']}  (chance={chance_choice:.3f})")

        # Per-type peaks for the implicit / explicit probes
        type_keys   = [tk for k in PROBE_KEYS for tk in TYPE_KEYS_BY_PROBE[k]]
        n_val_saved = results.get("n_val", {})
        missing     = [tk for tk in type_keys if results.get(tk) is None]
        if missing:
            print(f"  Per-type curves missing from {json_path.name}: {', '.join(missing)}")
        print("\nPer-type peak accuracies (implicit / explicit probes):")
        print(f"  {'type':<12} {'implicit':>22} {'explicit':>22}")
        for t in TYPES:
            cells = []
            for style in ("implicit", "explicit"):
                b = best.get(f"choice_{style}_{t}")
                n = n_val_saved.get(f"choice_{style}_{t}")
                n_str = f", n={n}" if n is not None else ""
                cells.append(f"{b['acc']:.3f} @ L{b['layer']}{n_str}" if b else "--")
            print(f"  {t:<12} {cells[0]:>22} {cells[1]:>22}")

        print(f"\nBest layers → {best_path}")
        best_by_model[model_slug] = record

    if best_by_model:
        print_best_layers_table(best_by_model)


if __name__ == "__main__":
    main()