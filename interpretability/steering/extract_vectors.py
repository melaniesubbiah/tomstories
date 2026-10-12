"""
Build steering vectors by averaging implicit-minus-explicit activation differences over
matched training pairs, at the layer linear_probes.py chose.

Outputs:
  <output_dir>/<model>_layer<N>_vectors.pt   steering vectors, overall and per question type
  <output_dir>/<model>_activations.pt        cached activations, when extracting
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root, for `common`
from common.data import load_best_layer, load_json
from common.hf import clamp_layers, extract_activations, load_model, supports_system_role
from common.models import MODELS, add_model_args, selected_models
from common.prompts import TYPES


# ---------------------------------------------------------------------------
# Steering vector computation
# ---------------------------------------------------------------------------

def compute_vectors(impl_acts, expl_acts, types, granularity):
    diff    = impl_acts - expl_acts
    vectors = {}
    if granularity in ("overall", "both"):
        vectors["overall"] = diff.mean(dim=0)
    if granularity in ("per_type", "both"):
        for typ in TYPES:
            idx = [i for i, t in enumerate(types) if t == typ]
            if idx:
                vectors[typ] = diff[idx].mean(dim=0)
    return vectors

def load_probe_activations_for_steering(model_slug, probe_acts_dir, implicit_data, explicit_data, train_ids):
    """
    Load all-layer activations cached by the probe script and reindex them
    into implicit/explicit tensors ordered to match `train_ids` exactly
    (the same order Phase 2's `train_types` was built from).
    """
    act_path = Path(probe_acts_dir) / f"{model_slug}_probe_acts.pt"
    if not act_path.exists():
        raise FileNotFoundError(f"--probe_acts_path set but not found: {act_path}")

    saved      = torch.load(act_path, map_location="cpu", weights_only=True)
    layer_acts = saved["acts"]   # {layer_idx: tensor (N_total, hidden)}
    sids       = saved["sids"]   # [*implicit_keys, *explicit_keys], probe script's order

    n_impl, n_expl = len(implicit_data), len(explicit_data)
    if len(sids) != n_impl + n_expl:
        raise ValueError(
            f"sid count mismatch: probe cache has {len(sids)} samples but "
            f"implicit_data+explicit_data have {n_impl + n_expl}. Make sure "
            f"--implicit_data/--explicit_data match the files used to build "
            f"{act_path.name}."
        )

    impl_sids, expl_sids = sids[:n_impl], sids[n_impl:]
    if impl_sids != list(implicit_data.keys()) or expl_sids != list(explicit_data.keys()):
        raise ValueError(
            "sid ordering doesn't match implicit_data/explicit_data key order — "
            "the probe cache may have been built from different or reordered JSON files."
        )

    impl_pos = {sid: i for i, sid in enumerate(impl_sids)}
    expl_pos = {sid: i for i, sid in enumerate(expl_sids)}

    missing = [tid for tid in train_ids if tid not in impl_pos or tid not in expl_pos]
    if missing:
        raise KeyError(f"{len(missing)} train_ids not found in probe cache, e.g. {missing[:5]}")

    # reindex into train_ids order; explicit segment is offset by n_impl in the combined tensor
    impl_idx = [impl_pos[tid] for tid in train_ids]
    expl_idx = [n_impl + expl_pos[tid] for tid in train_ids]

    impl_acts, expl_acts = {}, {}
    for li, acts in layer_acts.items():
        impl_acts[li] = acts[impl_idx]
        expl_acts[li] = acts[expl_idx]

    return impl_acts, expl_acts, sorted(layer_acts.keys())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    parser.add_argument("--implicit_data", default="tomstories_vignettes/implicit_vignettes.json")
    parser.add_argument("--explicit_data", default="tomstories_vignettes/explicit_vignettes.json")
    parser.add_argument("--layers",        nargs="+", type=int, default=None,
                        help="Override the layers to extract from (default: the layer "
                             "linear_probes.py saved for the model)")
    parser.add_argument("--probe_results_dir", type=Path, default=Path("results/probes"),
                        help="Directory holding <model>_best_layers.json from "
                             "linear_probes.py, which decides the steering layer")
    parser.add_argument("--granularity",   choices=["overall", "per_type", "both"],
                        default="both")
    parser.add_argument("--batch_size",    type=int, default=4)
    parser.add_argument("--output_dir",    default="results/steering")
    parser.add_argument("--probe_acts_path", type=Path, default=None,
                        help="Load activations cached by the probe script instead of "
                             "re-running the model")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ToMStories ──────────────────────────────────────────────────────────
    implicit_data = load_json(args.implicit_data)
    explicit_data = load_json(args.explicit_data)

    common_ids  = sorted(set(implicit_data) & set(explicit_data))
    train_ids   = [k for k in common_ids if implicit_data[k]["split"] == "train"]
    print(f"Matched pairs: {len(common_ids)}  (train: {len(train_ids)})")

    impl_train  = [implicit_data[k] for k in train_ids]
    expl_train  = [explicit_data[k] for k in train_ids]
    train_types = [implicit_data[k]["type"] for k in train_ids]

    model_names = selected_models(args, parser)

    for modelname in model_names:
        model_kwargs = MODELS[modelname]
        model_slug = modelname.split("/")[-1]

        # Layers come from linear_probes.py unless --layers overrides them, so the
        # vector files line up with the layer tune_steering.py will ask for
        if args.layers is not None:
            requested = sorted(set(args.layers))
        else:
            requested = [load_best_layer(args.probe_results_dir, model_slug)["layer"]]
            print(f"\n{model_slug}: extracting at layer {requested[0]} "
                  f"(from {model_slug}_best_layers.json)")

        if args.probe_acts_path is not None:
            print(f"Loading probe-script activations from {args.probe_acts_path}")
            impl_acts, expl_acts, cached_layers = load_probe_activations_for_steering(
                model_slug, args.probe_acts_path, implicit_data, explicit_data, train_ids
            )
            layers_to_use = [li for li in requested if li in cached_layers]
            if not layers_to_use:
                raise ValueError(f"none of the layers {requested} are in the probe "
                                 f"cache (it has {len(cached_layers)} layers)")
            print(f"  Loaded {len(cached_layers)} layers, {len(train_ids)} matched "
                  f"train samples; using {layers_to_use}")
        else:
            # ── Load model ─────────────────────────────────────────────────────────
            print(f"Loading {modelname} …")
            model, tokenizer = load_model(model_kwargs)
            sys_ok = supports_system_role(tokenizer)
            if not sys_ok:
                print("  Note: model does not support system role — merging into user message")

            device     = next(model.parameters()).device
            num_layers = model.config.num_hidden_layers
            layers_to_use = clamp_layers(requested, num_layers)
            print(f"Layers: {layers_to_use}  (model has {num_layers} layers, hidden={model.config.hidden_size})")

            # ── Phase 1: Activations ───────────────────────────────────────────────
            act_path = out_dir / f"{model_slug}_activations.pt"
            print("\n── Phase 1: Extracting activations ──")
            print("  Implicit …")
            impl_acts = extract_activations(
                model, tokenizer, impl_train, layers_to_use, args.batch_size, device,
                system_supported=sys_ok,
            )
            print("  Explicit …")
            expl_acts = extract_activations(
                model, tokenizer, expl_train, layers_to_use, args.batch_size, device,
                system_supported=sys_ok,
            )
            torch.save({"implicit": impl_acts, "explicit": expl_acts}, act_path)
            print(f"  Saved → {act_path}")

        # ── Phase 2: Steering vectors ──────────────────────────────────────────
        print("\n── Phase 2: Computing steering vectors ──")
        all_vectors = {}
        for li in layers_to_use:
            vecs = compute_vectors(
                impl_acts[li], expl_acts[li], train_types, args.granularity
            )
            all_vectors[li] = vecs
            vec_path = out_dir / f"{model_slug}_layer{li}_vectors.pt"
            torch.save(vecs, vec_path)
            for key, v in vecs.items():
                print(f"  layer={li}  {key:<12}  norm={v.norm():.3f}")


if __name__ == "__main__":
    main()
