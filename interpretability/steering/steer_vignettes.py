"""
Generate vignettes while steering toward implicit or explicit style, using the layer and
coefficient tuned by tune_steering.py, alongside an unsteered baseline.

Outputs:
  <output_dir>/<model>_steered_vignettes_<condition>[_x<scale>].json   generations and settings
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root, for `common`
from common.data import load_json
from common.hf import (RowSteerer, format_chat, free_cuda, load_model, load_unit_vector,
                       supports_system_role)
from common.models import MODELS, add_model_args, selected_models
from common.prompts import TYPES, VIGNETTE_PROMPT_TEMPLATE

SYSTEM_PROMPTS = {
    "emotion": (
        "You are a skilled fiction writer. You are performing writing exercises that ask "
        "you to complete short scenarios given a setting, character relationship, and emotion. "
        "You need to write a short narrative in which one of the characters is feeling this emotion, "
        "while adhering to the setting and character relationship provided. "
    ),
    "wound": (
        "You are a skilled fiction writer. You are performing writing exercises that ask "
        "you to complete short scenarios given a setting, character relationship, and emotional wound. "
        "You need to write a short narrative in which one of the characters has experienced the emotional wound in their past, "
        "while adhering to the setting and character relationship provided. "
    ),
    "fear": (
        "You are a skilled fiction writer. You are performing writing exercises that ask "
        "you to complete short scenarios given a setting, character relationship, and fear. "
        "You need to write a short narrative in which one of the characters has this core fear, "
        "while adhering to the setting and character relationship provided. "
    ),
    "motivation": (
        "You are a skilled fiction writer. You are performing writing exercises that ask "
        "you to complete short scenarios given a setting, character relationship, and motivation. "
        "You need to write a short narrative in which one of the characters has this core emotion, "
        "while adhering to the setting and character relationship provided. "
    ),
}

PROMPT_TEMPLATE = (VIGNETTE_PROMPT_TEMPLATE +
                   "\n\nWrite a short narrative based on this scenario.")

CONDITION_SIGN = {"implicit": +1, "explicit": -1, "none": 0}


# ---------------------------------------------------------------------------
# Tuned settings
# ---------------------------------------------------------------------------

def resolve_steering(entry, conditions, source, coeff_scale):
    """
    From one model's entry in steer_coherence_summary.json, return
    {condition: signed coefficient on the unit vector}.
    source = "shared" (same magnitude both ways) or "direction" (each
    direction's own tuned coefficient).
    """
    if "error" in entry:
        raise RuntimeError(f"tuning failed for this model: {entry['error']}")
    coeffs = {}
    for cond in conditions:
        sign = CONDITION_SIGN[cond]
        if sign == 0:
            coeffs[cond] = 0.0
            continue
        if source == "shared":
            sh = entry.get("shared")
            if not sh:
                raise RuntimeError("no shared coefficient in the summary (tune with --signs 1 -1), "
                                   "or use --coeff_source direction")
            if not sh.get("passes_both", True):
                print("  [warn] tuned shared coefficient did not pass the coherence "
                      "threshold in both directions", file=sys.stderr)
            mag = sh["coeff"]
        else:
            ch = entry["chosen"].get(str(sign))
            if ch is None:
                raise RuntimeError(f"direction {sign:+d} was not tuned for this model")
            mag = abs(ch["coeff"])
        coeffs[cond] = sign * mag * coeff_scale
    return coeffs


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_batched(model, tok, steerer, texts, coeffs, batch_size, max_new_tokens):
    device = model.get_input_embeddings().weight.device
    # Sort by length so batches pad less; restore order afterwards
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    outs = [None] * len(texts)
    for b in tqdm(range(0, len(order), batch_size), desc="generating", ncols=80):
        idx = order[b:b + batch_size]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)      # chat template already adds BOS
        steerer.coeffs = torch.tensor([coeffs[i] for i in idx], dtype=torch.float32)
        try:
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
        finally:
            steerer.coeffs = None
        dec = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, dec):
            outs[i] = t
    return outs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    parser.add_argument("--tuning_dir", type=Path, default=Path("results/steering_coherence"),
                        help="Directory with steer_coherence_summary.json")
    parser.add_argument("--vector_key", default="overall", choices=["overall"] + TYPES,
                        help="Vector inside the file; must match the one used for tuning")
    parser.add_argument("--conditions", nargs="+", default=["implicit", "explicit", "none"],
                        choices=list(CONDITION_SIGN))
    parser.add_argument("--coeff_source", choices=["shared", "direction"], default="shared",
                        help="'shared': same tuned magnitude both ways (default); "
                             "'direction': each direction's own tuned coefficient")
    parser.add_argument("--coeff_scale", type=float, default=1.0,
                        help="Multiply the tuned coefficient (e.g. 0.5, 1.5 for dose-response)")
    parser.add_argument("--data", default="tomstories_vignettes/implicit_vignettes.json")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13743846335)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--output_dir", default="results/steered_vignettes")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_names = selected_models(args, parser)

    summary_path = args.tuning_dir / "steer_coherence_summary.json"
    summary = load_json(summary_path)

    implicit_data = load_json(args.data)
    random.seed(args.seed)
    sample_keys = random.sample(list(implicit_data.keys()), args.n_samples)

    scale_tag = "" if args.coeff_scale == 1.0 else f"_x{args.coeff_scale:g}"

    for model_slug in model_names:
        print(f"\n=== {model_slug} ===")
        if model_slug not in summary:
            print(f"  [skip] no tuning results in {summary_path}", file=sys.stderr)
            continue
        entry = summary[model_slug]
        try:
            coeffs = resolve_steering(entry, args.conditions, args.coeff_source, args.coeff_scale)
        except RuntimeError as e:
            print(f"  [skip] {e}", file=sys.stderr)
            continue
        layer = entry["layer"]
        vec_path = Path(entry["vector"])
        tuned_key = entry.get("vector_key")
        if tuned_key is not None and tuned_key != args.vector_key:
            print(f"  [warn] tuned with vector key {tuned_key!r} but steering with "
                  f"{args.vector_key!r}; the coefficient may not transfer", file=sys.stderr)

        out_paths = {c: out_dir / f"{model_slug}_steered_vignettes_{c}{'' if c == 'none' else scale_tag}.json"
                     for c in args.conditions}
        todo_conds = [c for c in args.conditions if not out_paths[c].exists()]
        if not todo_conds:
            print("  all conditions already generated; skipping")
            continue

        v = load_unit_vector(vec_path, args.vector_key)
        print(f"  layer {layer}, vector {vec_path.name}[{args.vector_key}], "
              + ", ".join(f"{c}: {coeffs[c]:+.2f}" for c in todo_conds))

        model, tok = load_model(MODELS[model_slug], flash_attention=True)
        if v.numel() != model.config.hidden_size:
            raise ValueError(f"vector dim {v.numel()} != hidden size {model.config.hidden_size}")
        system_ok = supports_system_role(tok)
        if not system_ok:
            print("  Note: no system role in chat template; prepending system prompt to user turn")
        steerer = RowSteerer(model, layer, v)

        # One job per (scenario, condition), all generated together
        jobs, texts, row_coeffs = [], [], []
        for c in todo_conds:
            for ik in sample_keys:
                d = implicit_data[ik]
                typ = d["type"]
                prompt = PROMPT_TEMPLATE.format(setting=d["setting"], relationship=d["relationship"],
                                                type_cap=typ.capitalize(), answer=d["answer"])
                jobs.append((c, ik))
                texts.append(format_chat(tok, SYSTEM_PROMPTS[typ], prompt, system_ok))
                row_coeffs.append(coeffs[c])

        try:
            outputs = generate_batched(model, tok, steerer, texts, row_coeffs,
                                       args.batch_size, args.max_tokens)
        finally:
            steerer.remove()

        for c in todo_conds:
            vignettes = {}
            for (jc, ik), out in zip(jobs, outputs):
                if jc != c:
                    continue
                d = implicit_data[ik]
                vignettes[ik] = {"setting": d["setting"], "relationship": d["relationship"],
                                 "answer": d["answer"], "type": d["type"], "output": out}
            meta = {"model": model_slug, "condition": c, "layer": layer,
                    "coeff": coeffs[c], "coeff_source": args.coeff_source,
                    "coeff_scale": args.coeff_scale, "vector": str(vec_path),
                    "vector_key": args.vector_key, "normalized": True,
                    "max_tokens": args.max_tokens, "seed": args.seed}
            out_paths[c].write_text(json.dumps({"meta": meta, "vignettes": vignettes}, indent=2))
            print(f"  saved → {out_paths[c]}")

        del model
        free_cuda()


if __name__ == "__main__":
    main()