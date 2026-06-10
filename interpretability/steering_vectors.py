"""
Extracts implicit/explicit steering vectors.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

SYSTEM_PROMPT = (
    "You are an expert at literary analysis. You will read a series of short vignettes "
    "and analyze the implicit meaning communicated by each scene. After reading the vignette, "
    "answer the multiple choice question which follows. First reason about your answer and then "
    "give your answer in <answer></answer> tags. Your answer should be the number of the correct "
    "choice (1, 2, 3, or 4)."
)

TYPES  = ["wound", "emotion", "fear", "motivation"]
COLORS = {"wound": "#4C72B0", "emotion": "#DD8452", "fear": "#55A868", "motivation": "#C44E52"}



def supports_system_role(tokenizer):
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "x"}],
            tokenize=False, add_generation_prompt=True,
        )
        return True
    except Exception:
        return False


def build_mc_prompt(sample, tokenizer, system_supported=True):
    options  = "".join(f"{i+1}. {sample['answer_options'][i]}\n" for i in range(4))
    user_msg = (
        f"Vignette: {sample['scenario']}\n"
        f"Question: {sample['question']}\n"
        f"{options}"
    )
    if system_supported:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
    else:
        messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user_msg}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Phase 1 — Activation extraction
# ---------------------------------------------------------------------------

def extract_activations(model, tokenizer, samples, layer_indices, batch_size, device,
                        system_supported=True):
    buffers = {li: [] for li in layer_indices}
    hooks   = []

    def make_hook(li):
        def hook_fn(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            buffers[li].append(hs[:, -1, :].detach().float().cpu())
        return hook_fn

    for li in layer_indices:
        hooks.append(model.model.layers[li].register_forward_hook(make_hook(li)))

    try:
        for i in tqdm(range(0, len(samples), batch_size), desc="extracting", leave=False):
            batch   = samples[i : i + batch_size]
            prompts = [build_mc_prompt(s, tokenizer, system_supported) for s in batch]
            inputs  = tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=1024,
            ).to(device)
            with torch.no_grad():
                model(**inputs)
    finally:
        for h in hooks:
            h.remove()

    return {li: torch.cat(buffers[li], dim=0) for li in layer_indices}


# ---------------------------------------------------------------------------
# Phase 2 — Steering vector computation
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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model",         default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--implicit_data", default="tomstories_vignettes/implicit_vignettes.json")
    parser.add_argument("--explicit_data", default="tomstories_vignettes/explicit_vignettes.json")
    parser.add_argument("--layers",        nargs="+", type=int, default=None,
                        help="Layers to extract from (default: [mid-4, mid, mid+4])")
    parser.add_argument("--granularity",   choices=["overall", "per_type", "both"],
                        default="both")
    parser.add_argument("--batch_size",    type=int, default=4)
    parser.add_argument("--output_dir",    default="results/steering")
    parser.add_argument("--quantize",      action="store_true",
                        help="4-bit bitsandbytes quantization (for 70B+ models)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_slug = args.model.split("/")[-1]

    # ── Load ToMStories ──────────────────────────────────────────────────────────
    with open(args.implicit_data) as f:
        implicit_data = json.load(f)
    with open(args.explicit_data) as f:
        explicit_data = json.load(f)

    common_ids  = sorted(set(implicit_data) & set(explicit_data))
    train_ids   = [k for k in common_ids if implicit_data[k]["split"] == "train"]
    print(f"Matched pairs: {len(common_ids)}  (train: {len(train_ids)})")

    impl_train  = [implicit_data[k] for k in train_ids]
    expl_train  = [explicit_data[k] for k in train_ids]
    train_types = [implicit_data[k]["type"] for k in train_ids]

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\nLoading {args.model} …")
    if args.quantize:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        n_gpus = torch.cuda.device_count()
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
        max_memory = {i: f"{int(gpu_mem_gb * 0.85)}GiB" for i in range(n_gpus)}
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb, device_map="auto",
            max_memory=max_memory,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto"
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    sys_ok = supports_system_role(tokenizer)
    if not sys_ok:
        print("  Note: model does not support system role — merging into user message")

    device     = next(model.parameters()).device
    num_layers = model.config.num_hidden_layers

    if args.layers is None:
        mid         = num_layers // 2
        args.layers = sorted({mid - 4, mid, mid + 4})
        args.layers = [l for l in args.layers if 0 <= l < num_layers]
    print(f"Layers: {args.layers}  (model has {num_layers} layers, hidden={model.config.hidden_size})")

    # ── Phase 1: Activations ───────────────────────────────────────────────
    act_path = out_dir / f"{model_slug}_activations.pt"
    print("\n── Phase 1: Extracting activations ──")
    print("  Implicit …")
    impl_acts = extract_activations(
        model, tokenizer, impl_train, args.layers, args.batch_size, device,
        system_supported=sys_ok,
    )
    print("  Explicit …")
    expl_acts = extract_activations(
        model, tokenizer, expl_train, args.layers, args.batch_size, device,
        system_supported=sys_ok,
    )
    torch.save({"implicit": impl_acts, "explicit": expl_acts}, act_path)
    print(f"  Saved → {act_path}")

    # ── Phase 2: Steering vectors ──────────────────────────────────────────
    print("\n── Phase 2: Computing steering vectors ──")
    all_vectors = {}
    for li in args.layers:
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
