"""
Linear probes across all transformer layers. For each layer, trains a logistic regression probe on last-token residual-stream
activations and reports held-out (val-split) accuracy.
"""

import argparse
import json
from pathlib import Path

import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

TYPES = ["wound", "emotion", "fear", "motivation"]


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


def extract_all_layers(model, tokenizer, samples, batch_size, device, system_supported=True):
    num_layers = model.config.num_hidden_layers
    buffers    = {li: [] for li in range(num_layers)}
    hooks      = []

    def make_hook(li):
        def hook_fn(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            buffers[li].append(hs[:, -1, :].detach().float().cpu())
        return hook_fn

    for li in range(num_layers):
        hooks.append(model.model.layers[li].register_forward_hook(make_hook(li)))

    try:
        for i in tqdm(range(0, len(samples), batch_size), desc="extracting activations"):
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

    return {li: torch.cat(buffers[li], dim=0) for li in range(num_layers)}


def fit_probe(X_train, y_train, X_val, y_val):
    if len(X_train) == 0 or len(X_val) == 0:
        return float("nan")
    if len(set(y_train)) < 2:
        majority = list(set(y_train))[0]
        return float(np.mean(y_val == majority))

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
    return clf.score(X_va, y_val)


def chance_accuracy(labels):
    counts = np.bincount(labels)
    return counts.max() / counts.sum()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model",           default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--implicit_data",   default="tomstories_vignettes/implicit_vignettes.json")
    parser.add_argument("--explicit_data",   default="tomstories_vignettes/explicit_vignettes.json")
    parser.add_argument("--batch_size",      type=int, default=4)
    parser.add_argument("--output_dir",      default="results/probes")
    parser.add_argument("--skip_extraction", action="store_true",
                        help="Load cached activations instead of re-running the model")
    parser.add_argument("--quantize",        action="store_true",
                        help="4-bit bitsandbytes quantization (for 70B+ models)")
    parser.add_argument("--answer_per_style", action="store_true",
                        help="Plot separate answer-probe lines for implicit and explicit")
    parser.add_argument("--choice_probe", action="store_true",
                        help="Fit the 4-class choice-index probe (which slot is correct)")
    parser.add_argument("--skip_probes", action="store_true",
                        help="Load saved probe results JSON and skip extraction + fitting")
    args = parser.parse_args()

    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_slug = args.model.split("/")[-1]

    # ── Load datasets ──────────────────────────────────────────────────────
    with open(args.implicit_data) as f:
        implicit_data = json.load(f)
    with open(args.explicit_data) as f:
        explicit_data = json.load(f)

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

    # Encode labels
    le_style  = LabelEncoder().fit(styles_raw)

    y_style  = le_style.transform(styles_raw)

    # Choice probe: 0-based index of the correct answer in answer_options (0–3).
    y_choice = np.array([r["answer_options"].index(r["answer"]) for r in records])

    train_mask = splits == "train"
    val_mask   = splits == "val"

    n_train = train_mask.sum()
    n_val   = val_mask.sum()
    print(f"Train: {n_train}  Val: {n_val}")

    # ── Optionally skip extraction + probing and load saved results ────────
    json_path = out_dir / f"{model_slug}_probe_results.json"

    if args.skip_probes:
        if not json_path.exists():
            raise FileNotFoundError(f"--skip_probes set but results not found: {json_path}")
        print(f"\nLoading probe results from {json_path}")
        results    = json.load(open(json_path))
        layers     = results["layers"]
        num_layers = len(layers)
        acc_choice       = results.get("choice")
        acc_choice_impl  = results.get("choice_implicit")
        acc_choice_expl  = results.get("choice_explicit")
        # jump straight to plotting
        chance_choice = 0.25
    else:
        # ── Load model ─────────────────────────────────────────────────────
        act_path = out_dir / f"{model_slug}_probe_acts.pt"

        if args.skip_extraction:
            if not act_path.exists():
                raise FileNotFoundError(f"--skip_extraction set but cache not found: {act_path}")
            print(f"\nLoading activations from {act_path}")
            saved      = torch.load(act_path, map_location="cpu", weights_only=True)
            layer_acts = saved["acts"]
            num_layers = max(layer_acts.keys()) + 1
            print(f"  Loaded {num_layers} layers")
        else:
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
            print(f"  {num_layers} layers, hidden={model.config.hidden_size}")

            samples_flat = [r for r in records]
            print("\nExtracting activations at all layers …")
            layer_acts = extract_all_layers(
                model, tokenizer, samples_flat, args.batch_size, device,
                system_supported=sys_ok,
            )
            torch.save({"acts": layer_acts, "sids": sids}, act_path)
            print(f"  Saved → {act_path}")

        num_layers = max(layer_acts.keys()) + 1
        layers     = list(range(num_layers))

        # ── Fit probes at every layer ──────────────────────────────────────
        print(f"\nFitting probes across {num_layers} layers …")

        acc_choice = [] if args.choice_probe else None
        acc_choice_impl = [] if (args.answer_per_style and args.choice_probe) else None
        acc_choice_expl = [] if (args.answer_per_style and args.choice_probe) else None

        impl_train = train_mask & (styles_raw == "implicit")
        impl_val   = val_mask   & (styles_raw == "implicit")
        expl_train = train_mask & (styles_raw == "explicit")
        expl_val   = val_mask   & (styles_raw == "explicit")

        for li in tqdm(layers, desc="probing layers"):
            acts = layer_acts[li].numpy()

            if args.choice_probe:
                acc_choice.append(fit_probe(
                    acts[train_mask], y_choice[train_mask],
                    acts[val_mask],   y_choice[val_mask],
                ))

            if args.answer_per_style and args.choice_probe:
                acc_choice_impl.append(fit_probe(
                    acts[impl_train], y_choice[impl_train],
                    acts[impl_val],   y_choice[impl_val],
                ))
                acc_choice_expl.append(fit_probe(
                    acts[expl_train], y_choice[expl_train],
                    acts[expl_val],   y_choice[expl_val],
                ))

        # ── Save raw numbers ───────────────────────────────────────────────
        results = {"layers": layers}
        if args.choice_probe:
            results["choice"] = acc_choice
        if args.choice_per_type:
            for t in TYPES:
                results[f"choice_{t}"] = acc_choice_by_type[t]

        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nRaw results → {json_path}")

        chance_choice = 0.25

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    if args.choice_probe:
        ax.plot(layers, acc_choice, color="#9C27B0", lw=2, label="all")

    ax.set_xlabel("Layer", fontsize=19)
    ax.set_ylabel("Val accuracy", fontsize=19)
    ax.set_xlim(0, num_layers - 1)
    ax.set_ylim(0, 1.02)
    ax.tick_params(labelsize=17)
    ax.legend(fontsize=17, loc="upper left", ncol=1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    suffix = ""
    if args.choice_probe:
        suffix = "_choice_probe"
    plot_path = out_dir / f"{model_slug}_linear_probes_{suffix}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot → {plot_path}")

    # ── Summary ────────────────────────────────────────────────────────────
    def peak(accs, label, chance):
        arr = [a for a in accs if not np.isnan(a)]
        if not arr:
            return
        best = int(np.argmax(arr))
        print(f"  {label:<32}  peak={arr[best]:.3f} at layer {best}  (chance={chance:.3f})")

    print("\nPeak probe accuracies:")
    if args.choice_probe:
        peak(acc_choice, "choice index (0–3)", chance_choice)
    if args.answer_per_style and args.choice_probe:
        peak(acc_choice_impl, "choice — implicit", chance_choice)
        peak(acc_choice_expl, "choice — explicit", chance_choice)


if __name__ == "__main__":
    main()
