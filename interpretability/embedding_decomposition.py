"""
Embedding-space factorization analysis.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
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

TYPES   = ["wound", "emotion", "fear", "motivation"]
COLORS  = {"wound": "#4C72B0", "emotion": "#DD8452", "fear": "#55A868", "motivation": "#C44E52"}
MARKERS = {"implicit": "o", "explicit": "^"}



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
        f"Vignette: {sample['scenario']}\n\n"
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


def compute_centroids(acts, labels):
    centroids = {}
    for typ in TYPES:
        for style in ("implicit", "explicit"):
            idx = [i for i, (t, s) in enumerate(labels) if t == typ and s == style]
            if idx:
                centroids[(typ, style)] = acts[idx].mean(dim=0)
    return centroids


def _draw_centroid_pca_axes(ax, pca, acts, labels, centroids, ev, pc_x, pc_y):
    coords_all = pca.transform(acts.numpy())
    for typ in TYPES:
        for style in ("implicit", "explicit"):
            idx = [i for i, (t, s) in enumerate(labels) if t == typ and s == style]
            if idx:
                ax.scatter(coords_all[idx, pc_x], coords_all[idx, pc_y],
                           c=COLORS[typ], marker=MARKERS[style],
                           alpha=0.10, s=7)
    for typ in TYPES:
        if (typ, "explicit") not in centroids or (typ, "implicit") not in centroids:
            continue
        c_e = pca.transform(centroids[(typ, "explicit")].numpy().reshape(1, -1))[0]
        c_i = pca.transform(centroids[(typ, "implicit")].numpy().reshape(1, -1))[0]
        ax.scatter(c_e[pc_x], c_e[pc_y], c=COLORS[typ], marker="^", s=140, zorder=5,
                   edgecolors="black", linewidths=0.8)
        ax.scatter(c_i[pc_x], c_i[pc_y], c=COLORS[typ], marker="o", s=140, zorder=5,
                   edgecolors="black", linewidths=0.8, label=typ)
        ax.annotate("", xy=(c_i[pc_x], c_i[pc_y]), xytext=(c_e[pc_x], c_e[pc_y]),
                    arrowprops=dict(arrowstyle="->", color=COLORS[typ], lw=2.0))
    ax.set_xlabel(f"PCA-{pc_x+1}  ({ev[pc_x]*100:.1f}% var)", fontsize=14)
    ax.set_ylabel(f"PCA-{pc_y+1}  ({ev[pc_y]*100:.1f}% var)", fontsize=14)
    ax.tick_params(labelsize=14)


def analysis_centroid_pca(acts, labels, centroids, layer_idx, out_dir, modelname):
    pca = PCA(n_components=4)
    pca.fit(acts.numpy())
    ev  = pca.explained_variance_ratio_

    style_handles = [
        Line2D([0], [0], marker="^", color="grey", markersize=9, linestyle="", label="explicit"),
        Line2D([0], [0], marker="o", color="grey", markersize=9, linestyle="", label="implicit"),
    ]

    fig, ax = plt.subplots(figsize=(7, 6))
    _draw_centroid_pca_axes(ax, pca, acts, labels, centroids, ev, 0, 1)
    leg1 = ax.legend(handles=style_handles, loc="upper right", fontsize=14,
                     title="Style", title_fontsize=16)
    ax.add_artist(leg1)
    ax.legend(loc="upper left", fontsize=14, title="Type", title_fontsize=16)
    plt.tight_layout()
    path = out_dir / f"{modelname}_layer{layer_idx}_centroid_pca_pc12.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path.name}")

    return pca

def summary_statistics(centroids, layer_idx, out_dir):
    types_present = [t for t in TYPES
                     if (t, "implicit") in centroids and (t, "explicit") in centroids]

    diffs   = {t: (centroids[(t, "implicit")] - centroids[(t, "explicit")]).numpy()
               for t in types_present}
    norms = {t: float(np.linalg.norm(diffs[t])) for t in types_present}
    v_impl  = np.mean(list(diffs.values()), axis=0)       # (hidden,)
    v_hat   = v_impl / (np.linalg.norm(v_impl) + 1e-9)

    print(f"\n[Layer {layer_idx}] Summary statistics")
    print(f"{'cosine(d_t, v_impl)':>20}  {'||d_t||':>10}")
    cosines = {}
    for t in types_present:
        d       = diffs[t]
        cos     = float(d @ v_hat / (np.linalg.norm(d) + 1e-9))
        cosines[t] = cos
        print(f"  {cos:>20.4f}  {np.linalg.norm(d):>10.4f}")

    # Type-averaged summary stats for cross-model table
    summary = {
        "mean_cos":          float(np.mean([cosines[t] for t in types_present])),
        "norm_cv":           float(np.std( [norms[t]   for t in types_present]) /
                                   (np.mean([norms[t]  for t in types_present]) + 1e-9)),
    }

    return v_impl, summary


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model",           default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--implicit_data",   default="tomstories_vignettes/implicit_vignettes.json")
    parser.add_argument("--explicit_data",   default="tomstories_vignettes/explicit_vignettes.json")
    parser.add_argument("--layers",          nargs="+", type=int, default=None,
                        help="Layers to analyse (default: mid−4, mid, mid+4)")
    parser.add_argument("--batch_size",      type=int, default=4)
    parser.add_argument("--output_dir",      default="results/embeddings")
    parser.add_argument("--skip_extraction", action="store_true",
                        help="Load saved activations instead of re-extracting")
    parser.add_argument("--quantize",        action="store_true",
                        help="4-bit bitsandbytes quantization (for 70B+ models)")
    parser.add_argument("--print_summary",   action="store_true",
                        help="Read all saved *_summary.json files in output_dir and print "
                             "the cross-model LaTeX table, then exit")
    args = parser.parse_args()

    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_slug = args.model.split("/")[-1]

    if args.print_summary:
        _print_cross_model_table(out_dir)
        return

    # ── Load tomstories_vignettes ──────────────────────────────────────────────────────────
    with open(args.implicit_data) as f:
        implicit_data = json.load(f)
    with open(args.explicit_data) as f:
        explicit_data = json.load(f)

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
    act_path = out_dir / f"{model_slug}_factorization_acts_all.pt"
    if args.skip_extraction:
        if not act_path.exists():
            raise FileNotFoundError(f"--skip_extraction set but cache not found: {act_path}")
        print(f"\nLoading activations from {act_path}")
        saved      = torch.load(act_path, map_location="cpu", weights_only=True)
        layer_acts = saved["acts"]
        labels     = saved["labels"]
        if args.layers is None:
            args.layers = sorted(layer_acts.keys())
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
        if args.layers is None:
            anchor      = num_layers // 2
            args.layers = sorted({num_layers * 1 // 3, anchor, num_layers * 2 // 3})
            args.layers = [l for l in args.layers if 0 <= l < num_layers]
        print(f"Layers: {args.layers}  (model has {num_layers} layers, "
              f"hidden={model.config.hidden_size})")

        print("\nExtracting activations …")
        layer_acts = extract_activations(
            model, tokenizer, samples_flat, args.layers, args.batch_size, device,
            system_supported=sys_ok,
        )
        torch.save({"acts": layer_acts, "labels": labels, "sids": sids}, act_path)
        print(f"  Saved → {act_path}")

    # ── Per-layer analysis ─────────────────────────────────────────────────
    layer_summaries = {}   # {layer_idx: summary_dict}
    for li in args.layers:
        print(f"\n{'='*60}\nLayer {li}\n{'='*60}")
        acts      = layer_acts[li]                            # (N_total, hidden)
        centroids = compute_centroids(acts, labels)

        analysis_centroid_pca(acts, labels, centroids, li, out_dir, args.model.split('/')[1])
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
        json.dump({"model": args.model, "layers": layer_summaries}, f, indent=2)
    print(f"Summary stats → {summary_path}")

    _print_run_table(model_slug, layer_summaries)


def _print_run_table(model_slug, layer_summaries):
    """Print a LaTeX table for one model run (one row per layer)."""
    print(f"\n--- LaTeX summary table: {model_slug} ---")
    print(r"\begin{tabular}{lrrrr}")
    print(r"\toprule")
    print(r"\textbf{Layer} & "
          r"$\overline{\cos(d_t,\bar{v})}$ & "
          r"$\mathrm{CV}(\|d_t\|)$ \\")
    print(r"\midrule")
    for li, s in sorted(layer_summaries.items()):
        print(f"  {li} & {s['mean_cos']:.3f} & "
              f"{s['norm_cv']:.3f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print("---")


def _print_cross_model_table(out_dir):
    """
    Read all *_summary.json files in out_dir and print a LaTeX table with one
    row per (model, layer) pair, columns averaged across question types.
    """
    import glob
    paths = sorted(glob.glob(str(out_dir / "*_summary.json")))
    if not paths:
        print(f"No *_summary.json files found in {out_dir}")
        return

    rows = []
    for path in paths:
        data = json.load(open(path))
        model = data["model"].split("/")[-1]
        for li, s in sorted(data["layers"].items(), key=lambda x: int(x[0])):
            rows.append((model, int(li), s))

    print(r"\begin{tabular}{llrrrr}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{Layer}& "
          r"$\overline{\cos(d_t,\bar{v})}$ & "
          r"$\mathrm{CV}(\|d_t\|)$ \\")
    print(r"\midrule")
    prev_model = None
    for model, li, s in rows:
        model_cell = model if model != prev_model else ""
        prev_model = model
        print(f"  {model_cell} & {li} & {s['mean_cos']:.3f} & "
              f"{s['norm_cv']:.3f}t \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
