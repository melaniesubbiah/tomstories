"""
Evaluate models on ToMStories in multiple-choice or open-ended mode. Local models run
through vLLM, API models through their provider. Open-ended answers are scored by
embedding similarity to the options and/or an LLM judge.

Outputs:
  <results_dir>/<model>_<dataset>_<mode>_results.json   per-sample predictions and scores
  <results_dir>/<dataset>_metrics.json                  per-model accuracy and related metrics
"""

import argparse
import asyncio
import glob
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, for `common`
from common.api import map_async, with_backoff
from common.data import dataset_stem, load_json, load_vignettes, save_json
from common.hf import load_vllm, supports_system_role
from common.models import ALL_MODELS, MODELS, add_model_args, selected_models
from common.prompts import (SYSTEM_PROMPT, SYSTEM_PROMPT_OE, TYPES, chat_messages,
                            mc_question, oe_question, parse_answer, parse_answer_oe)

load_dotenv()


# ---------------------------------------------------------------------------
# Open-ended scoring (sentence-transformer embeddings)
# ---------------------------------------------------------------------------

def load_embedder():
    from sentence_transformers import SentenceTransformer
    print("Loading sentence-transformers embedder (all-MiniLM-L6-v2)...")
    return SentenceTransformer("all-MiniLM-L6-v2")


def score_oe(embedder, pred_text, gold, options):
    from sklearn.metrics.pairwise import cosine_similarity
    texts = [pred_text] + options
    embs  = embedder.encode(texts, show_progress_bar=False)
    sims  = cosine_similarity(embs[0:1], embs[1:])[0]
    resolved  = options[int(np.argmax(sims))]
    sim_gold  = float(sims[options.index(gold)])
    return resolved, sim_gold


# ---------------------------------------------------------------------------
# Open-ended scoring (LLM judge)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """\
You are evaluating answers to Theory of Mind questions about literary vignettes.
Given the question, the correct answer, and a model's free-form answer, rate the
model's answer on this scale:

  0 – Completely wrong or unrelated
  1 – Loosely related but not the right concept
  2 – Partially correct; captures part of the idea
  3 – Essentially correct; minor wording differences
  4 – Exactly or semantically equivalent to the correct answer

Respond with only the integer (0, 1, 2, 3, or 4). No explanation."""

JUDGE_USER_TEMPLATE = (
    "Question: {question}\n"
    "Correct answer: {gold}\n"
    "Model's answer: {pred}\n"
    "Score:"
)


def make_async_judge_fn(judge_model_name):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    async def judge_fn_async(pred, gold, question):
        user_msg = JUDGE_USER_TEMPLATE.format(question=question, gold=gold, pred=pred)

        async def _call():
            r = await client.chat.completions.create(
                model=judge_model_name, max_tokens=5, temperature=0,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
            )
            return r.choices[0].message.content

        try:
            raw   = await with_backoff(_call)
            match = re.search(r"[0-4]", raw)
            score = int(match.group()) if match else 0
        except Exception as e:
            tqdm.write(f"\nJudge error: {e}")
            score = 0
        return score, score >= 3

    return judge_fn_async


def run_api_batch(cfg, user_messages, system_prompt, max_tokens=400, concurrency=20):
    """
    One chat completion per user message from an API model, in input order.

    A request that keeps failing yields None rather than an empty string, so the
    sample is left unsaved and a rerun retries it instead of recording a
    permanent parse failure.
    """
    provider, api_id = cfg["provider"], cfg["api_id"]

    if provider == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

        async def call(user_msg):
            r = await client.chat.completions.create(
                model=api_id, max_tokens=max_tokens, temperature=0,
                messages=chat_messages(system_prompt, user_msg),
            )
            return r.choices[0].message.content or ""

    elif provider == "anthropic":
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        async def call(user_msg):
            r = await client.messages.create(
                model=api_id, max_tokens=max_tokens, temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")

    else:
        raise ValueError(f"unknown provider {provider!r} for API model {api_id!r}")

    async def worker(user_msg):
        try:
            return await with_backoff(lambda: call(user_msg))
        except Exception as e:
            tqdm.write(f"\nAPI error (sample left unsaved for retry): {e}")
            return None

    return asyncio.run(map_async(user_messages, worker, concurrency,
                                 desc=f"{api_id} ({provider})"))


def run_vllm_batch(cfg, user_messages, system_prompt, max_tokens=400):
    from vllm import SamplingParams

    hf_id = cfg["hf_id"]
    llm = load_vllm(
        cfg,
        dtype="float16" if cfg.get("quantize") else "bfloat16",
        max_model_len=None if "gemma" in hf_id else 8192,
    )

    system_ok = supports_system_role(llm.get_tokenizer())
    if not system_ok:
        print("  Note: model does not support system role — merging into user message")
    messages_list = [chat_messages(system_prompt, user_msg, system_ok)
                     for user_msg in user_messages]

    max_tokens = 1500 if "deepseek" in hf_id else max_tokens
    sampling_params = SamplingParams(temperature=0.01, max_tokens=max_tokens)
    outputs = llm.chat(messages_list, sampling_params, use_tqdm=True)
    return [o.outputs[0].text for o in outputs]


async def _run_judge_batch_async(judge_fn_async, items, results, concurrency=20):
    """
    Concurrently score a list of (sid, pred_raw, gold, question) tuples.
    Updates results[sid] in place with llm_score / llm_correct.
    """
    semaphore = asyncio.Semaphore(concurrency)
    lock      = asyncio.Lock()
    pbar      = tqdm(total=len(items), desc="LLM judge")

    async def score_one(sid, pred_raw, gold, question):
        async with semaphore:
            llm_score, llm_correct = await judge_fn_async(pred_raw, gold, question)
        async with lock:
            e = results[sid]
            e["llm_score"]   = llm_score
            e["llm_correct"] = llm_correct
            if e.get("pred") is None:
                e["correct"] = llm_correct
            pbar.update(1)

    await asyncio.gather(*[score_one(*item) for item in items])
    pbar.close()


def process_output(output, sample, is_oe, embedder=None):
    if is_oe:
        pred_raw = parse_answer_oe(output)
        gold = sample.get("answer")
        entry = {
            "type":       sample["type"],
            "split":      sample["split"],
            "pred_raw":   pred_raw,
            "raw_output": output,
        }
        if gold is not None:
            entry["gold"] = gold
        if embedder is not None and gold is not None:
            pred_resolved, sim = score_oe(
                embedder, pred_raw, gold, sample["answer_options"]
            )
            entry["pred"]           = pred_resolved
            entry["sim_to_correct"] = sim
            entry["correct"]        = pred_resolved == gold
    else:
        pred_num  = parse_answer(output)
        pred_text = sample["answer_options"][pred_num - 1] if pred_num is not None else None
        gold      = sample.get("answer")   # None for conflicted samples (no single gold answer)
        entry = {
            "type":       sample["type"],
            "split":      sample["split"],
            "pred":       pred_text,
            "raw_output": output,
        }
        if gold is not None:
            entry["gold"]    = gold
            entry["correct"] = pred_text == gold
    return entry


def benchmark(model_name, samples, output_path, mode="mc",
              oe_scorer="embedding", judge_model="claude-sonnet-4-6",
              api_concurrency=20):
    cfg    = ALL_MODELS[model_name]
    is_oe  = (mode == "oe")
    sys_prompt = SYSTEM_PROMPT_OE if is_oe else SYSTEM_PROMPT
    make_p     = oe_question      if is_oe else mc_question
    use_embed  = is_oe and oe_scorer in ("embedding", "both")
    use_llm    = is_oe and oe_scorer in ("llm", "both")

    embedder        = load_embedder()             if use_embed else None
    async_judge_fn  = make_async_judge_fn(judge_model) if use_llm   else None

    results = load_json(output_path) if os.path.exists(output_path) else {}

    # Samples that need model inference (not yet run at all).
    pending_model = [(sid, s) for sid, s in samples if sid not in results]

    # Samples already run (have pred_raw) but missing a requested scorer.
    # Applies only in OE mode when switching e.g. from --oe_scorer embedding
    # to --oe_scorer both.
    pending_score = []
    if is_oe:
        for sid, s in samples:
            if sid not in results or "pred_raw" not in results[sid]:
                continue
            e = results[sid]
            if (use_embed and "sim_to_correct" not in e) or \
               (use_llm   and "llm_score"      not in e):
                pending_score.append((sid, s))

    if not pending_model and not pending_score:
        print(f"{model_name}: all {len(results)} samples already done.")
        return results


    if pending_score:
        print(f"Applying missing scorers to {len(pending_score)} cached outputs…")
        # Embedding scorer (sync, fast locally)
        if use_embed:
            for sid, sample in tqdm(pending_score, desc="embedding rescore"):
                e = results[sid]
                if "sim_to_correct" not in e:
                    pred_resolved, sim = score_oe(
                        embedder, e["pred_raw"], sample["answer"], sample["answer_options"]
                    )
                    e["pred"]           = pred_resolved
                    e["sim_to_correct"] = sim
                    e["correct"]        = pred_resolved == sample["answer"]
        # LLM judge (async batch)
        if use_llm:
            judge_items = [
                (sid, results[sid]["pred_raw"], sample["answer"], sample["question"])
                for sid, sample in pending_score
                if "llm_score" not in results[sid]
            ]
            if judge_items:
                print(f"Running async LLM judge on {len(judge_items)} cached outputs…")
                asyncio.run(_run_judge_batch_async(async_judge_fn, judge_items, results,
                                                   concurrency=api_concurrency))
        save_json(output_path, results)

    if not pending_model:
        return results

    user_messages = [make_p(s) for _, s in pending_model]
    if "provider" in cfg:
        print(f"\nQuerying {cfg['api_id']} via {cfg['provider']} "
              f"(concurrency={api_concurrency})…")
        raw_outputs = run_api_batch(cfg, user_messages, sys_prompt, max_tokens=400,
                                    concurrency=api_concurrency)
    else:
        print(f"\nLoading {cfg['hf_id']} with vLLM (tp={cfg.get('tp', 1)}, "
              f"quantize={cfg.get('quantize', False)})…")
        raw_outputs = run_vllm_batch(cfg, user_messages, sys_prompt, max_tokens=400)

    print(f"Processing {len(pending_model)} outputs…")
    n_failed = 0
    for (sid, sample), output in zip(pending_model, raw_outputs):
        if output is None:      # request failed; leave unsaved so a rerun retries it
            n_failed += 1
            continue
        results[sid] = process_output(output, sample, is_oe, embedder)
    if n_failed:
        print(f"  {n_failed} request(s) failed and were not saved; rerun to retry them")

    if use_llm:
        judge_items = [
            (sid, results[sid]["pred_raw"], sample["answer"], sample["question"])
            for sid, sample in pending_model
            if sid in results
        ]
        print(f"Running async LLM judge on {len(judge_items)} outputs…")
        asyncio.run(_run_judge_batch_async(async_judge_fn, judge_items, results,
                                           concurrency=api_concurrency))

    save_json(output_path, results)
    return results


def compute_metrics(results):
    by_type = defaultdict(lambda: {"gold": [], "pred": []})
    overall = {"gold": [], "pred": []}
    sims = []

    for entry in results.values():
        if "gold" not in entry:   # conflicted samples have no single gold answer
            continue
        by_type[entry["type"]]["gold"].append(entry["gold"])
        by_type[entry["type"]]["pred"].append(entry["pred"] if entry["pred"] else "__none__")
        overall["gold"].append(entry["gold"])
        overall["pred"].append(entry["pred"] if entry["pred"] else "__none__")
        if "sim_to_correct" in entry:
            sims.append(entry["sim_to_correct"])

    def safe_metrics(gold, pred):
        labels = sorted(set(gold))
        if len(labels) < 2:
            return {"accuracy": float(np.mean([g == p for g, p in zip(gold, pred)]))}
        bacc = balanced_accuracy_score(gold, pred)
        f1   = f1_score(gold, pred, average="macro", zero_division=0)
        acc  = np.mean([g == p for g, p in zip(gold, pred)])
        return {"accuracy": acc, "bacc": bacc, "f1_macro": f1}

    metrics = {"overall": safe_metrics(overall["gold"], overall["pred"])}
    for typ, d in by_type.items():
        metrics[typ] = safe_metrics(d["gold"], d["pred"])

    metrics["parse_failures"] = sum(1 for e in results.values() if e.get("pred") is None)
    metrics["n"] = len(results)
    if sims:
        metrics["mean_sim"] = float(np.mean(sims))

    llm_entries = [e for e in results.values() if "llm_score" in e]
    if llm_entries:
        metrics["llm_accuracy"]   = float(np.mean([e["llm_correct"] for e in llm_entries]))
        metrics["mean_llm_score"] = float(np.mean([e["llm_score"]   for e in llm_entries]))

    return metrics


def print_summary(all_metrics):
    types   = TYPES
    has_sim = any("mean_sim"     in m for m in all_metrics.values())
    has_llm = any("llm_accuracy" in m for m in all_metrics.values())
    extra = ""
    if has_sim: extra += f"  {'EmbSim':>7}"
    if has_llm: extra += f"  {'LLMAcc':>7}  {'LLMScore':>8}"
    header = (
        f"{'Model':<28} {'Overall Acc':>11} {'BAcc':>7} {'F1':>6}  "
        + "  ".join(f"{t[:6]:>7}" for t in types)
        + extra
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for model_name, m in all_metrics.items():
        ov  = m.get("overall", {})
        row = (
            f"{model_name:<28}"
            f" {ov.get('accuracy', 0)*100:>10.1f}%"
            f" {ov.get('bacc', 0)*100:>6.1f}%"
            f" {ov.get('f1_macro', 0)*100:>5.1f}%"
        )
        for t in types:
            row += f"  {m.get(t, {}).get('accuracy', 0)*100:>6.1f}%"
        if has_sim:
            row += f"  {m.get('mean_sim', 0):>7.3f}"
        if has_llm:
            row += f"  {m.get('llm_accuracy', 0)*100:>6.1f}%  {m.get('mean_llm_score', 0):>8.2f}"
        row += f"  (n={m['n']})"
        print(row)
    print("=" * len(header))

def _mode(key):
    """Extract 'mc' or 'oe' from a key like 'gpt-4o-mini [mc]'."""
    return key.split("[")[-1].rstrip("]") if "[" in key else "mc"


def save_metrics(all_metrics, path):
    """
    Write per-model metrics so tables and plots can be rebuilt by a separate script.

    Each entry keeps the model key, mode, parameter count / provider, and every
    computed score, which is everything the paper's tables and the accuracy-vs-scale
    plot are derived from. Per-sample predictions stay in the
    <model>_<dataset>_<mode>_results.json files written during the run.
    """
    out = {}
    for key, m in all_metrics.items():
        model_name = m.get("_model_name", key)
        cfg = ALL_MODELS.get(model_name, {})
        out[key] = {
            "model":    model_name,
            "mode":     _mode(key),
            "params_b": cfg.get("params_b"),
            "provider": cfg.get("provider"),
            **{k: v for k, v in m.items() if k != "_model_name"},
        }
    save_json(path, out, indent=2)
    print(f"\nMetrics → {path}")


def main():
    parser = argparse.ArgumentParser()
    add_model_args(
        parser, registry=ALL_MODELS,
        help="Models to evaluate. Local models run through vLLM; API models "
             "(gpt-4o-mini, claude-sonnet-4.6) run only when named here.",
        all_help="Run all local models; API models stay opt-in via --models",
    )
    parser.add_argument("--mode", choices=["mc", "oe", "both"], default="mc",
                        help="mc=multiple-choice, oe=open-ended, both=run both")
    parser.add_argument("--oe_scorer", choices=["embedding", "llm", "both"], default="both",
                        help="OE scoring method: embedding similarity, LLM judge, or both")
    parser.add_argument("--judge_model", default="gpt-4o-mini",
                        help="Closed model to use as LLM judge (OE mode only)")
    parser.add_argument("--tomstories_vignettes",    default="tomstories_vignettes/implicit_vignettes.json")
    parser.add_argument("--split",   choices=["train", "val", "all"], default="all")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--results_dir", default="results/benchmark")
    parser.add_argument("--api_concurrency", type=int, default=20,
                        help="Max concurrent requests for API models (default: 20)")
    parser.add_argument("--summarize", action="store_true",
                        help="Summarize existing result files without running any model, "
                             "then exit")
    args = parser.parse_args()

    ds_stem      = dataset_stem(args.tomstories_vignettes)
    metrics_path = os.path.join(args.results_dir, f"{ds_stem}_metrics.json")

    # ── Summary from existing result files ─────────────────────────────────
    if args.summarize:
        valid_sids = set(load_vignettes(args.tomstories_vignettes, args.split))

        all_metrics = {}
        for mode in ("mc", "oe"):
            pattern = os.path.join(args.results_dir, f"*_{ds_stem}_{mode}_results.json")
            for fpath in sorted(glob.glob(pattern)):
                stem = os.path.basename(fpath)
                model_name = stem[: stem.index(f"_{ds_stem}")]
                results  = load_json(fpath)
                filtered = {sid: e for sid, e in results.items() if sid in valid_sids}
                if not filtered:
                    continue
                key = f"{model_name} [{mode}]"
                all_metrics[key] = compute_metrics(filtered)
                all_metrics[key]["_model_name"] = model_name

        if not all_metrics:
            print(f"No result files found in {args.results_dir}/ for dataset '{ds_stem}'.")
            return
        print_summary(all_metrics)
        save_metrics(all_metrics, metrics_path)
        return

    # --all covers the local registry only, so it never spends API budget
    model_names = selected_models(args, parser, registry=MODELS,
                                  extra_hint=", or --summarize")

    modes = ["mc", "oe"] if args.mode == "both" else [args.mode]

    os.makedirs(args.results_dir, exist_ok=True)

    samples = list(load_vignettes(args.tomstories_vignettes, args.split).items())
    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"Evaluating {len(samples)} samples | models: {len(model_names)} | mode(s): {modes}")

    all_metrics = {}
    for mode in modes:
        print(f"\n{'='*60}\nMode: {mode.upper()}\n{'='*60}")
        for model_name in model_names:
            output_path = os.path.join(
                args.results_dir, f"{model_name}_{ds_stem}_{mode}_results.json"
            )
            results = benchmark(
                model_name, samples, output_path,
                mode=mode, oe_scorer=args.oe_scorer, judge_model=args.judge_model,
                api_concurrency=args.api_concurrency,
            )
            valid_sids = {sid for sid, _ in samples}
            filtered   = {sid: e for sid, e in results.items() if sid in valid_sids}
            if len(filtered) < len(results):
                print(f"  (dropped {len(results) - len(filtered)} results not in current dataset)")
            key = f"{model_name} [{mode}]"
            all_metrics[key] = compute_metrics(filtered)
            all_metrics[key]["_model_name"] = model_name

    print_summary(all_metrics)
    save_metrics(all_metrics, metrics_path)


if __name__ == "__main__":
    main()
