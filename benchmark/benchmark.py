"""
Benchmark open-source models on ToMStories
"""

import argparse
import asyncio
import glob
import json
import os
import random
import re
from collections import defaultdict

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm import tqdm
from adjustText import adjust_text
from openai import AsyncOpenAI

load_dotenv()


MODELS = {
    "llama-3.2-1b":      {"hf_id": "meta-llama/Llama-3.2-1B-Instruct",           "params_b": 1},
    "llama-3.2-3b":      {"hf_id": "meta-llama/Llama-3.2-3B-Instruct",           "params_b": 3},
    "llama-3.1-8b":      {"hf_id": "meta-llama/Llama-3.1-8B-Instruct",           "params_b": 8},
    "llama-3.1-70b":     {"hf_id": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4", "params_b": 70,  "tp": 2, "quantize": "awq_marlin"},
    "llama-3.3-70b":     {"hf_id": "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4", "params_b": 70,  "tp": 2, "quantize": "awq_marlin"},
    "qwen-2.5-7b":       {"hf_id": "Qwen/Qwen2.5-7B-Instruct",                            "params_b": 7},
    "qwen-2.5-72b":      {"hf_id": "Qwen/Qwen2.5-72B-Instruct-AWQ",                       "params_b": 72,  "tp": 2, "quantize": "awq_marlin"},
    "mistral-7b":        {"hf_id": "mistralai/Mistral-7B-Instruct-v0.3",         "params_b": 7},
    "gemma-2-9b":        {"hf_id": "google/gemma-2-9b-it",                       "params_b": 9},
    "gemma-2-27b":       {"hf_id": "google/gemma-2-27b-it",                      "params_b": 27, "tp": 2},
    "phi-4":             {"hf_id": "microsoft/phi-4",                            "params_b": 14},
    "deepseek-r1-7b":    {"hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",   "params_b": 7},
    "deepseek-r1-8b":    {"hf_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",  "params_b": 8},
}

SYSTEM_PROMPT = (
    "You are an expert at literary analysis. You will read a series of short vignettes "
    "and analyze the implicit meaning communicated by each scene. After reading the vignette, "
    "answer the multiple choice question which follows. First reason about your answer and then "
    "give your answer in <answer></answer> tags. Your answer should be the number of the correct "
    "choice (1, 2, 3, or 4)."
)

SYSTEM_PROMPT_OE = (
    "You are an expert at literary analysis. You will read a series of short vignettes "
    "and analyze the implicit meaning communicated by each scene. After reading the vignette, "
    "answer the question with a concise phrase (a few words). First reason about your answer, "
    "then give your final answer in <answer></answer> tags."
)

MODE_COLORS = {"mc": "#4C72B0", "oe": "#DD8452"}


def make_prompt(sample):
    options = "".join(f"{i+1}. {sample['answer_options'][i]}\n" for i in range(4))
    return f"Vignette: {sample['scenario']}\nQuestion: {sample['question']}\n{options}"


def make_prompt_oe(sample):
    return f"Vignette: {sample['scenario']}\nQuestion: {sample['question']}"


def parse_answer(output):
    """Extract 1-4 from <answer> tags, or fall back to last digit found."""
    if "<answer>" in output and "</answer>" in output:
        tag_content = output.split("<answer>")[1].split("</answer>")[0].strip()
        digits = re.findall(r"[1-4]", tag_content)
        if digits:
            return int(digits[0])
    tail = output[-30:]
    digits = re.findall(r"[1-4]", tail)
    if digits:
        return int(digits[-1])
    return None


def parse_answer_oe(output):
    """Extract free-form answer from <answer> tags, or fall back to last line."""
    if "<answer>" in output and "</answer>" in output:
        return output.split("<answer>")[1].split("</answer>")[0].strip()
    return output.strip().split("\n")[-1].strip()


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
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    async def _call(user_msg):
        r = await client.chat.completions.create(
            model=cfg["api_id"], max_tokens=5, temperature=0,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
        )
        return r.choices[0].message.content

    async def judge_fn_async(pred, gold, question):
        user_msg = JUDGE_USER_TEMPLATE.format(question=question, gold=gold, pred=pred)
        async def _inner():
            return await _call(user_msg)
        try:
            raw   = await _call_with_backoff(_inner)
            match = re.search(r"[0-4]", raw)
            score = int(match.group()) if match else 0
        except Exception as e:
            tqdm.write(f"\nJudge error: {e}")
            score = 0
        return score, score >= 3

    return judge_fn_async


def run_vllm_batch(hf_id, messages_list, tp=1, max_tokens=400, quantize=False):
    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=hf_id,
        tensor_parallel_size=tp,
        dtype="float16" if quantize else "bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=8192,
    )
    if quantize:
        llm_kwargs["quantization"] = quantize  # e.g. "awq_marlin"

    llm = LLM(**llm_kwargs)

    tokenizer = llm.get_tokenizer()
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "x"}],
            tokenize=False, add_generation_prompt=True,
        )
        supports_system = True
    except Exception:
        supports_system = False

    if not supports_system:
        merged = []
        for msgs in messages_list:
            msgs = list(msgs)
            if msgs and msgs[0]["role"] == "system":
                sys_text = msgs[0]["content"]
                msgs = msgs[1:]
                if msgs and msgs[0]["role"] == "user":
                    msgs[0] = {"role": "user",
                               "content": sys_text + "\n\n" + msgs[0]["content"]}
            merged.append(msgs)
        messages_list = merged

    max_tokens = 1500 if 'deepseek' in hf_id else max_tokens
    sampling_params = SamplingParams(temperature=0.01, max_tokens=max_tokens)
    outputs = llm.chat(messages_list, sampling_params, use_tqdm=True)
    return [o.outputs[0].text for o in outputs]


async def _call_with_backoff(coro_fn, max_retries=6):
    delay = 1.0
    for attempt in range(max_retries):
        try:
            return await coro_fn()
        except Exception as e:
            name = type(e).__name__.lower()
            msg  = str(e).lower()
            is_rate_limit = ("ratelimit" in name or "resourceexhausted" in name
                             or "rate limit" in msg or "429" in msg)
            if not is_rate_limit or attempt == max_retries - 1:
                raise
            jitter = random.uniform(0, delay * 0.2)
            await asyncio.sleep(delay + jitter)
            delay = min(delay * 2, 60.0)


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
              tensor_parallel_size=None, api_concurrency=20):
    cfg    = MODELS[model_name]
    is_oe  = (mode == "oe")
    sys_prompt = SYSTEM_PROMPT_OE if is_oe else SYSTEM_PROMPT
    make_p     = make_prompt_oe   if is_oe else make_prompt
    use_embed  = is_oe and oe_scorer in ("embedding", "both")
    use_llm    = is_oe and oe_scorer in ("llm", "both")

    embedder        = load_embedder()             if use_embed else None
    async_judge_fn  = make_async_judge_fn(judge_model) if use_llm   else None

    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
    else:
        results = {}

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
        with open(output_path, "w") as f:
            json.dump(results, f)

    if not pending_model:
        return results


    tp       = tensor_parallel_size if tensor_parallel_size is not None else cfg.get("tp", 1)
    quantize = cfg.get("quantize", False)
    print(f"\nLoading {cfg['hf_id']} with vLLM (tp={tp}, quantize={quantize})…")

    messages_list = [
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": make_p(s)},
        ]
        for _, s in pending_model
    ]

    raw_outputs = run_vllm_batch(cfg["hf_id"], messages_list, tp=tp, max_tokens=400,
                                 quantize=quantize)

    print(f"Processing {len(pending_model)} outputs…")
    for (sid, sample), output in zip(pending_model, raw_outputs):
        results[sid] = process_output(output, sample, is_oe, embedder)

    if use_llm:
        judge_items = [
            (sid, results[sid]["pred_raw"], sample["answer"], sample["question"])
            for sid, sample in pending_model
        ]
        print(f"Running async LLM judge on {len(judge_items)} outputs…")
        asyncio.run(_run_judge_batch_async(async_judge_fn, judge_items, results,
                                           concurrency=api_concurrency))

    with open(output_path, "w") as f:
        json.dump(results, f)

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
    types   = ["wound", "emotion", "fear", "motivation"]
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

def print_latex_table(all_metrics, human_dir="human_baseline/results",
                      dataset_path="tomstories_vignettes/implicit_vignettes.json", split="val"):
    """Print a LaTeX table with MC and OE columns, bolding per-column maxima."""
    types      = ["wound", "emotion", "fear", "motivation"]
    type_abbr  = ["Wound", "Emot.", "Fear", "Motiv."]

    # Collect per-model MC and OE metrics, preserving MODELS registry order
    model_order = list(MODELS.keys())
    mc, oe = {}, {}
    for key, m in all_metrics.items():
        mname = m.get("_model_name", "")
        mode  = _mode(key)
        if mode == "mc":
            mc[mname] = m
        elif mode == "oe":
            oe[mname] = m

    rows = [n for n in model_order if n in mc or n in oe]

    # Human baseline
    human_row = None
    dataset = json.load(open(dataset_path))
    impl_files = glob.glob(os.path.join(human_dir, "*_implicit*.json"))
    if impl_files:
        combined_gold, combined_pred = [], []
        by_type = defaultdict(lambda: {"gold": [], "pred": []})
        for fp in impl_files:
            res = json.load(open(fp))
            for sid, entry in res.items():
                if sid not in dataset:
                    continue
                if split != "all" and dataset[sid].get("split") != split:
                    continue
                if entry.get("flag"):
                    continue
                pred = entry.get("pred")
                gold = dataset[sid]["answer"]
                if pred is None:
                    continue
                combined_gold.append(gold)
                combined_pred.append(pred)
                by_type[dataset[sid]["type"]]["gold"].append(gold)
                by_type[dataset[sid]["type"]]["pred"].append(pred)
        if combined_gold:
            human_row = {
                "overall": float(np.mean([g == p for g, p in zip(combined_gold, combined_pred)])),
                **{t: float(np.mean([g == p for g, p in zip(by_type[t]["gold"], by_type[t]["pred"])]))
                   for t in types if by_type[t]["gold"]},
            }

    def acc(m, key):
        if m is None:
            return None
        if key == "overall":
            return m.get("overall", {}).get("accuracy")
        return m.get(key, {}).get("accuracy")

    def sim(m):
        return m.get("mean_sim") if m else None

    def llm(m):
        return m.get("mean_llm_score") if m else None

    # Find per-column maxima for bolding (model rows only, not human)
    cols_mc = [("overall", "MC Overall")] + [(t, t) for t in types]
    cols_oe = [("overall", "OE Overall")] + [(t, t) for t in types]
    has_sim = any(sim(oe.get(n)) is not None for n in rows)
    has_llm = any(llm(oe.get(n)) is not None for n in rows)

    best_mc = {k: max((acc(mc.get(n), k) or 0) for n in rows) for k, _ in cols_mc}
    best_oe = {k: max((acc(oe.get(n), k) or 0) for n in rows) for k, _ in cols_oe}
    best_sim = max((sim(oe.get(n)) or 0) for n in rows) if has_sim else None
    best_llm = max((llm(oe.get(n)) or 0) for n in rows) if has_llm else None

    def fmt_acc(val, best, pct=True):
        if val is None:
            return "--"
        s = f"{val*100:.1f}" if pct else f"{val:.2f}"
        return r"\textbf{" + s + "}" if abs(val - best) < 1e-9 else s

    def fmt_val(val, best):
        if val is None:
            return "--"
        s = f"{val:.2f}"
        return r"\textbf{" + s + "}" if abs(val - best) < 1e-9 else s

    # Header
    n_oe_cols = 5 + (1 if has_sim else 0) + (1 if has_llm else 0)
    print()
    print(r"\begin{tabular}{lc|cccc||c|cccc" + ("c" * (n_oe_cols - 5)) + "}")
    print(r"\toprule")
    oe_span = 5 + (1 if has_sim else 0) + (1 if has_llm else 0)
    print(r"&  \multicolumn{5}{c||}{Multiple-Choice} &  "
          r"\multicolumn{" + str(oe_span) + r"}{c}{Open-Ended}\\")
    oe_extra = (r" & \textbf{EmbSim}" if has_sim else "") + \
               (r" & \textbf{LLMScore}" if has_llm else "")
    print(r"\textbf{Model} & \textbf{Overall} & \textbf{Wound} & \textbf{Emot.} & "
          r"\textbf{Fear} & \textbf{Motiv.} & \textbf{Overall} & \textbf{Wound} & "
          r"\textbf{Emot.} & \textbf{Fear} & \textbf{Motiv.}" + oe_extra + r" \\")
    print(r"\midrule")

    for name in rows:
        m_mc = mc.get(name)
        m_oe = oe.get(name)
        cells = [name.replace("_", r"\_")]
        # MC columns
        cells.append(fmt_acc(acc(m_mc, "overall"), best_mc["overall"]))
        for t in types:
            cells.append(fmt_acc(acc(m_mc, t), best_mc[t]))
        # OE columns
        cells.append(fmt_acc(acc(m_oe, "overall"), best_oe["overall"]))
        for t in types:
            cells.append(fmt_acc(acc(m_oe, t), best_oe[t]))
        if has_sim:
            cells.append(fmt_val(sim(m_oe), best_sim))
        if has_llm:
            cells.append(fmt_val(llm(m_oe), best_llm))
        print("  " + " & ".join(cells) + r" \\")

    if human_row:
        print(r"\midrule")
        cells = ["Human"]
        cells.append(f"{human_row['overall']*100:.1f}")
        for t in types:
            v = human_row.get(t)
            cells.append(f"{v*100:.1f}" if v is not None else "--")
        print("  " + " & ".join(cells) + r" \\")

    print(r"\midrule\midrule")
    print(r"\end{tabular}")


def _mode(key):
    """Extract 'mc' or 'oe' from a key like 'gpt-4o [mc]'."""
    return key.split("[")[-1].rstrip("]") if "[" in key else "mc"


def plot_results(all_metrics, results_dir):
    import matplotlib.pyplot as plt

    def _params(m):
        return MODELS.get(m.get("_model_name", ""), {}).get("params_b")

    open_models   = [(n, _params(m), m["overall"].get("accuracy", 0))
                     for n, m in all_metrics.items() if _params(m)]
    closed_models = [(n, m["overall"].get("accuracy", 0))
                     for n, m in all_metrics.items() if not _params(m)]

    has_closed = bool(closed_models)
    fig, axes = plt.subplots(
        1, 2 if has_closed else 1,
        figsize=(14 if has_closed else 8, 6),
        gridspec_kw={"width_ratios": [3, 1]} if has_closed else None,
    )
    ax_open = axes[0] if has_closed else axes

    MODE_LABELS = {"mc": "Multiple choice", "oe": "Open ended"}

    seen_modes = set()
    texts = []
    for name, params, acc in sorted(open_models, key=lambda x: x[1]):
        mode  = _mode(name)
        color = MODE_COLORS.get(mode, "#555555")
        ax_open.scatter(params, acc * 100, color=color, s=90, zorder=3,
                        label=MODE_LABELS.get(mode, mode) if mode not in seen_modes else None,
                        alpha=0.7)
        clean_name = name.split('[')[0].strip()
        if '70b' in clean_name:
            clean_name = clean_name.split('-')[0] + '-'.join(clean_name.split('-')[1:])
        else:
            clean_name = clean_name.split('-')[0] + '-'.join(clean_name.split('-')[-1:])
        seen_modes.add(mode)
        texts.append(ax_open.text(params, acc*100, clean_name))
    ax_open.set_xscale("log")
    adjust_text(
        texts,
        expand_points=(1.7, 1.7),
        ensure_inside_axes=False,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5)
    )


    ax_open.set_xlabel("Parameters (billions, log scale)", fontsize=15)
    ax_open.set_ylabel("Accuracy (%)", fontsize=15)
    ax_open.grid(True, alpha=0.3)
    ax_open.tick_params(labelsize=13)
    ax_open.legend(title="Mode", fontsize=13, title_fontsize=13)

    if has_closed:
        ax_closed = axes[1]
        seen_modes_closed = set()
        for i, (name, acc) in enumerate(sorted(closed_models, key=lambda x: x[1])):
            mode  = _mode(name)
            color = MODE_COLORS.get(mode, "#555555")
            ax_closed.scatter(i, acc * 100, color=color, s=90, marker="D", zorder=3,
                              label=MODE_LABELS.get(mode, mode) if mode not in seen_modes_closed else None)
            ax_closed.annotate(name, (i, acc * 100),
                               fontsize=11.5, alpha=0.85, textcoords="offset points", xytext=(5, 3))
            seen_modes_closed.add(mode)
        ax_closed.set_xticks([])
        ax_closed.set_xlabel("(size undisclosed)", fontsize=13)
        ax_closed.set_title("Closed models", fontsize=16)
        ax_closed.grid(True, alpha=0.3, axis="y")
        ax_closed.tick_params(labelsize=13)
        ax_closed.legend(title="Mode", fontsize=13, title_fontsize=13)

    all_accs = [a for _, _, a in open_models] + [a for _, a in closed_models]
    if all_accs:
        lo = 0
        hi = 100
        for ax in ([ax_open, axes[1]] if has_closed else [ax_open]):
            ax.set_ylim(lo, hi)

    #fig.suptitle("ToM Reasoning Benchmark — Scale vs. Accuracy", fontsize=17)
    plt.tight_layout()

    save_path = os.path.join(results_dir, "scaling_plot.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODELS), metavar="MODEL")
    parser.add_argument("--all", action="store_true", help="Run all registered models")
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
    parser.add_argument("--tensor_parallel_size", type=int, default=None,
                        help="Override tensor_parallel_size for vLLM (default: per-model registry value)")
    parser.add_argument("--api_concurrency", type=int, default=20,
                        help="Max concurrent requests for API models (default: 20)")
    parser.add_argument("--latex", action="store_true",
                        help="Print LaTeX table from existing result files and exit")
    args = parser.parse_args()

    # ── Standalone LaTeX table from existing result files ──────────────────
    if args.latex:
        dataset_stem = os.path.splitext(os.path.basename(args.tomstories_vignettes))[0]
        with open(args.tomstories_vignettes) as f:
            raw = json.load(f)
        valid_sids = {k for k, v in raw.items()
                      if args.split == "all" or v.get("split") == args.split}

        all_metrics = {}
        for mode in ("mc", "oe"):
            pattern = os.path.join(args.results_dir, f"*_{dataset_stem}_{mode}_results.json")
            for fpath in sorted(glob.glob(pattern)):
                stem = os.path.basename(fpath)
                model_name = stem[: stem.index(f"_{dataset_stem}")]
                results  = json.load(open(fpath))
                filtered = {sid: e for sid, e in results.items() if sid in valid_sids}
                if not filtered:
                    continue
                key = f"{model_name} [{mode}]"
                all_metrics[key] = compute_metrics(filtered)
                all_metrics[key]["_model_name"] = model_name

        if not all_metrics:
            print(f"No result files found in {args.results_dir}/ for dataset '{dataset_stem}'.")
            return
        print_latex_table(all_metrics, dataset_path=args.tomstories_vignettes, split=args.split)
        return

    if args.all:
        model_names = list(MODELS)
    elif args.models:
        model_names = args.models
    else:
        parser.error("Specify --models, --all, or --latex")

    modes = ["mc", "oe"] if args.mode == "both" else [args.mode]

    os.makedirs(args.results_dir, exist_ok=True)

    with open(args.tomstories_vignettes) as f:
        raw = json.load(f)

    samples = list(raw.items())
    if args.split != "all":
        samples = [(k, v) for k, v in samples if v["split"] == args.split]
    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"Evaluating {len(samples)} samples | models: {len(model_names)} | mode(s): {modes}")

    dataset_stem = os.path.splitext(os.path.basename(args.tomstories_vignettes))[0]

    all_metrics = {}
    for mode in modes:
        print(f"\n{'='*60}\nMode: {mode.upper()}\n{'='*60}")
        for model_name in model_names:
            output_path = os.path.join(
                args.results_dir, f"{model_name}_{dataset_stem}_{mode}_results.json"
            )
            results = benchmark(
                model_name, samples, output_path,
                mode=mode, oe_scorer=args.oe_scorer, judge_model=args.judge_model,
                tensor_parallel_size=args.tensor_parallel_size,
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
    print_latex_table(all_metrics, dataset_path=args.tomstories_vignettes, split=args.split)
    plot_results(all_metrics, args.results_dir)


if __name__ == "__main__":
    main()
