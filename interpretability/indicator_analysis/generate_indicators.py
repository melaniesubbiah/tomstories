"""
Ask each model for subtle indicator phrases that would show a wound, fear, motivation or
emotion without naming it, for a sample of thesaurus entries.

Outputs:
  <output>   {category: {item: {model: [phrases]}}}, rewritten after each model
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root, for `common`
from common.models import ALL_MODELS as MODELS
from common.models import MODELS as LOCAL_MODEL_REGISTRY
from common.models import add_model_args, selected_models
from common.prompts import chat_messages

LOCAL_MODELS = list(LOCAL_MODEL_REGISTRY)


# --------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a writing craft assistant helping novelists with characterization. "
    "You generate lists of SUBTLE, SHOWABLE behavioral or physical cues — "
    "things a reader could observe in a scene — that hint a character is "
    "privately carrying a particular emotion, wound, fear, or motivation, "
    "WITHOUT the narration ever naming or stating it directly. "
    "Think body language, habits, speech patterns, small avoidances, "
    "physiological tells, and indirect behaviors. "
    "Avoid cliches where possible and keep each cue concrete and specific "
    "enough to write into a scene."
)

CATEGORY_LABELS = {
    "wounds": "past wound / core trauma",
    "fears": "fear",
    "motivations": "motivation / goal",
    "emotions": "emotion",
}


def build_user_prompt(category: str, item: str, n: int) -> str:
    label = CATEGORY_LABELS.get(category, category)
    return (
        f'Category: {label}\n'
        f'Item: "{item}"\n\n'
        f"List exactly {n} distinct, concrete, SHOW-DON'T-TELL behavioral or "
        f"physical cues a character might display if they are experiencing "
        f'"{item}" but are NOT talking about it openly. '
        f"Return ONLY a JSON array of {n} short strings, nothing else — "
        f"no numbering, no preamble, no explanation."
    )


def build_messages(user_prompt: str, system_ok: bool = True) -> List[Dict[str, str]]:
    return chat_messages(SYSTEM_PROMPT, user_prompt, system_ok)


# --------------------------------------------------------------------------
# Output parsing
# --------------------------------------------------------------------------

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning_trace(text: str) -> str:
    return THINK_BLOCK_RE.sub("", text).strip()


def extract_json_array(text: str) -> List[str]:
    text = strip_reasoning_trace(text).strip()

    if "```" in text:
        for part in text.split("```"):
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[len("json"):].strip()
            if candidate.startswith("["):
                text = candidate
                break

    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
        pass

    # Fallback: treat as a plain/numbered list.
    lines = [ln.strip(" -*\t") for ln in text.splitlines()]
    lines = [ln.lstrip("0123456789.) ") for ln in lines if ln.strip()]
    return lines


# --------------------------------------------------------------------------
# Task construction
# --------------------------------------------------------------------------

def sample_items(items: List[str], k: int, seed: int, category: str) -> List[str]:
    if k >= len(items):
        return list(items)
    rng = random.Random(f"{seed}:{category}")
    order = list(range(len(items)))
    rng.shuffle(order)
    return [items[i] for i in sorted(order[:k])]


def build_tasks(
    taxonomy: Dict[str, List[str]],
    categories: List[str],
    limit_per_category: Optional[int],
    seed: int,
) -> List[Dict[str, str]]:
    tasks = []
    for category in categories:
        items = taxonomy.get(category)
        if items is None:
            print(f"  [skip] category '{category}' not found in input JSON", file=sys.stderr)
            continue
        if limit_per_category:
            items = sample_items(items, limit_per_category, seed, category)
            print(f"  {category}: randomly selected {len(items)} of "
                  f"{len(taxonomy[category])} items (seed={seed})")
        for item in items:
            tasks.append({"category": category, "item": item})
    return tasks


# --------------------------------------------------------------------------
# VLLM generation for a single model
# --------------------------------------------------------------------------

def run_model(
    model_key: str,
    cfg: Dict[str, Any],
    tasks: List[Dict[str, str]],
    per_item: int,
    max_tokens: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
) -> Dict[str, List[str]]:
    from vllm import SamplingParams

    # imported here so the API-only path does not need torch/transformers installed
    from common.hf import destroy_vllm, load_vllm, supports_system_role

    print(f"\n=== Loading {model_key} ({cfg['hf_id']}, {cfg['params_b']}B, "
          f"tp={cfg.get('tp', 1)}) ===")
    llm = load_vllm(cfg, gpu_memory_utilization=gpu_memory_utilization,
                    max_model_len=max_model_len, trust_remote_code=True)
    try:
        system_ok = supports_system_role(llm.get_tokenizer())
        if not system_ok:
            print("  Note: chat template has no system role — "
                  "prepending the system prompt to the user message")

        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_tokens)
        conversations = [
            build_messages(build_user_prompt(t["category"], t["item"], per_item), system_ok)
            for t in tasks
        ]

        t0 = time.time()
        outputs = llm.chat(conversations, sampling_params)
        elapsed = time.time() - t0
        print(f"  generated {len(outputs)} completions in {elapsed:.1f}s "
              f"({elapsed / max(len(outputs), 1):.2f}s/item)")

        results: Dict[str, List[str]] = {}
        n_empty = 0
        for task, out in zip(tasks, outputs):
            phrases = extract_json_array(out.outputs[0].text)
            if not phrases:
                n_empty += 1
            results[f'{task["category"]}::{task["item"]}'] = phrases
        if n_empty:
            print(f"  [warn] {n_empty}/{len(tasks)} items produced no parseable phrases",
                  file=sys.stderr)
    finally:
        destroy_vllm(llm)

    return results


# --------------------------------------------------------------------------
# API generation (OpenAI / Anthropic)
# --------------------------------------------------------------------------

def run_api_model(
    model_key: str,
    cfg: Dict[str, Any],
    tasks: List[Dict[str, str]],
    per_item: int,
    max_tokens: int,
    concurrency: int,
    timeout: float = 120.0,
) -> Dict[str, List[str]]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    provider, api_id = cfg["provider"], cfg["api_id"]
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(max_retries=5, timeout=timeout)          # reads OPENAI_API_KEY

        def call(user_prompt: str) -> str:
            resp = client.chat.completions.create(
                model=api_id,
                messages=build_messages(user_prompt),
                temperature=0.8, top_p=0.95, max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(max_retries=5, timeout=timeout)   # reads ANTHROPIC_API_KEY

        def call(user_prompt: str) -> str:
            # Recent Claude models reject setting temperature and top_p together,
            # so only temperature is set here.
            resp = client.messages.create(
                model=api_id,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.8, max_tokens=max_tokens,
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    else:
        raise ValueError(f"unknown provider {provider!r} for {model_key}")

    print(f"\n=== Calling {model_key} ({provider}: {api_id}), "
          f"{len(tasks)} prompts, concurrency={concurrency}, timeout={timeout:.0f}s ===",
          flush=True)

    results: Dict[str, List[str]] = {}
    n_empty, errors = 0, []
    report_every = max(1, min(10, len(tasks) // 10))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(call, build_user_prompt(t["category"], t["item"], per_item)): t
            for t in tasks
        }
        for i, fut in enumerate(as_completed(futures), 1):
            task = futures[fut]
            try:
                phrases = extract_json_array(fut.result())
            except Exception as e:
                errors.append((task, e))
                if len(errors) == 1:   # surface the first failure right away (auth, model id, network…)
                    print(f"  [warn] request failed for {task['category']}::{task['item']}: {e!r}",
                          file=sys.stderr, flush=True)
            else:
                if not phrases:
                    n_empty += 1
                results[f'{task["category"]}::{task["item"]}'] = phrases
            if i % report_every == 0 or i == len(tasks):
                elapsed = time.time() - t0
                eta = elapsed / i * (len(tasks) - i)
                print(f"  {i}/{len(tasks)} done, {len(errors)} failed "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s left)", flush=True)

    if n_empty:
        print(f"  [warn] {n_empty}/{len(tasks)} items produced no parseable phrases", file=sys.stderr)
    if errors:
        print(f"  [warn] {len(errors)}/{len(tasks)} requests failed and were skipped "
              f"(rerun to retry them). First error: {errors[0][1]!r}", file=sys.stderr)
    return results


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate implicit indicator phrases for wounds/fears/"
                    "motivations/emotions using local vLLM models."
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="Path to the taxonomy JSON file.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Path to write the generated phrases JSON (updated after each model).")
    parser.add_argument("--categories", nargs="+",
                        default=["wounds", "fears", "motivations", "emotions"],
                        help="Which top-level keys in the input JSON to process.")
    add_model_args(
        parser, registry=MODELS, default=LOCAL_MODELS,
        help="Which model keys (from the MODELS roster) to run. Defaults to "
             "all local models; API models (gpt-4o-mini, claude-sonnet-4.6) "
             "run only when named here.",
        all_help="Run all local models (the default); API models stay opt-in via --models",
    )
    parser.add_argument("--per-item", type=int, default=6,
                        help="How many phrases to request per item per model.")
    parser.add_argument("--max-tokens", type=int, default=400,
                        help="Max new tokens per generation (raise for reasoning models).")
    parser.add_argument("--limit-per-category", type=int, default=None,
                        help="Optional cap on items per category. Items are chosen at random "
                             "(seeded by --seed), and every model gets the same selection.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the --limit-per-category random selection. Keep it "
                             "fixed across runs so resumed/added models see the same items.")
    parser.add_argument("--api-concurrency", type=int, default=8,
                        help="Parallel requests for API models (lower it if you hit rate limits).")
    parser.add_argument("--api-timeout", type=float, default=120.0,
                        help="Per-request timeout in seconds for API models (each request is "
                             "retried up to 5 times on timeouts, rate limits and server errors).")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Override vLLM's max_model_len (useful to shrink KV cache footprint).")

    args = parser.parse_args()
    model_names = selected_models(args, parser, registry=LOCAL_MODEL_REGISTRY)

    taxonomy = json.loads(args.input.read_text(encoding="utf-8"))
    tasks = build_tasks(taxonomy, args.categories, args.limit_per_category, args.seed)
    print(f"Built {len(tasks)} (category, item) tasks across categories: {args.categories}")

    all_results: Dict[str, Any] = {}
    if args.output.exists():
        try:
            all_results = json.loads(args.output.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[warn] {args.output} is not valid JSON; starting fresh "
                  f"(it will be overwritten)", file=sys.stderr)

    for model_key in model_names:
        # Only this run's tasks that the model hasn't answered yet
        todo = [
            t for t in tasks
            if model_key not in all_results.get(t["category"], {}).get(t["item"], {})
        ]
        if not todo:
            print(f"[skip] {model_key} already has results for all {len(tasks)} items")
            continue
        if len(todo) < len(tasks):
            print(f"[resume] {model_key}: {len(tasks) - len(todo)} items done, "
                  f"generating the remaining {len(todo)}")

        cfg = MODELS[model_key]
        try:
            if "provider" in cfg:
                per_task_results = run_api_model(
                    model_key=model_key,
                    cfg=cfg,
                    tasks=todo,
                    per_item=args.per_item,
                    max_tokens=args.max_tokens,
                    concurrency=args.api_concurrency,
                    timeout=args.api_timeout,
                )
            else:
                per_task_results = run_model(
                    model_key=model_key,
                    cfg=cfg,
                    tasks=todo,
                    per_item=args.per_item,
                    max_tokens=args.max_tokens,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_model_len=args.max_model_len,
                )
        except Exception as e:
            print(f"[error] {model_key} failed: {e}", file=sys.stderr)
            continue

        for flat_key, phrases in per_task_results.items():
            category, item = flat_key.split("::", 1)
            all_results.setdefault(category, {}).setdefault(item, {})
            all_results[category][item][model_key] = phrases

        args.output.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  saved progress to {args.output}")

    print(f"\nDone. Final results in {args.output}")


if __name__ == "__main__":
    main()