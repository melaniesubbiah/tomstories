"""
Find the largest steering coefficient that keeps generations coherent: sweep alpha at the
layer linear_probes.py chose, score each completion's coherence with an LLM judge, and keep
the largest alpha within --max_drop of the unsteered baseline. Reports one coefficient per
direction plus a shared magnitude; the sweep itself is fixed by the constants below.

Outputs:
  <output_dir>/<model>_steer_cache.json       generations and judge scores
  <output_dir>/steer_coherence_summary.json   chosen coefficients and the coherence curve
"""

import argparse
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root, for `common`
from common.data import load_best_layer
from common.hf import RowSteerer, free_cuda, layer_module, load_model, load_unit_vector
from common.models import MODELS, add_model_args, selected_models

# ── Fixed experiment settings ─────────────────────────────────────────────────

# Where extract_vectors.py writes each layer's vectors, and which one to steer with
# ("overall" is the implicit-minus-explicit direction averaged over question types;
# steer_vignettes.py defaults to the same key).
VECTOR_TEMPLATE = "{dir}/{model}_layer{layer}_vectors.pt"
VECTOR_KEY      = "overall"

SIGNS        = [1, -1]   # both directions, so a shared magnitude can be reported
ALPHA_GRID   = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]  # fractions of the median residual norm
REFINE_STEPS = 3         # bisection steps after the coarse grid
MAX_NEW_TOKENS = 200     # matched to the "about 150 words" in the prompts below

# Open-ended writing prompts (edit this list to use different ones)
PROMPTS = [
    "Write a short scene (about 150 words) in which a woman returns to her childhood home after many years.",
    "Write a short scene (about 150 words) in which two old friends run into each other at a train station.",
    "Write a short scene (about 150 words) about a man cooking dinner for his family on a weeknight.",
    "Write a short scene (about 150 words) in which a teenager waits for a phone call.",
    "Write a short scene (about 150 words) set in a small-town diner just before closing.",
    "Write a short scene (about 150 words) in which a father teaches his daughter to ride a bike.",
    "Write a short scene (about 150 words) about a retired teacher packing up her classroom.",
    "Write a short scene (about 150 words) in which a nurse finishes a long night shift.",
    "Write a short scene (about 150 words) in which two siblings sort through their late grandmother's belongings.",
    "Write a short scene (about 150 words) about a new employee's first morning at an office.",
    "Write a short scene (about 150 words) in which a fisherman heads out before dawn.",
    "Write a short scene (about 150 words) in which a couple waits at an airport gate.",
]

JUDGE_SYSTEM = (
    "You rate the COHERENCE of text written by a language model. Judge only whether "
    "the text is fluent, grammatical, readable and internally consistent, and free of "
    "degenerate output (repetition loops, gibberish, broken or invented words, abrupt "
    "language switches, trailing nonsense). Do NOT judge relevance to the request, "
    "style, creativity, or storytelling quality. Text that is cut off at the end because "
    "of a length limit is fine.\n"
    "Scale: 10 = perfectly fluent and coherent; 8 = coherent with minor awkwardness; "
    "6 = understandable but clearly odd phrasing or some repetition; 4 = substantially "
    "broken or repetitive, only partly readable; 2 = mostly incoherent; 1 = gibberish or empty.\n"
    'Respond with JSON only: {"score": <integer 1-10>}'
)


# ── Model session: loaded lazily, only if there is something to generate ────

class ModelSession:
    def __init__(self, cfg, layer, vec_path, batch_size):
        self.cfg, self.layer = cfg, layer
        self.vec_path  = vec_path
        self.batch_size = batch_size
        self.model = None
        self.steerer = None

    def _load(self):
        if self.model is not None:
            return
        hf_id = self.cfg["hf_id"]
        # fail fast, before loading the model
        self.vec = load_unit_vector(self.vec_path, VECTOR_KEY)
        print(f"  loading {hf_id} …", flush=True)
        self.model, self.tok = load_model(self.cfg)
        self.device = self.model.get_input_embeddings().weight.device
        if self.vec.numel() != self.model.config.hidden_size:
            raise ValueError(f"vector dim {self.vec.numel()} != hidden size "
                             f"{self.model.config.hidden_size} ({self.vec_path})")
        self.steerer = RowSteerer(self.model, self.layer, self.vec)

    def format_prompt(self, prompt):
        # User turn only (works for templates without a system role, e.g. Gemma-2);
        # the template adds BOS, so tokenization below skips special tokens.
        return self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

    def median_resid_norm(self, prompts):
        """Median L2 norm of layer-L residuals over prompt tokens (first token excluded:
        attention-sink tokens have outsized norms)."""
        import torch
        self._load()
        norms, grabbed = [], []
        h = layer_module(self.model, self.layer).register_forward_hook(
            lambda m, i, o: grabbed.append((o[0] if isinstance(o, tuple) else o)
                                           .detach().float().norm(dim=-1).cpu()))
        try:
            texts = [self.format_prompt(p) for p in prompts]
            for i in range(0, len(texts), self.batch_size):
                enc = self.tok(texts[i:i + self.batch_size], return_tensors="pt",
                               padding=True, add_special_tokens=False).to(self.device)
                with torch.no_grad():
                    self.model(**enc)
                n, mask = grabbed.pop(), enc["attention_mask"].cpu().bool()
                first = mask.float().argmax(dim=1)
                mask[torch.arange(len(first)), first] = False
                norms.append(n[mask])
        finally:
            h.remove()
        return float(torch.cat(norms).median())

    def generate(self, prompts, coeffs):
        """Greedy generation; row i is steered with coeffs[i]."""
        import torch
        self._load()
        texts = [self.format_prompt(p) for p in prompts]
        outs = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tok(texts[i:i + self.batch_size], return_tensors="pt",
                           padding=True, add_special_tokens=False).to(self.device)
            self.steerer.coeffs = torch.tensor(coeffs[i:i + self.batch_size],
                                               dtype=torch.float32)
            try:
                with torch.no_grad():
                    out = self.model.generate(
                        **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                        pad_token_id=self.tok.pad_token_id,
                    )
            finally:
                self.steerer.coeffs = None
            outs += self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
            print(f"    generated {min(i + self.batch_size, len(texts))}/{len(texts)}", flush=True)
        return outs

    def close(self):
        if self.model is None:
            return
        if self.steerer is not None:
            self.steerer.remove()
            self.steerer = None
        del self.model
        self.model = None
        free_cuda()


# ── Judge ─────────────────────────────────────────────────────────────────────

def parse_score(text):
    try:
        s = int(json.loads(text)["score"])
    except Exception:
        m = re.search(r"\b(10|[1-9])\b", text or "")
        if not m:
            return None
        s = int(m.group(1))
    return min(10, max(1, s))


def judge_all(items, judge_model, concurrency):
    """items: {hash: (prompt, completion)} -> {hash: score or None}."""
    if not items:
        return {}
    from openai import OpenAI
    client = OpenAI(max_retries=5, timeout=60)

    def call(prompt, completion):
        if not completion.strip():
            return 1
        resp = client.chat.completions.create(
            model=judge_model, temperature=0, max_tokens=20,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": JUDGE_SYSTEM},
                      {"role": "user", "content":
                       f"Request given to the model:\n{prompt}\n\nModel output:\n{completion}"}],
        )
        return parse_score(resp.choices[0].message.content)

    out, errors = {}, 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(call, p, c): h for h, (p, c) in items.items()}
        for fut in as_completed(futs):
            try:
                out[futs[fut]] = fut.result()
            except Exception as e:
                errors += 1
                if errors == 1:
                    print(f"    [warn] judge call failed: {e!r}", file=sys.stderr, flush=True)
    if errors:
        print(f"    [warn] {errors} judge calls failed; rerun to retry them", file=sys.stderr)
    return out


# ── Sweep logic ───────────────────────────────────────────────────────────────

def _h(*parts):
    return hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:16]


def choose_alpha(curve, threshold):
    """Scan alphas upward; return (largest passing alpha before the first failure,
    first failing alpha or None). Alpha 0 always counts as passing."""
    passing, first_fail = 0.0, None
    for a in sorted(curve):
        if a == 0:
            continue
        if curve[a] is not None and curve[a] >= threshold:
            passing = a
        else:
            first_fail = a
            break
    return passing, first_fail


def evaluate(points, prompts, cache, session, judge_model, judge_concurrency):
    """
    points: list of (sign, alpha). Generates/judges whatever is missing from the
    cache and returns {(sign, alpha): mean coherence}.
    """
    norm = cache["resid_norm"]
    jobs = [(s, a, pi) for s, a in points for pi in range(len(prompts))]
    # alpha 0 is the same unsteered generation for both signs: share one key
    gkey = lambda s, a, pi: f"{s if a else 1:+d}|{a:.4f}|{_h(prompts[pi])}"

    todo, seen = [], set()
    for j in jobs:
        k = gkey(*j)
        if k not in cache["gens"] and k not in seen:
            seen.add(k)
            todo.append(j)
    if todo:
        print(f"  generating {len(todo)} completions "
              f"({len({(s, a) for s, a, _ in todo})} coefficient settings) …", flush=True)
        texts = session.generate([prompts[pi] for _, _, pi in todo],
                                 [s * a * norm for s, a, _ in todo])
        for j, t in zip(todo, texts):
            cache["gens"][gkey(*j)] = t

    to_judge = {}
    for j in jobs:
        comp = cache["gens"][gkey(*j)]
        jh = _h(prompts[j[2]], comp)
        if cache["scores"].get(jh) is None:
            to_judge[jh] = (prompts[j[2]], comp)
    if to_judge:
        print(f"  judging {len(to_judge)} completions with {judge_model} …", flush=True)
        cache["scores"].update(judge_all(to_judge, judge_model, judge_concurrency))

    means = {}
    for s, a in points:
        sc = [cache["scores"].get(_h(prompts[pi], cache["gens"][gkey(s, a, pi)]))
              for pi in range(len(prompts))]
        sc = [x for x in sc if x is not None]
        means[(s, a)] = float(np.mean(sc)) if sc else None
    return means


def sweep_model(name, args, prompts):
    cfg = MODELS[name]
    selected = load_best_layer(args.probe_results_dir, name)
    layer, n_layers = selected["layer"], selected["n_layers"]

    vec_path = Path(VECTOR_TEMPLATE.format(dir=args.vectors_dir, model=name, layer=layer))
    if not vec_path.exists():
        raise FileNotFoundError(
            f"steering vector not found: {vec_path}\nRun:\n"
            f"    python interpretability/steering/extract_vectors.py --models {name}"
        )

    load_unit_vector(vec_path, VECTOR_KEY)   # validate the file before any model work

    settings = {"layer": layer, "vector": str(vec_path), "key": VECTOR_KEY,
                "max_new_tokens": MAX_NEW_TOKENS, "alpha_grid": ALPHA_GRID,
                "signs": SIGNS}
    cache_path = args.output_dir / f"{name}_steer_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if cache.get("settings") != settings:
        if cache:
            print(f"  settings changed since last run; discarding cached generations")
        cache = {"settings": settings, "resid_norm": None, "gens": {},
                 "scores": cache.get("scores", {})}          # scores are keyed by text; safe to keep

    print(f"\n=== {name}: layer {layer}/{n_layers - 1} "
          f"({100 * layer / (n_layers - 1):.0f}% depth), vector {vec_path.name} ===", flush=True)

    session = ModelSession(cfg, layer, vec_path, args.batch_size)
    save = lambda: cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
    try:
        if cache["resid_norm"] is None:
            cache["resid_norm"] = session.median_resid_norm(prompts)
            save()
        print(f"  median residual norm at layer {layer}: {cache['resid_norm']:.2f}")

        grid  = sorted(set([0.0] + ALPHA_GRID))
        curve = {s: {} for s in SIGNS}
        pts   = [(s, a) for s in SIGNS for a in grid]
        res = evaluate(pts, prompts, cache, session, args.judge_model, args.judge_concurrency)
        save()
        for (s, a), m in res.items():
            curve[s][a] = m
        baseline = res[(SIGNS[0], 0.0)]
        if baseline is None:
            raise RuntimeError("no judge scores for the unsteered baseline")
        threshold = baseline - args.max_drop

        # Bisection between the last passing and first failing alpha (per sign, batched)
        for _ in range(REFINE_STEPS):
            mids = []
            for s in SIGNS:
                ok, bad = choose_alpha(curve[s], threshold)
                if bad is not None and bad - ok > 1e-3:
                    mids.append((s, round((ok + bad) / 2, 4)))
            if not mids:
                break
            res = evaluate(mids, prompts, cache, session, args.judge_model, args.judge_concurrency)
            save()
            for (s, a), m in res.items():
                curve[s][a] = m

        chosen = {}
        for s in SIGNS:
            ok, bad = choose_alpha(curve[s], threshold)
            chosen[s] = {"alpha": ok, "coeff": s * ok * cache["resid_norm"],
                         "coherence": curve[s].get(ok), "first_failing_alpha": bad}

        # Shared magnitude: the smaller per-direction alpha, checked in both directions
        a_sh    = min(c["alpha"] for c in chosen.values())
        missing = [(s, a_sh) for s in SIGNS if a_sh not in curve[s]]
        if missing:
            res = evaluate(missing, prompts, cache, session, args.judge_model,
                           args.judge_concurrency)
            save()
            for (s, a), m in res.items():
                curve[s][a] = m

        coh = {s: curve[s].get(a_sh) for s in SIGNS}
        shared = {"alpha": a_sh, "coeff": a_sh * cache["resid_norm"],
                  "coherence": {str(s): c for s, c in coh.items()},
                  "passes_both": all(c is not None and c >= threshold for c in coh.values())}
        if not shared["passes_both"]:
            print(f"  [warn] shared alpha {a_sh:.3f} misses the threshold in one direction "
                  f"(non-monotone curve); inspect the cached generations")
    finally:
        session.close()

    return {
        "layer": layer, "n_layers": n_layers, "resid_norm": cache["resid_norm"],
        "vector": str(vec_path), "vector_key": VECTOR_KEY,
        "baseline": baseline, "threshold": threshold,
        "curve": {str(s): {f"{a:.4f}": m for a, m in sorted(curve[s].items())}
                  for s in SIGNS},
        "chosen": {str(s): v for s, v in chosen.items()},
        "shared": shared,
    }


def print_summary(summary):
    print("\n" + "=" * 86)
    print(f"{'model':<16} {'layer':>9} {'baseline':>9} {'sign':>5} {'alpha':>7} "
          f"{'coherence':>10} {'coeff':>9} {'next fails at':>14}")
    print("-" * 86)
    for name, r in summary.items():
        if "error" in r:
            print(f"{name:<16}  ERROR: {r['error']}")
            continue
        for s, c in r["chosen"].items():
            coh = "--" if c["coherence"] is None else f"{c['coherence']:.2f}"
            nxt = "--" if c["first_failing_alpha"] is None else f"{c['first_failing_alpha']:.3f}"
            print(f"{name:<16} {r['layer']:>4}/{r['n_layers'] - 1:<4} {r['baseline']:>9.2f} "
                  f"{int(s):>+5d} {c['alpha']:>7.3f} {coh:>10} {c['coeff']:>9.2f} {nxt:>14}")
        sh = r.get("shared")
        if sh:
            coh = "/".join("--" if v is None else f"{v:.2f}" for v in sh["coherence"].values())
            flag = "" if sh["passes_both"] else "  (!)"
            print(f"{'':<16} {'':>9} {'':>9} {'±':>5} {sh['alpha']:>7.3f} {coh:>10} "
                  f"{sh['coeff']:>9.2f} {'shared' + flag:>14}")
    print("=" * 86)
    print("alpha = fraction of the median residual norm at the steering layer; "
          "coeff = raw multiplier on the unit vector.\n"
          "± row = shared alpha (smaller of the two directions); coherence shown as +/-.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p)
    p.add_argument("--probe_results_dir", type=Path, default=Path("results/probes"),
                   help="Directory holding <model>_best_layers.json from "
                        "linear_probes.py, which decides the steering layer")
    p.add_argument("--vectors_dir", type=Path, default=Path("results/steering"),
                   help="Directory holding the vector files from extract_vectors.py")
    p.add_argument("--max_drop", type=float, default=1.0,
                   help="Allowed drop in mean coherence below the unsteered baseline (1-10 scale)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--judge_model", default="gpt-4o-mini")
    p.add_argument("--judge_concurrency", type=int, default=16)
    p.add_argument("--output_dir", type=Path, default=Path("results/steering_coherence"))
    args = p.parse_args()
    model_names = selected_models(args, p)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(model_names)} model(s) x {len(PROMPTS)} prompts x "
          f"{len(ALPHA_GRID) + 1} alphas x {len(SIGNS)} sign(s)")

    summary_path = args.output_dir / "steer_coherence_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for name in model_names:
        t0 = time.time()
        try:
            summary[name] = sweep_model(name, args, PROMPTS)
        except Exception as e:
            print(f"[error] {name}: {e!r}", file=sys.stderr)
            summary[name] = {"error": repr(e)}
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"  done in {time.time() - t0:.0f}s")

    print_summary({m: summary[m] for m in model_names})
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()