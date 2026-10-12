"""
Clean the generated cues, embed and cluster them by behavior phrase, and measure how much
cue vocabulary the other models share with the reference model.

Outputs:
  <out>/cues_clustered.csv                 every cue with its cluster
  <out>/clusters.csv                       cluster sizes, models and an example cue
  <out>/summary.json                       sharing and reference-overlap statistics
  <out>/state_overlap_top<N>.{json,csv}    per-state overlap with the reference model
  <out>/model_overlap_top<N>[_frac].json   per-model overlap with the reference model
"""
import argparse
import collections
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

LEVELS = ("question", "type", "overall")        # same psychological aspect / same aspect type / any aspect

# =============================================================================
# 1. Cleaning
# =============================================================================
_KEY_DEBRIS_RE = re.compile(r'^\s*"?\s*:\s*"?')
_DICT_KEY_RE = re.compile(r"""^\{?\s*['"]?[\w ]{1,30}['"]\s*:\s*['"]?""")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_REASONING_RE = re.compile(
    r"^(okay\b|ok,|alright\b|hmm\b|wait[,.!]|let me\b|i need to\b|i should\b|first,|"
    r"so,|the user\b|here are\b|here is\b|sure[,!])", re.IGNORECASE)
_PUNCT_ONLY_RE = re.compile(r"^[\W_]*$")
MIN_LEN, MAX_LEN = 4, 300


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def _strip_debris(s):
    s = s.strip()
    s = _KEY_DEBRIS_RE.sub("", s) if s.startswith(('":', '" :', ':')) else s
    if s.startswith("{") or re.match(r"""^[\w ]{1,30}['"]\s*:""", s):
        s = _DICT_KEY_RE.sub("", s)
        s = s.rstrip("}").strip().rstrip("'\"").strip()
    s = s.strip().strip("[]{}").strip()
    s = s.rstrip(",").strip().rstrip(':').rstrip('"').strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1]
    s = s.strip().lstrip('"').rstrip('"').rstrip(",").strip()
    s = s.strip("'").strip() if s.count("'") == 1 or (s[:1] == "'" and s[-1:] == "'") else s
    s = _BULLET_RE.sub("", s)
    s = s.replace('\\"', '"').replace("\\'", "'")
    return s.strip()


def _explode(raw):
    """Split a 'cue' that is really several cues into its items."""
    s = raw.strip()
    if s.startswith("["):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass
    if len(re.findall(r'"\s*,\s*"', s)) >= 2:                  # malformed / truncated arrays
        return re.split(r'"\s*,\s*"', s.strip("[]{} "))
    segs = [x.strip() for x in s.split(",")]                    # comma-joined list of short cues
    if len(segs) >= 6 and all(0 < len(x.split()) <= 12 for x in segs):
        return segs
    return [s]


def clean_cues(cues, log):
    out, seen = [], set()
    for raw in cues:
        parts = _explode(raw if isinstance(raw, str) else str(raw))
        if len(parts) > 1:
            log["split_lists"] += 1
        for p in parts:
            c = _strip_debris(p)
            if c != p.strip():
                log["stripped_debris"] += 1
            if (_PUNCT_ONLY_RE.match(c) or len(c) < MIN_LEN or c.startswith("```")
                    or c.lower() in ("json", "```json")):
                log["dropped_empty"] += 1
                continue
            if len(c) > MAX_LEN or _REASONING_RE.match(c):
                log["dropped_reasoning_or_long"] += 1
                continue
            key = normalize(c)
            if key in seen:
                log["dropped_duplicate"] += 1
                continue
            seen.add(key)
            out.append(c)
    return out


# =============================================================================
# 2. Behavior phrase + embedding
# =============================================================================
_SUB = (r"(?:even )?(?:when|whenever|while|during|before|after|if|as if|as though|especially|"
        r"every time|each time|until|because|though|rather than|without)")
_LEAD_SUB_RE = re.compile(rf"^\s*{_SUB}\b[^,]*,\s*", re.I)
_HARD_RE = re.compile(rf"\s*(?:;|:|—|–|\s-\s|\b{_SUB}\b)\s*", re.I)
_COMMA_RE = re.compile(
    r",\s*(?=(?:and |then |(?:(?:their|her|his|the) )?(?:eyes|hands?|voice|gaze|jaw|shoulders|face|"
    r"smile|body|breath)\b|\w+ing\s+(?:their|his|her|the|a|an|its|to|at|up|down|back|as|over|into|"
    r"out|away|with|from|for|on)\b))", re.I)
_SUBJ_RE = re.compile(r"^(?:they|she|he|the character)\s+(?=\w)", re.I)


def behavior_phrase(cue):
    """'Avoids eye contact during talks about family' -> 'Avoids eye contact'."""
    s = _LEAD_SUB_RE.sub("", cue.strip().strip('"').strip())
    head = _COMMA_RE.split(_HARD_RE.split(s)[0])[0].strip().rstrip(".,")
    head = _SUBJ_RE.sub("", head)
    return head if len(head.split()) >= 2 else s


def embed(texts, model_name):
    from sentence_transformers import SentenceTransformer
    X = SentenceTransformer(model_name).encode(texts, batch_size=256, normalize_embeddings=True,
                                               show_progress_bar=True)
    return np.asarray(X, dtype=np.float32)


# =============================================================================
# 4. Analysis helpers
# =============================================================================
def pct(x):
    return round(100 * float(np.mean(x)), 2) if len(x) else None


def sharing_stats(rows, key):
    """For each cue: how many OTHER models have a cue in the same cluster within `key`."""
    pool = collections.defaultdict(set)
    for r in rows:
        pool[key(r)].add(r["model"])
    k = np.array([len(pool[key(r)] - {r["model"]}) for r in rows])
    return dict(shared_with_1plus_pct=pct(k >= 1), shared_with_3plus_pct=pct(k >= 3),
                shared_with_7plus_pct=pct(k >= 7), median_other_models=float(np.median(k)),
                mean_other_models=round(float(k.mean()), 3))


def reference_overlap(rows, ref, opens):
    """Per state and per open model overlap with the reference model's cues."""
    op = [r for r in rows if r["model"] != ref]
    by = {lv: collections.defaultdict(set) for lv in LEVELS}             # cluster pool -> models
    for r in op:
        by["question"][(r["category"], r["item"], r["cluster"])].add(r["model"])
        by["type"][(r["category"], r["cluster"])].add(r["model"])
        by["overall"][r["cluster"]].add(r["model"])
    keyf = {"question": lambda r: (r["category"], r["item"], r["cluster"]),
            "type": lambda r: (r["category"], r["cluster"]),
            "overall": lambda r: r["cluster"]}

    ref_cues = collections.defaultdict(list)                          # state -> ref cue clusters
    per_state = collections.defaultdict(lambda: {lv: [] for lv in LEVELS})
    for r in rows:
        if r["model"] != ref:
            continue
        q = (r["category"], r["item"])
        ref_cues[q].append(r["cluster"])
        for lv in LEVELS:
            per_state[q][lv].append(len(by[lv][keyf[lv](r)]))
    n_open = len(opens)
    states = sorted(per_state)
    state_rows, dist = [], {}
    for q in states:
        state_rows.append(dict(category=q[0], item=q[1], n_ref_cues=len(ref_cues[q]),
                               **{f"best_{lv}": max(per_state[q][lv]) for lv in LEVELS},
                               **{f"unmatched_{lv}": sum(k == 0 for k in per_state[q][lv]) for lv in LEVELS}))
    for lv in LEVELS:
        best = np.array([s[f"best_{lv}"] for s in state_rows])
        dist[lv] = dict(exact=[round(100 * float(v), 2) for v in np.bincount(best, minlength=n_open + 1)[: n_open + 1] / len(best)],
                        at_least=[round(100 * float((best >= x).mean()), 2) for x in range(n_open + 1)],
                        median=float(np.median(best)))
    all_k = {lv: np.array([k for q in states for k in per_state[q][lv]]) for lv in LEVELS}

    # per open model
    m_pool = {lv: collections.defaultdict(set) for lv in LEVELS}
    answered = collections.defaultdict(set)
    for r in op:
        q = (r["category"], r["item"])
        m_pool["question"][(r["model"], q)].add(r["cluster"])
        m_pool["type"][(r["model"], r["category"])].add(r["cluster"])
        m_pool["overall"][r["model"]].add(r["cluster"])
        answered[r["model"]].add(q)
    binary, frac = {}, {}
    for m in opens:
        qs = [q for q in ref_cues if q in answered[m]]
        pools = lambda q: {"question": m_pool["question"][(m, q)], "type": m_pool["type"][(m, q[0])],
                           "overall": m_pool["overall"][m]}
        b = {lv: 0.0 for lv in LEVELS}; f = {lv: 0.0 for lv in LEVELS}
        for q in qs:
            P = pools(q); C = ref_cues[q]
            for lv in LEVELS:
                b[lv] += bool(set(C) & P[lv]); f[lv] += np.mean([c in P[lv] for c in C])
        n = max(len(qs), 1)
        binary[m] = {lv: round(100 * b[lv] / n, 3) for lv in LEVELS} | {"n_questions": len(qs)}
        frac[m] = {lv: round(100 * f[lv] / n, 3) for lv in LEVELS} | {"n_questions": len(qs)}
    return state_rows, dist, all_k, binary, frac


# =============================================================================
# main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cues", default="cues.json")
    ap.add_argument("--out", default="results/cue_analysis")
    ap.add_argument("--threshold", type=float, default=0.5, help="cosine-distance cut (default 0.5)")
    ap.add_argument("--top-n", type=int, nargs="+", default=[5, 10],
                    help="cue-list lengths for the reference-overlap analyses (default: 5 10)")
    ap.add_argument("--reference", default="claude", help="substring identifying the reference model")
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Clean and parse
    data = json.load(open(args.cues))
    log, rows = collections.Counter(), []
    for cat, items in data.items():
        for item, per_model in items.items():
            for model, cues in per_model.items():
                for rank, cue in enumerate(clean_cues(cues, log)):
                    rows.append(dict(model=model, category=cat, item=item, rank=rank, cue=cue,
                                     behavior=behavior_phrase(cue)))
    models = sorted({r["model"] for r in rows})
    refs = [m for m in models if args.reference.lower() in m.lower()]
    if len(refs) != 1:
        sys.exit(f"--reference '{args.reference}' must match exactly one model; matched {refs}")
    ref = refs[0]; opens = [m for m in models if m != ref]
    print(f"[1] {len(rows)} cues, {len(models)} models (reference: {ref}); cleaning: "
          f"{ {k: v for k, v in log.items() if v} }")

    # Embed and cluster
    phrases = [r["behavior"] for r in rows]
    fp = hashlib.sha1(("\n".join(phrases) + args.embed_model).encode()).hexdigest()[:16]
    cache = out / f"cache_{fp}.npz"
    if cache.exists():
        Z = np.load(cache)["Z"]; print("[2] using cached embeddings and clustering tree")
    else:
        X = embed(phrases, args.embed_model)
        print(f"[2] embedded {X.shape[0]} behavior phrases")
        Z = linkage(pdist(X.astype(np.float64), "cosine"), method="average")
        np.savez_compressed(cache, Z=Z, X=X)
    lab = fcluster(Z, args.threshold, criterion="distance")
    order = [k for k, _ in collections.Counter(lab).most_common()]        # 0 = largest cluster
    remap = {k: i for i, k in enumerate(order)}
    for r, l in zip(rows, lab):
        r["cluster"] = remap[l]
    sizes = collections.Counter(r["cluster"] for r in rows)
    print(f"[3] threshold {args.threshold}: {len(order)} clusters, "
          f"{100 * sum(v == 1 for v in sizes.values()) / len(rows):.1f}% singleton cues")

    with open(out / "cues_clustered.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "category", "item", "rank", "cue", "behavior", "cluster"])
        w.writeheader(); w.writerows(rows)
    members = collections.defaultdict(list)
    for r in rows:
        members[r["cluster"]].append(r)
    with open(out / "clusters.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cluster", "size", "n_models", "n_reference_cues", "example"])
        for c in range(len(order)):
            ms = members[c]
            ex = collections.Counter(normalize(r["behavior"]) for r in ms).most_common(1)[0][0]
            w.writerow([c, len(ms), len({r["model"] for r in ms}), sum(r["model"] == ref for r in ms), ex])

    # Analysis of overlap
    kq = lambda r: (r["category"], r["item"], r["cluster"])
    kt = lambda r: (r["category"], r["cluster"])
    ka = lambda r: r["cluster"]
    summary = dict(cues=len(rows), models=len(models), reference=ref, threshold=args.threshold,
                   clusters=len(order), cleaning={k: v for k, v in log.items() if v},
                   singleton_cue_pct=round(100 * sum(v == 1 for v in sizes.values()) / len(rows), 2))
    for n in args.top_n:
        R = [r for r in rows if r["rank"] < n]
        block = {"cues": len(R),
                 "vocabulary_sharing_all_models": {
                     "same_state": sharing_stats(R, kq), "same_type": sharing_stats(R, kt),
                     "any_state": sharing_stats(R, ka)},
                 "by_type": {}}
        state_rows, dist, all_k, binary, frac = reference_overlap(R, ref, opens)
        block["reference_cues_unmatched_pct"] = {lv: pct(all_k[lv] == 0) for lv in LEVELS}
        block["states"] = {lv: {"any_overlap_pct": round(100 - dist[lv]["exact"][0], 2),
                                "best_cue_7plus_models_pct": dist[lv]["at_least"][7] if len(opens) >= 7 else None,
                                "median_models_best_cue": dist[lv]["median"]} for lv in LEVELS}
        block["per_model_overlap_mean_pct"] = {lv: round(float(np.mean([binary[m][lv] for m in opens])), 2)
                                               for lv in LEVELS}
        for cat in sorted({r["category"] for r in R}):
            Rc = [r for r in R if r["category"] == cat]
            sr = [s for s in state_rows if s["category"] == cat]
            block["by_type"][cat] = {
                "same_state": sharing_stats(Rc, kq), "same_type": sharing_stats(Rc, kt),
                "reference_cues_unmatched_same_state_pct": round(
                    100 * sum(s["unmatched_question"] for s in sr) / max(sum(s["n_ref_cues"] for s in sr), 1), 2),
                "states_best_cue_3plus_models_same_state_pct": pct([s["best_question"] >= 3 for s in sr]),
                "median_models_best_cue_same_state": float(np.median([s["best_question"] for s in sr]))}
        summary[f"top{n}"] = block

        json.dump(dist, open(out / f"state_overlap_top{n}.json", "w"), indent=1)
        with open(out / f"state_overlap_top{n}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(state_rows[0])); w.writeheader(); w.writerows(state_rows)
        json.dump(binary, open(out / f"model_overlap_top{n}.json", "w"), indent=1)
        json.dump(frac, open(out / f"model_overlap_top{n}_frac.json", "w"), indent=1)
        s = block
        print(f"[4] first {n} cues: median other models sharing a cue (any state) "
              f"{s['vocabulary_sharing_all_models']['any_state']['median_other_models']:g}; "
              f"reference cues unmatched (same state) {s['reference_cues_unmatched_pct']['question']}%; "
              f"median open models sharing best reference cue (same state) {s['states']['question']['median_models_best_cue']:g}; "
              f"mean per-model same-state overlap {s['per_model_overlap_mean_pct']['question']}%")
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s -> {out}/")


if __name__ == "__main__":
    main()