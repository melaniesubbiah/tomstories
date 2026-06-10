"""
Check for structural correlations between implicit vignettes and their answers.

Tests:
  1. Vignette word count          × type  (Kruskal-Wallis)
  2. Vocabulary richness (TTR)    × type  (Kruskal-Wallis)
  3. Distractor similarity        × type  (mean Levenshtein distance, correct vs. wrong options)
  4. Setting                      × type  (chi-squared)
  5. Relationship                 × type  (chi-squared)
  6. Answer frequency             × type  (top 15 most common answers)
  7. Lexical correlations: log-odds words per type

Usage:
    python analyse_implicit_correlations.py
    python analyse_implicit_correlations.py --split val
"""

import argparse
import json
import re
from collections import Counter, defaultdict
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def word_count(text):
    return len(text.split())


def type_token_ratio(text):
    tokens = re.findall(r"[a-z']+", text.lower())
    return len(set(tokens)) / len(tokens) if tokens else 0


def levenshtein(a, b):
    a, b = a.lower(), b.lower()
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        ndp = [i + 1]
        for j, cb in enumerate(b):
            ndp.append(min(dp[j] + (ca != cb), dp[j + 1] + 1, ndp[-1] + 1))
        dp = ndp
    return dp[-1]


def mean_distractor_distance(sample):
    """Mean Levenshtein distance between correct answer and wrong options."""
    correct = sample["answer"]
    wrong   = [o for o in sample["answer_options"] if o != correct]
    if not wrong:
        return None
    return np.mean([levenshtein(correct, w) for w in wrong])


def kruskal_summary(groups: dict, label: str):
    """Kruskal-Wallis across groups, print result."""
    arrays = list(groups.values())
    names  = list(groups.keys())
    if len(arrays) < 2:
        return
    stat, p = stats.kruskal(*arrays)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    means = {n: np.mean(a) for n, a in groups.items()}
    print(f"  Kruskal-Wallis H={stat:.2f}  p={p:.4f} {sig}")
    for n in sorted(names):
        print(f"    {n:<14}  mean={means[n]:.2f}  median={np.median(groups[n]):.2f}  n={len(groups[n])}")


def chi2_summary(contingency: dict, row_label: str, top_n: int = 8):
    """Chi-squared test on a {row: {col: count}} contingency table."""
    from scipy.stats import chi2_contingency
    types   = sorted({t for counts in contingency.values() for t in counts})
    rows    = sorted(contingency.keys(),
                     key=lambda r: sum(contingency[r].values()), reverse=True)[:top_n]
    matrix  = np.array([[contingency[r].get(t, 0) for t in types] for r in rows])
    # Drop all-zero rows/cols
    matrix  = matrix[matrix.sum(axis=1) > 0][:, matrix.sum(axis=0) > 0]
    if matrix.shape[0] < 2:
        print("  Too few categories for chi-squared.")
        return
    chi2, p, dof, _ = chi2_contingency(matrix)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"  χ²={chi2:.2f}  df={dof}  p={p:.4f} {sig}  (top {len(matrix)} {row_label}s)")
    # Print type distribution for top rows
    header = f"  {'':25}" + "".join(f"  {t:<12}" for t in types) + "  total"
    print(header)
    for row, r in zip(rows[:len(matrix)], matrix):
        total = r.sum()
        pcts  = "".join(f"  {100*v/total:>10.1f}%" for v in r)
        print(f"  {row:<25}{pcts}  {total:>5}")


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "that",
    "this", "it", "its", "he", "she", "they", "them", "his", "her", "their",
    "i", "we", "you", "my", "your", "our", "which", "who", "what", "when",
    "where", "how", "not", "no", "so", "if", "then", "than", "into", "out",
    "up", "down", "over", "about", "after", "before", "just", "like", "s",
    "t", "re", "ve", "ll", "d", "m",
}


def tokenise(text):
    # Strip trailing markdown artefacts (--- and anything after)
    text = re.split(r"\n---", text)[0]
    # Strip leading generation fragment: if text starts lowercase, drop up to first whitespace
    if text and text[0].islower():
        text = re.sub(r"^\S+\s*", "", text)
    return [w for w in re.findall(r"[a-z']+", text.lower()) if w not in STOPWORDS and len(w) > 2]


def build_doc_term(docs, min_df=5):
    """Return (vocab list, binary doc-term matrix as list-of-sets)."""
    sets = [set(tokenise(d)) for d in docs]
    freq = Counter(w for s in sets for w in s)
    vocab = [w for w, c in freq.most_common() if c >= min_df]
    return vocab, sets


def log_odds_per_type(data: dict, types: list, min_df: int = 5, top_n: int = 12):
    """
    For each type, compute the log-odds ratio of each word appearing in
    vignettes of that type vs. all other types.  Returns
    {type: [(word, log_odds, p_value), ...]} sorted by log_odds descending.
    """
    from scipy.stats import chi2_contingency

    docs_by_type = defaultdict(list)
    for v in data.values():
        docs_by_type[v["type"]].append(v["scenario"])

    all_docs  = [v["scenario"] for v in data.values()]
    vocab, _  = build_doc_term(all_docs, min_df=min_df)

    results = {}
    for t in types:
        in_docs  = [set(tokenise(d)) for d in docs_by_type[t]]
        out_docs = [set(tokenise(d)) for d in all_docs
                    if True]  # rebuilt below
        # Count per-word presence
        n_in  = len(docs_by_type[t])
        n_out = sum(len(v) for k, v in docs_by_type.items() if k != t)
        out_sets = [set(tokenise(d)) for t2, docs in docs_by_type.items()
                    if t2 != t for d in docs]

        word_scores = []
        for w in vocab:
            a = sum(1 for s in in_docs  if w in s)   # in-type, has word
            b = sum(1 for s in out_sets if w in s)   # out-type, has word
            c = n_in  - a
            d = n_out - b
            if a + b == 0 or c + d == 0:
                continue
            table = np.array([[a, b], [c, d]])
            try:
                chi2, p, _, _ = chi2_contingency(table, correction=False)
            except Exception:
                continue
            # Log-odds: positive = over-represented in this type
            lo = np.log((a + 0.5) / (c + 0.5)) - np.log((b + 0.5) / (d + 0.5))
            word_scores.append((w, lo, p))

        word_scores.sort(key=lambda x: -x[1])
        results[t] = word_scores[:top_n], word_scores[-top_n:][::-1]
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tomstories_vignettes",  default="tomstories_vignettes/implicit_vignettes.json")
    parser.add_argument("--split", default="val", choices=["train", "val", "all"])
    args = parser.parse_args()

    with open(args.tomstories_vignettes) as f:
        raw = json.load(f)

    data = {k: v for k, v in raw.items()
            if args.split == "all" or v.get("split") == args.split}
    print(f"Loaded {len(data)} samples  (split={args.split})\n")

    types = sorted(set(v["type"] for v in data.values()))

    # ------------------------------------------------------------------
    # Pre-compute features
    # ------------------------------------------------------------------
    for sid, v in data.items():
        v["_wc"]   = word_count(v["scenario"])
        v["_ttr"]  = type_token_ratio(v["scenario"])
        v["_dist"] = mean_distractor_distance(v)

    # ------------------------------------------------------------------
    # 1. Word count × type
    # ------------------------------------------------------------------
    print("=" * 65)
    print("1. VIGNETTE WORD COUNT × TYPE")
    print("=" * 65)
    wc_by_type = defaultdict(list)
    for v in data.values():
        wc_by_type[v["type"]].append(v["_wc"])
    kruskal_summary(wc_by_type, "word count")

    # ------------------------------------------------------------------
    # 2. TTR × type
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("2. VOCABULARY RICHNESS (TYPE-TOKEN RATIO) × TYPE")
    print("=" * 65)
    ttr_by_type = defaultdict(list)
    for v in data.values():
        ttr_by_type[v["type"]].append(v["_ttr"])
    kruskal_summary(ttr_by_type, "TTR")

    # ------------------------------------------------------------------
    # 3. Distractor string distance × type
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("3. DISTRACTOR LEVENSHTEIN DISTANCE (correct vs. wrong options) × TYPE")
    print("=" * 65)
    dd_by_type = defaultdict(list)
    for v in data.values():
        if v["_dist"] is not None:
            dd_by_type[v["type"]].append(v["_dist"])
    kruskal_summary(dd_by_type, "distractor distance")

    # ------------------------------------------------------------------
    # 4. Setting × type
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("4. SETTING × TYPE  (top 8 settings)")
    print("=" * 65)
    setting_type = defaultdict(lambda: defaultdict(int))
    for v in data.values():
        setting_type[v["setting"]][v["type"]] += 1
    chi2_summary(setting_type, "setting", top_n=8)

    # ------------------------------------------------------------------
    # 5. Relationship × type
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("5. RELATIONSHIP × TYPE  (top 10 relationships)")
    print("=" * 65)
    rel_type = defaultdict(lambda: defaultdict(int))
    for v in data.values():
        rel_type[v["relationship"]][v["type"]] += 1
    chi2_summary(rel_type, "relationship", top_n=10)

    # ------------------------------------------------------------------
    # 6. Answer frequency × type
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("6. ANSWER FREQUENCY (top 15 most common answers)")
    print("=" * 65)
    ans_counts = Counter(v["answer"] for v in data.values())
    freq_type  = defaultdict(lambda: defaultdict(int))
    for v in data.values():
        freq_type[v["answer"]][v["type"]] += 1
    print(f"  {'Answer':<40}  {'Type':<12}  Count")
    for ans, cnt in ans_counts.most_common(15):
        typ = max(freq_type[ans], key=freq_type[ans].get)
        print(f"  {ans:<40}  {typ:<12}  {cnt}")

    # ------------------------------------------------------------------
    # 7. Lexical correlations: log-odds words per type
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("7. LEXICAL CORRELATIONS: TOP WORDS PER TYPE (log-odds vs. all other types)")
    print("=" * 65)
    lo_results = log_odds_per_type(data, types, min_df=5, top_n=12)
    for t in types:
        over, under = lo_results[t]
        print(f"\n  {t.upper()}")
        print(f"  {'Over-represented':<45}  Under-represented")
        for (wo, lo_o, po), (wu, lo_u, pu) in zip(over, under):
            so = "***" if po < 0.001 else ("**" if po < 0.01 else ("*" if po < 0.05 else ""))
            su = "***" if pu < 0.001 else ("**" if pu < 0.01 else ("*" if pu < 0.05 else ""))
            print(f"    {wo:<18} logOR={lo_o:+.2f}{so:<4}    {wu:<18} logOR={lo_u:+.2f}{su}")



if __name__ == "__main__":
    main()
