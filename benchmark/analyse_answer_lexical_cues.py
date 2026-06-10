"""
Check for lexical cues in implicit vignettes that predict the correct answer.

Analyses:
  1. Answer keyword leakage  — do words from the answer appear verbatim in the scenario?
  2. Classifier predictability — TF-IDF + logistic regression cross-val accuracy
     for predicting (a) 4-way type and (b) specific answer label
  3. Top TF-IDF features per type from the trained classifier
  4. Log-odds words per answer  (for answers with n ≥ MIN_ANS_DF occurrences)
  5. Mutual information: which unigrams are most informative about the answer?

"""

import argparse
import json
import re
import warnings
from collections import Counter, defaultdict

import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "that",
    "this", "it", "its", "he", "she", "they", "them", "his", "her", "their",
    "i", "we", "you", "my", "your", "our", "which", "who", "what", "when",
    "where", "how", "not", "no", "so", "if", "then", "than", "into", "out",
    "up", "down", "over", "about", "after", "before", "just", "like",
    "one", "said", "say", "says", "s", "t", "re", "ve", "ll", "d", "m",
}


def clean(text):
    text = re.split(r"\n---", text)[0]
    if text and text[0].islower():
        text = re.sub(r"^\S+\s*", "", text)
    return text


def content_words(text):
    return [w for w in re.findall(r"[a-z]+", text.lower())
            if w not in STOPWORDS and len(w) > 2]


def content_word_set(text):
    return set(content_words(text))


# ---------------------------------------------------------------------------
# 1. Answer keyword leakage
# ---------------------------------------------------------------------------

def analyse_leakage(data):
    print("=" * 65)
    print("1. ANSWER KEYWORD LEAKAGE")
    print("=" * 65)
    exact = partial = none_ = 0
    examples_partial = []
    for v in data.values():
        sc  = content_word_set(clean(v["scenario"]))
        ans = content_word_set(v["answer"])
        if not ans:
            continue
        overlap = ans & sc
        if ans.issubset(sc):
            exact += 1
        elif overlap:
            partial += 1
            if len(examples_partial) < 4:
                examples_partial.append((v["answer"], sorted(overlap)))
        else:
            none_ += 1

    n = exact + partial + none_
    print(f"  All answer words in scenario : {exact:>4}  ({100*exact/n:.1f}%)")
    print(f"  Partial overlap              : {partial:>4}  ({100*partial/n:.1f}%)")
    print(f"  No overlap                   : {none_:>4}  ({100*none_/n:.1f}%)")
    print(f"\n  Partial overlap examples:")
    for ans, words in examples_partial:
        print(f"    answer='{ans}'  shared words={words}")

    # Also check: does any *wrong* answer option overlap more than the correct one?
    more_distractor = 0
    for v in data.values():
        sc      = content_word_set(clean(v["scenario"]))
        correct = v["answer"]
        correct_overlap = len(content_word_set(correct) & sc)
        for opt in v["answer_options"]:
            if opt != correct and len(content_word_set(opt) & sc) > correct_overlap:
                more_distractor += 1
                break
    print(f"\n  Cases where a distractor has MORE lexical overlap than correct: "
          f"{more_distractor}/{n} ({100*more_distractor/n:.1f}%)")


# ---------------------------------------------------------------------------
# 2 & 3. Classifier predictability + top features per type
# ---------------------------------------------------------------------------

def analyse_classifier(data):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder

    print("\n" + "=" * 65)
    print("2. CLASSIFIER PREDICTABILITY (TF-IDF unigrams+bigrams)")
    print("=" * 65)

    docs    = [clean(v["scenario"]) for v in data.values()]
    answers = [v["answer"] for v in data.values()]
    types   = [v["type"]   for v in data.values()]
    n       = len(docs)

    vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2),
                          min_df=3, sublinear_tf=True)
    X   = vec.fit_transform(docs)
    clf = LogisticRegression(C=0.5, max_iter=1000, solver="lbfgs", random_state=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Type prediction
        le_type  = LabelEncoder()
        y_type   = le_type.fit_transform(types)
        cv_type  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        sc_type  = cross_val_score(clf, X, y_type, cv=cv_type, scoring="accuracy")
        print(f"\n  Type (4 classes):")
        print(f"    Cross-val accuracy: {sc_type.mean()*100:.1f}% ± {sc_type.std()*100:.1f}%"
              f"  (chance=25.0%)")

        # Answer prediction
        le_ans   = LabelEncoder()
        y_ans    = le_ans.fit_transform(answers)
        n_ans    = len(le_ans.classes_)
        cv_ans   = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        sc_ans   = cross_val_score(clf, X, y_ans, cv=cv_ans, scoring="accuracy")
        print(f"\n  Answer ({n_ans} classes):")
        print(f"    Cross-val accuracy: {sc_ans.mean()*100:.2f}% ± {sc_ans.std()*100:.2f}%"
              f"  (chance={100/n_ans:.2f}%)")

        # Fit on all tomstories_vignettes to extract type-level features
        clf.fit(X, y_type)

    print("\n" + "=" * 65)
    print("3. TOP TF-IDF FEATURES PER TYPE")
    print("=" * 65)
    feat_names = np.array(vec.get_feature_names_out())
    for i, t in enumerate(le_type.classes_):
        coefs   = clf.coef_[i]
        top_pos = feat_names[np.argsort(coefs)[-12:][::-1]]
        top_neg = feat_names[np.argsort(coefs)[:12]]
        print(f"\n  {t.upper()}")
        print(f"  Positive (predicts {t}):  {', '.join(top_pos)}")
        print(f"  Negative (predicts other): {', '.join(top_neg)}")


# ---------------------------------------------------------------------------
# 4. Log-odds words per frequent answer
# ---------------------------------------------------------------------------

def analyse_log_odds_per_answer(data, min_df=8, top_n=8):
    print("\n" + "=" * 65)
    print(f"4. LOG-ODDS WORDS PER ANSWER  (answers with n ≥ {min_df})")
    print("=" * 65)

    ans_counts = Counter(v["answer"] for v in data.values())
    frequent   = {a for a, c in ans_counts.items() if c >= min_df}

    # Build token sets
    by_answer = defaultdict(list)
    all_sets   = []
    for v in data.values():
        s = set(content_words(clean(v["scenario"])))
        all_sets.append(s)
        by_answer[v["answer"]].append(s)

    vocab = Counter(w for s in all_sets for w in s)
    vocab = [w for w, c in vocab.most_common() if c >= 5]

    n_total = len(all_sets)

    for answer in sorted(frequent, key=lambda a: -ans_counts[a]):
        in_sets  = by_answer[answer]
        out_sets = [s for a, sets in by_answer.items() if a != answer for s in sets]
        n_in     = len(in_sets)
        n_out    = len(out_sets)

        scored = []
        for w in vocab:
            a = sum(1 for s in in_sets  if w in s)
            b = sum(1 for s in out_sets if w in s)
            c = n_in  - a
            d = n_out - b
            if a + b == 0:
                continue
            table = np.array([[a, b], [c, d]])
            try:
                chi2, p, _, _ = chi2_contingency(table, correction=False)
            except Exception:
                continue
            lo = np.log((a + 0.5) / (c + 0.5)) - np.log((b + 0.5) / (d + 0.5))
            scored.append((w, lo, p))

        scored.sort(key=lambda x: -x[1])
        over  = [(w, lo, p) for w, lo, p in scored[:top_n]  if p < 0.05]
        under = [(w, lo, p) for w, lo, p in scored[-top_n:] if p < 0.05][::-1]

        if not over and not under:
            continue

        typ = Counter(v["type"] for v in data.values() if v["answer"] == answer).most_common(1)[0][0]
        print(f"\n  '{answer}'  (n={ans_counts[answer]}, type={typ})")
        if over:
            words = [f"{w}({lo:+.1f})" for w, lo, p in over]
            print(f"    over : {', '.join(words)}")
        if under:
            words = [f"{w}({lo:+.1f})" for w, lo, p in under]
            print(f"    under: {', '.join(words)}")


# ---------------------------------------------------------------------------
# 5. Mutual information: top words most informative about the answer
# ---------------------------------------------------------------------------

def analyse_mutual_information(data, top_n=20):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder

    print("\n" + "=" * 65)
    print("5. MUTUAL INFORMATION: TOP WORDS PREDICTING ANSWER LABEL")
    print("=" * 65)

    docs    = [clean(v["scenario"]) for v in data.values()]
    answers = [v["answer"] for v in data.values()]
    types   = [v["type"]   for v in data.values()]

    vec  = CountVectorizer(max_features=2000, min_df=5,
                           stop_words=list(STOPWORDS))
    X    = vec.fit_transform(docs).toarray()
    feat = np.array(vec.get_feature_names_out())

    le   = LabelEncoder()

    for label_name, label_list in [("answer", answers), ("type", types)]:
        y  = le.fit_transform(label_list)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mi = mutual_info_classif(X, y, discrete_features=True, random_state=42)
        top_idx = np.argsort(mi)[-top_n:][::-1]
        print(f"\n  Top {top_n} words by MI with {label_name}:")
        for idx in top_idx:
            print(f"    {feat[idx]:<20}  MI={mi[idx]:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tomstories_vignettes",        default="tomstories_vignettes/implicit_vignettes.json")
    parser.add_argument("--split",       default="all", choices=["train", "val", "all"])
    parser.add_argument("--min_ans_df",  type=int, default=8,
                        help="Min occurrences for per-answer log-odds (default: 8)")
    parser.add_argument("--top_n",       type=int, default=8,
                        help="Words to show per answer/type (default: 8)")
    args = parser.parse_args()

    with open(args.tomstories_vignettes) as f:
        raw = json.load(f)
    data = {k: v for k, v in raw.items()
            if args.split == "all" or v.get("split") == args.split}
    print(f"Loaded {len(data)} samples  (split={args.split})\n")

    analyse_leakage(data)
    analyse_classifier(data)
    analyse_log_odds_per_answer(data, min_df=args.min_ans_df, top_n=args.top_n)
    analyse_mutual_information(data, top_n=args.top_n)


if __name__ == "__main__":
    main()
