"""Dataset loading, incremental result files, and answer-option sampling."""

import json
import os

from common.prompts import TYPE_TO_THESAURUS_KEY, TYPES


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, obj, indent=None):
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)


def load_vignettes(path, split="all"):
    """Load a vignette JSON file, optionally keeping only one split."""
    data = load_json(path)
    if split and split != "all":
        data = {k: v for k, v in data.items() if v.get("split") == split}
    return data


def dataset_stem(path):
    """'tomstories_vignettes/implicit_vignettes.json' -> 'implicit_vignettes'."""
    return os.path.splitext(os.path.basename(path))[0]


def load_best_layer(probes_dir, model):
    """
    The layer to analyse/steer at, as chosen and saved by linear_probes.py.

    Returns the whole <model>_best_layers.json record; callers normally want
    ["layer"]. Nothing here re-derives the layer — run linear_probes.py to
    change how it is picked.
    """
    path = os.path.join(str(probes_dir), f"{model}_best_layers.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no layer selection for {model!r} at {path}. Run:\n"
            f"    python interpretability/linear_probes.py --models {model}\n"
            f"(or point --probe_results_dir at the directory holding "
            f"{model}_best_layers.json)"
        )
    record = load_json(path)
    if "layer" not in record:
        raise ValueError(
            f"{path} has no 'layer' field; it was written by an older version of "
            f"linear_probes.py. Rerun it (results are cached, so probes are not refit)."
        )
    return record


def resume(path):
    """Previously written results for an incremental run, or {} if there are none."""
    if os.path.exists(path):
        results = load_json(path)
        print(f"Resuming: {len(results)} samples already done.")
        return results
    return {}


def split_sizes(train_per_type, val_per_type, max_per_type=None):
    """Per-type train/val counts, rescaled to `max_per_type` total when given."""
    if max_per_type is None:
        return train_per_type, val_per_type
    train = max(1, int(max_per_type * train_per_type / (train_per_type + val_per_type)))
    return train, max(1, max_per_type - train)


def add_generation_args(parser, output_default, model_default="claude-sonnet-4-6"):
    parser.add_argument("--model", default=model_default)
    parser.add_argument("--output", default=output_default, help="Output file path")
    parser.add_argument("--type_filter", choices=TYPES, default=None,
                        help="Only generate for one question type")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="Max concurrent API requests (default: 20)")


def add_combo_args(parser, thesaurus_default="data_gen/thesaurus/thesaurus.json"):
    parser.add_argument("--thesaurus", default=thesaurus_default)
    parser.add_argument("--train_per_type", type=int, default=400)
    parser.add_argument("--val_per_type", type=int, default=100)
    parser.add_argument("--max_per_type", type=int, default=None,
                        help="Cap total per type (train+val). Useful for smoke tests.")
    parser.add_argument("--seed", type=int, default=42)


def sample_answer_options(thesaurus, typ, answers, rng, n_options=4):
    """Shuffled option list holding `answers` plus distractors drawn from the thesaurus."""
    pool = [a for a in thesaurus[TYPE_TO_THESAURUS_KEY[typ]] if a not in answers]
    options = list(answers) + rng.sample(pool, n_options - len(answers))
    rng.shuffle(options)
    return options
