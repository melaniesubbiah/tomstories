# ToMStories

Code and data for the ToMStories paper: <link todo>.

ToMStories is a Theory-of-Mind benchmark of short literary vignettes. Each vignette conveys
a character's **wound**, **emotion**, **fear**, or **motivation** and asks a four-choice
question about it. The implicit split conveys the mental state through subtext; the explicit
split states it outright; the conflicted split makes dialogue and narration disagree.

## Setup

Python 3.9+.

```bash
pip install -r requirements.txt
pip install vllm          # local-model inference; needs a CUDA GPU
```

Put API keys in a `.env` file in the repo root (or export them). They are only needed for
data generation and the LLM judges:

```
ANTHROPIC_API_KEY=...     # data generation
OPENAI_API_KEY=...        # LLM judges
```

Run every script from the repo root, e.g. `python benchmark/benchmark.py`.

## Repository layout

```
tomstories_vignettes/   the three benchmark splits (JSON)
data_gen/               regenerate the splits with the Anthropic API
benchmark/              evaluation and scoring
interpretability/       probes, representation analysis, steering, indicator phrases
common/                 shared model registry, prompts, data IO, model hooks
results/                created at runtime
```

### Benchmark

```bash
# Multiple-choice and open-ended, all local models
python benchmark/benchmark.py --all --mode both

# API models are opt-in, never included in --all
python benchmark/benchmark.py --models gpt-4o-mini --mode mc

# Modality bias, after benchmarking the conflicted split
python benchmark/benchmark.py --all --mode mc \
    --tomstories_vignettes tomstories_vignettes/conflicted_vignettes.json
python benchmark/score_conflicted.py
```

`benchmark.py --summarize` reprints the summary from existing result files without
rerunning anything.

### Interpretability

`linear_probes.py` caches the activations the other scripts reuse and
chooses the layer they all work at, saving it to `results/probes/<model>_best_layers.json`.
Each steering step then consumes the previous one's output.

```bash
# 1. Probes: fit every layer, cache activations, choose the layer
python interpretability/linear_probes.py --models llama-3.1-8b

# 2. Implicit vs. explicit representation structure
python interpretability/analyze_embeddings.py --models llama-3.1-8b \
    --probe_acts_path results/probes

# 3. Steering vectors at the chosen layer
python interpretability/steering/extract_vectors.py --models llama-3.1-8b \
    --probe_acts_path results/probes

# 4. Largest coefficient that stays coherent
python interpretability/steering/tune_steering.py --models llama-3.1-8b

# 5. Generate steered vignettes, then score them against the NRC lexicons
python interpretability/steering/steer_vignettes.py --models llama-3.1-8b
python interpretability/steering/analyze_steering.py \
    --eil NRC-Emotion-Intensity-Lexicon-v1.txt
```

Pass `--layers` to any of steps 2-3 to override the chosen layer. Our tuned steering coefficients and steered vignettes are saved in results/steering_coherence and results/steered_vignettes.

### Indicator phrases

Independent of the above. `generate_indicators.py` first, then `indicator_overlap.py`.

```bash
python interpretability/indicator_analysis/generate_indicators.py --all \
    --input data_gen/thesaurus/thesaurus.json \
    --output interpretability/indicator_analysis/indicators.json --limit-per-category 25
python interpretability/indicator_analysis/indicator_overlap.py \
    --indicators interpretability/indicator_analysis/indicators.json \
    --out results/indicator_analysis
```

### Regenerating the datasets

Needs `ANTHROPIC_API_KEY` and a filled-in `data_gen/thesaurus/thesaurus.json`. Run implicit
first as explicit pairs against it.

```bash
python data_gen/generate_implicit.py
python data_gen/generate_explicit.py
python data_gen/generate_conflicted.py
```

## Data

| File | Samples | Description |
|------|---------|-------------|
| `tomstories_vignettes/implicit_vignettes.json`   | 1988 (1593 train / 395 val) | The mental state must be inferred from subtext |
| `tomstories_vignettes/explicit_vignettes.json`   | 1988 (1593 train / 395 val) | Matched counterparts that state it outright, same options |
| `tomstories_vignettes/conflicted_vignettes.json` | 1956 (1561 train / 395 val) | Dialogue and narration point to different answers |

Implicit and explicit entries share sample IDs and have `scenario`, `question`, `answer`,
`answer_options` (4 choices), `type`, `setting`, `relationship`, `style`, and `split`.
Conflicted entries replace `answer` with `dialogue_answer` and `description_answer`.

## Citation

TODO
