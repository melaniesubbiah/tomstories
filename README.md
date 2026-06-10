# ToMStories

This repository provides code for the ToMStories paper: <link todo>.

## Repository structure

```
tomstories_vignettes/   Benchmark datasets (JSON)
data_gen/               Scripts to regenerate the datasets with the Anthropic API
benchmark/              Evaluation scripts and analysis tools
interpretability/       Activation steering and representation analysis scripts
results/                (created at runtime) Output from experiments
```

## Quick start

**Run the benchmark on a subset of ToMStories:**
```bash
python benchmark/benchmark.py --all --mode both --tomstories_vignettes tomstories_vignettes/implicit_vignettes.json
```

**Analyse conflicted results after running the benchmark:**
```bash
python benchmark/analyse_conflicted.py
python benchmark/plot_implicit_vs_conflicted.py
```

**Analyse structural correlations in the dataset:**
```bash
python benchmark/analyse_answer_lexical_cues.py
python benchmark/analyse_implicit_correlations.py
```

**Regenerate dataset splits** (requires `ANTHROPIC_API_KEY` and manual entry of thesaurus entries):
```bash
python data_gen/generate_implicit.py
python data_gen/generate_explicit.py
python data_gen/generate_conflicted.py
```

**Extract steering vectors:**
```bash
python interpretability/steering_vectors.py
```

**Train linear probes across key layers:**
```bash
python interpretability/linear_probes.py --answer_per_style --choice_probe
```

**Try steering:**
```bash
python interpretability/steer_chat.py --vector results/steering/Llama-3.1-8B-Instruct_layer20_vectors.pt --layer 20```
```

## Datasets

| File                                 | Description                                                                                                                                                                                                                    |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `data_gen/implicit_vignettes.json`   | Vignettes where the correct mental state must be inferred from subtext. Each entry has `scenario`, `question`, `answer`, `answer_options` (4-choice), `type`, `setting`, `relationship`, and `split` (train/val).              |
| `data_gen/explicit_vignettes.json`   | Matched explicit counterparts: same concepts, settings, and answer options as implicit, but the mental state is described directly.                                                                                            |
| `data_gen/conflicted_vignettes.json` | Vignettes where dialogue signals one answer and behavioral description signals a different answer. Used to measure modality bias. Each entry has `dialogue_answer` and `description_answer` instead of a single gold `answer`. |

## Requirements / setup

Python 3.9+ is required.

```bash
pip install -r requirements.txt
```

API keys (only needed for data generation and LLM judges):
- `ANTHROPIC_API_KEY` — Claude API (data generation)
- `OPENAI_API_KEY` — OpenAI API (LLM judge)

Place keys in a `.env` file in the repo root or export them as environment variables.

Open-source model inference uses [vLLM](https://github.com/vllm-project/vllm) (`pip install vllm`), which requires a CUDA GPU.

## Citation

TODO
