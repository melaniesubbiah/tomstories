"""Model registry and the --models/--all argument pair shared by all scripts."""

MODELS = {
    "llama-3.2-1b":   {"hf_id": "meta-llama/Llama-3.2-1B-Instruct",                    "params_b": 1},
    "llama-3.2-3b":   {"hf_id": "meta-llama/Llama-3.2-3B-Instruct",                    "params_b": 3},
    "llama-3.1-8b":   {"hf_id": "meta-llama/Llama-3.1-8B-Instruct",                    "params_b": 8},
    "llama-3.1-70b":  {"hf_id": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4", "params_b": 70, "tp": 2, "quantize": "awq_marlin"},
    "llama-3.3-70b":  {"hf_id": "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",    "params_b": 70, "tp": 2, "quantize": "awq_marlin"},
    "qwen-2.5-7b":    {"hf_id": "Qwen/Qwen2.5-7B-Instruct",                            "params_b": 7},
    "qwen-2.5-72b":   {"hf_id": "Qwen/Qwen2.5-72B-Instruct-AWQ",                       "params_b": 72, "tp": 2, "quantize": "awq_marlin"},
    "mistral-7b":     {"hf_id": "mistralai/Mistral-7B-Instruct-v0.3",                  "params_b": 7},
    "gemma-2-9b":     {"hf_id": "google/gemma-2-9b-it",                                "params_b": 9},
    "gemma-2-27b":    {"hf_id": "google/gemma-2-27b-it",                               "params_b": 27, "tp": 2},
    "phi-4":          {"hf_id": "microsoft/phi-4",                                     "params_b": 14},
    "deepseek-r1-7b": {"hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",             "params_b": 7},
    "deepseek-r1-8b": {"hf_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",            "params_b": 8},
}

API_MODELS = {
    "gpt-4o-mini":       {"provider": "openai",    "api_id": "gpt-4o-mini"},
    "claude-sonnet-4.6": {"provider": "anthropic", "api_id": "claude-sonnet-4-6"},
}

ALL_MODELS = {**MODELS, **API_MODELS}


def add_model_args(parser, registry=MODELS, default=None, help=None, all_help=None):
    parser.add_argument("--models", nargs="+", choices=list(registry), metavar="MODEL",
                        default=default, help=help)
    parser.add_argument("--all", action="store_true",
                        help=all_help or "Run all registered models")


def selected_models(args, parser, registry=MODELS, extra_hint=""):
    """Model names from --models/--all, or exit with a usage error."""
    if getattr(args, "all", False):
        return list(registry)
    if args.models:
        return args.models
    parser.error(f"Specify --models or --all{extra_hint}")
