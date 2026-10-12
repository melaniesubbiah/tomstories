"""Transformers / vLLM helpers: model loading, prompting, activations, steering hooks."""

import gc

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

from common.prompts import SYSTEM_PROMPT, chat_messages, mc_question

hf_logging.set_verbosity_error()


# ── Model loading ─────────────────────────────────────────────────────────────

def _dtype_kwarg():
    """transformers renamed `torch_dtype` to `dtype` in 4.56."""
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


def load_model(cfg, flash_attention=False):
    """Load a model/tokenizer pair from a MODELS entry, sharded across visible GPUs."""
    kwargs = {"device_map": "auto", "low_cpu_mem_usage": True}
    if cfg.get("quantize"):
        # AWQ kernels pick their own int4 compute dtype
        kwargs[_dtype_kwarg()] = "auto"
    else:
        kwargs[_dtype_kwarg()] = torch.bfloat16
        if flash_attention:
            try:
                import flash_attn  # noqa: F401
                kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                pass

    model = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], **kwargs).eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_vllm(cfg, gpu_memory_utilization=0.90, max_model_len=None, dtype=None,
              trust_remote_code=False):
    from vllm import LLM

    kwargs = dict(
        model=cfg["hf_id"],
        tensor_parallel_size=cfg.get("tp", 1),
        trust_remote_code=trust_remote_code,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    if cfg.get("quantize"):
        kwargs["quantization"] = cfg["quantize"]
    if dtype is not None:
        kwargs["dtype"] = dtype
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    return LLM(**kwargs)


def destroy_vllm(llm):
    """Release GPU memory held by a vLLM engine so another model can be loaded."""
    del llm
    free_cuda()
    try:
        from vllm.distributed.parallel_state import (destroy_distributed_environment,
                                                     destroy_model_parallel)
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass


# ── Prompting ─────────────────────────────────────────────────────────────────

def supports_system_role(tokenizer):
    """True if the model's chat template accepts a system message."""
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "x"}],
            tokenize=False, add_generation_prompt=True,
        )
        return True
    except Exception:
        return False


def format_chat(tokenizer, system, user, system_supported=True):
    return tokenizer.apply_chat_template(
        chat_messages(system, user, system_supported),
        tokenize=False, add_generation_prompt=True,
    )


def build_mc_prompt(sample, tokenizer, system_supported=True):
    """Chat-templated multiple-choice prompt for one vignette."""
    return format_chat(tokenizer, SYSTEM_PROMPT, mc_question(sample), system_supported)


# ── Activations ───────────────────────────────────────────────────────────────

def layer_module(model, idx):
    return model.model.layers[idx]


def clamp_layers(layers, num_layers):
    """Sorted, de-duplicated layer indices that actually exist in the model."""
    return sorted({li for li in layers if 0 <= li < num_layers})


def extract_activations(model, tokenizer, samples, layer_indices, batch_size, device,
                        system_supported=True, max_length=1024, desc="extracting"):
    """Last-token residual-stream activations at each requested layer: {layer: (N, hidden)}."""
    layer_indices = list(layer_indices)
    buffers = {li: [] for li in layer_indices}
    hooks   = []

    def make_hook(li):
        def hook_fn(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            buffers[li].append(hs[:, -1, :].detach().float().cpu())
        return hook_fn

    for li in layer_indices:
        hooks.append(layer_module(model, li).register_forward_hook(make_hook(li)))

    try:
        for i in tqdm(range(0, len(samples), batch_size), desc=desc, leave=False):
            prompts = [build_mc_prompt(s, tokenizer, system_supported)
                       for s in samples[i: i + batch_size]]
            inputs  = tokenizer(prompts, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_length).to(device)
            with torch.no_grad():
                model(**inputs, use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    return {li: torch.cat(buffers[li], dim=0) for li in layer_indices}


# ── Steering ──────────────────────────────────────────────────────────────────

def load_unit_vector(path, key):
    """Unit-normalized steering vector from a .pt file holding a dict or a bare tensor."""
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        if key in obj:
            v = obj[key]
        else:
            tensors = {k: x for k, x in obj.items() if torch.is_tensor(x)}
            if len(tensors) != 1:
                raise KeyError(f"key {key!r} not in {path}; available keys: {list(obj)}")
            v = next(iter(tensors.values()))   # file holds a single vector: use it
    else:
        v = obj
    v = torch.as_tensor(v, dtype=torch.float32).flatten()
    norm = float(v.norm())
    if norm == 0:
        raise ValueError(f"zero steering vector in {path}")
    return v / norm


class RowSteerer:
    """Forward hook adding coeffs[row] * vec at one layer, for every position."""

    def __init__(self, model, layer, vec):
        self.vec, self.coeffs, self._cache = vec, None, {}
        self.handle = layer_module(model, layer).register_forward_hook(self)

    def __call__(self, module, inputs, output):
        if self.coeffs is None:
            return None
        hs  = output[0] if isinstance(output, tuple) else output
        key = (hs.device, hs.dtype)
        if key not in self._cache:
            self._cache[key] = self.vec.to(hs.device, hs.dtype)
        hs = hs + self.coeffs.to(hs.device, hs.dtype)[:, None, None] * self._cache[key]
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

    def remove(self):
        self.handle.remove()
