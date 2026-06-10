"""
Interactive steered chat. Loads a model and a steering vector, then enters a loop where each prompt
you type is answered with the vector injected at a chosen layer.
"""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()


def make_hook(vector, alpha):
    def hook(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        hs = hs + alpha * vector.to(device=hs.device, dtype=hs.dtype)
        return (hs,) + output[1:] if isinstance(output, tuple) else hs
    return hook


def generate(model, tokenizer, prompt, layer_idx=None, vector=None, alpha=0,
             max_new_tokens=512, system_prompt=None):
    messages = []
    if system_prompt:
        try:
            tokenizer.apply_chat_template(
                [{"role": "system", "content": "x"}, {"role": "user", "content": "x"}],
                tokenize=False, add_generation_prompt=True,
            )
            messages.append({"role": "system", "content": system_prompt})
        except Exception:
            prompt = system_prompt + "\n\n" + prompt

    messages.append({"role": "user", "content": prompt})
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)

    hook_handle = None
    if vector is not None and alpha != 0 and layer_idx is not None:
        hook_handle = model.model.layers[layer_idx].register_forward_hook(
            make_hook(vector, alpha)
        )

    try:
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    new_tokens = out_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model",          default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--vector",         required=True,
                        help="Path to .pt vectors file from steering_vectors.py")
    parser.add_argument("--layer",          type=int, required=True)
    parser.add_argument("--vector_key",     default="overall",
                        choices=["overall", "wound", "emotion", "fear", "motivation"])
    parser.add_argument("--alpha",          type=float, default=20.0,
                        help="Steering strength (negative = explicit/dialogue direction)")
    parser.add_argument("--normalize",      action="store_true",
                        help="Unit-normalize the vector before scaling")
    parser.add_argument("--compare",        action="store_true",
                        help="Print unsteered response first, then steered response")
    parser.add_argument("--system_prompt",  default=None,
                        help="Optional system prompt prepended to every message")
    parser.add_argument("--max_tokens",     type=int, default=512)
    parser.add_argument("--quantize",       action="store_true",
                        help="4-bit bitsandbytes quantization")
    parser.add_argument("--prompt",         default=None,
                        help="Single prompt (non-interactive mode). Prints output and exits.")
    args = parser.parse_args()

    # ── Load vector ────────────────────────────────────────────────────────
    try:
        vectors = torch.load(args.vector, map_location="cpu", weights_only=True)
    except Exception:
        vectors = torch.load(args.vector, map_location="cpu")

    if args.vector_key not in vectors:
        raise KeyError(
            f"Key '{args.vector_key}' not in vector file. "
            f"Available: {list(vectors.keys())}"
        )
    v = vectors[args.vector_key].float()
    if args.normalize:
        v = v / (v.norm() + 1e-9)
    print(f"Vector '{args.vector_key}'  norm={v.norm():.3f}"
          f"  alpha={args.alpha:+.1f}"
          f"{'  (normalised)' if args.normalize else ''}")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"Loading {args.model} …")
    fa2_kwargs = {}
    try:
        import flash_attn  # noqa: F401
        fa2_kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass

    if args.quantize:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb, device_map="auto", **fa2_kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto", **fa2_kwargs
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()

    num_layers = model.config.num_hidden_layers
    if not (0 <= args.layer < num_layers):
        raise ValueError(f"--layer {args.layer} out of range (model has {num_layers} layers)")
    print(f"Ready — injecting at layer {args.layer}/{num_layers - 1}. "
          f"Type 'quit' or Ctrl-D to exit.\n")

    # ── Helper ─────────────────────────────────────────────────────────────
    def respond(prompt_text):
        if args.compare:
            print("\n── Unsteered ──────────────────────────────────────────")
            unsteered = generate(
                model, tokenizer, prompt_text,
                max_new_tokens=args.max_tokens, system_prompt=args.system_prompt
            )
            print(unsteered)
            print(f"\n── Steered (alpha={args.alpha:+.1f}) ──────────────────────────")
        steered = generate(
            model, tokenizer, prompt_text,
            layer_idx=args.layer, vector=v, alpha=args.alpha,
            max_new_tokens=args.max_tokens, system_prompt=args.system_prompt,
        )
        print(steered)

    # ── Non-interactive mode ───────────────────────────────────────────────
    if args.prompt:
        respond(args.prompt)
        return

    # ── Interactive loop ───────────────────────────────────────────────────
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            break
        print()
        respond(user_input)


if __name__ == "__main__":
    main()
