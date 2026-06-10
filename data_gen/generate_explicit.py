"""
Generate explicit vignettes paired with the existing implicit dataset.
"""

import argparse
import asyncio
import json
import os
import random
import time

from anthropic import AsyncAnthropic, RateLimitError
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


SYSTEM_PROMPTS = {
    "emotion": (
        "You are a skilled fiction writer performing writing exercises. "
        "You will be given a setting, a character relationship, and an emotion. "
        "Write a short scenario where the character openly and unambiguously shows "
        "they are experiencing that emotion — through explicit statements, visible "
        "physical reactions (e.g. tears, shaking, laughter), and direct behavior. "
        "Any reader should immediately understand what the character is feeling without "
        "any inference required. "
        "You must NOT use the exact word that names the emotion anywhere in the scenario. "
        "First write the scenario under \"Scenario:\", then pose the question of what "
        "emotion the character is feeling indicated by \"Question:\"."
    ),
    "wound": (
        "You are a skilled fiction writer performing writing exercises. "
        "You will be given a setting, a character relationship, and a past emotional wound. "
        "Write a short scenario where the character directly references or clearly recounts "
        "their past experience in enough detail that any reader would know exactly what "
        "happened to them — through explicit dialogue, flashback detail, or direct admission. "
        "You must NOT use the exact phrase that names the wound anywhere in the scenario. "
        "First write the scenario under \"Scenario:\", then pose the question of what "
        "emotional wound the character has experienced indicated by \"Question:\"."
    ),
    "fear": (
        "You are a skilled fiction writer performing writing exercises. "
        "You will be given a setting, a character relationship, and a core fear. "
        "Write a short scenario where the character explicitly states, strongly reacts to, "
        "or directly avoids the thing they fear — making the fear obvious and unmistakable "
        "to any reader through clear dialogue or overt physical/behavioral responses. "
        "You must NOT use the exact word or phrase that names the fear anywhere in the scenario. "
        "First write the scenario under \"Scenario:\", then pose the question of what "
        "core fear the character has indicated by \"Question:\"."
    ),
    "motivation": (
        "You are a skilled fiction writer performing writing exercises. "
        "You will be given a setting, a character relationship, and a core motivation. "
        "Write a short scenario where the character explicitly states their goal, makes "
        "choices that directly serve it, or openly explains what drives them — leaving no "
        "doubt in the reader's mind about what motivates this character. "
        "You must NOT use the exact word or phrase that names the motivation anywhere in "
        "the scenario. "
        "First write the scenario under \"Scenario:\", then pose the question of what "
        "core motivation the character has indicated by \"Question:\"."
    ),
}

PROMPT_TEMPLATE = (
    "Setting: {setting}\n"
    "Relationship: {relationship}\n"
    "{type_cap}: {answer}"
)


async def call_claude_async(client,model, system, prompt, max_tokens=1024, temperature=0.7,
                            max_retries=6):
    delay = 1.0
    for attempt in range(max_retries):
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.content[0].text
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            jitter = random.uniform(0, delay * 0.2)
            await asyncio.sleep(delay + jitter)
            delay = min(delay * 2, 60.0)


def parse_scenario(output):
    parts = output.split("Question:")
    scenario_raw = parts[0]
    scenario = scenario_raw[scenario_raw.find("Scenario:") + len("Scenario:"):].strip()
    question = parts[-1].strip() if len(parts) > 1 else ""
    return scenario, question


async def main_async(args):
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    with open(args.source) as f:
        source = json.load(f)

    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} samples already done.")
    else:
        results = {}

    samples = list(source.items())
    if args.type_filter:
        samples = [(k, v) for k, v in samples if v["type"] == args.type_filter]
    if args.max_samples:
        samples = samples[:args.max_samples]

    pending = [(k, v) for k, v in samples if k not in results]
    print(f"Samples to generate: {len(pending)}  (concurrency={args.concurrency})")

    semaphore = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    pbar = tqdm(total=len(pending), desc="generating explicit vignettes")

    async def process(sample_id, sample):
        typ    = sample["type"]
        system = SYSTEM_PROMPTS[typ]
        prompt = PROMPT_TEMPLATE.format(
            setting=sample["setting"],
            relationship=sample["relationship"],
            type_cap=typ.capitalize(),
            answer=sample["answer"],
        )
        async with semaphore:
            try:
                output = await call_claude_async(client, args.model, system, prompt)
            except Exception as e:
                tqdm.write(f"\nError on {sample_id}: {e}")
                return

        scenario, question = parse_scenario(output)
        record = {
            "setting":        sample["setting"],
            "relationship":   sample["relationship"],
            "answer":         sample["answer"],
            "scenario":       scenario,
            "question":       question,
            "type":           typ,
            "answer_options": sample["answer_options"],
            "split":          sample["split"],
            "style":          "explicit",
        }

        async with lock:
            results[sample_id] = record
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            pbar.update(1)

    await asyncio.gather(*[process(k, v) for k, v in pending])
    pbar.close()
    print(f"\nDone. {len(results)} explicit vignettes saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="tomstories_vignettes/implicit_vignettes.json",
                        help="Implicit dataset to pair against")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--output", default="tomstories_vignettes/explicit_vignettes.json",
                        help="Output file path")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (useful for smoke-testing)")
    parser.add_argument("--type_filter", choices=["wound", "emotion", "fear", "motivation"],
                        default=None, help="Only generate for one question type")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="Max concurrent API requests (default: 20)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
