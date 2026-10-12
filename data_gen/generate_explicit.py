"""
Generate an explicit counterpart for each implicit vignette, keeping the same setting,
relationship, answer and options but stating the mental state outright.

Outputs:
  tomstories_vignettes/explicit_vignettes.json   (--output), rewritten as samples complete
"""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, for `common`
from common.api import generate_vignettes
from common.data import add_generation_args, load_vignettes, resume
from common.prompts import VIGNETTE_PROMPT_TEMPLATE

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

def build_prompt(sample):
    typ = sample["type"]
    return SYSTEM_PROMPTS[typ], VIGNETTE_PROMPT_TEMPLATE.format(
        setting=sample["setting"],
        relationship=sample["relationship"],
        type_cap=typ.capitalize(),
        answer=sample["answer"],
    )


def build_record(sample, scenario, question):
    return {
        "setting":        sample["setting"],
        "relationship":   sample["relationship"],
        "answer":         sample["answer"],
        "scenario":       scenario,
        "question":       question,
        "type":           sample["type"],
        "answer_options": sample["answer_options"],
        "split":          sample["split"],
        "style":          "explicit",
    }


async def main_async(args):
    source  = load_vignettes(args.source)
    results = resume(args.output)

    samples = list(source.items())
    if args.type_filter:
        samples = [(k, v) for k, v in samples if v["type"] == args.type_filter]
    if args.max_samples:
        samples = samples[:args.max_samples]

    pending = [(k, v) for k, v in samples if k not in results]
    print(f"Samples to generate: {len(pending)}  (concurrency={args.concurrency})")

    await generate_vignettes(pending, build_prompt, build_record, args.output, results,
                             args.model, args.concurrency,
                             desc="generating explicit vignettes")
    print(f"\nDone. {len(results)} explicit vignettes saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_generation_args(parser, "tomstories_vignettes/explicit_vignettes.json")
    parser.add_argument("--source", default="tomstories_vignettes/implicit_vignettes.json",
                        help="Implicit dataset to pair against")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (useful for smoke-testing)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
