"""
Generate the implicit vignettes with the Anthropic API: short scenes that convey a mental
state through subtext without naming it, each with a four-choice question.

Outputs:
  tomstories_vignettes/implicit_vignettes.json   (--output), rewritten as samples complete
"""

import argparse
import asyncio
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, for `common`
from common.api import generate_vignettes
from common.data import (add_combo_args, add_generation_args, load_json, resume,
                         sample_answer_options, split_sizes)
from common.prompts import TYPE_TO_THESAURUS_KEY, TYPES, VIGNETTE_PROMPT_TEMPLATE

load_dotenv()


SYSTEM_PROMPTS = {
    "emotion": (
        "You are a skilled fiction writer. You are performing writing exercises that ask "
        "you to complete short scenarios given a setting, character relationship, and emotion. "
        "You need to write a scenario that does not explicitly mention the emotion but "
        "communicates that one of the characters is feeling that emotion, while adhering to "
        "the setting and character relationship provided. "
        "First write the scenario indicated by \"Scenario:\", then pose the question of what "
        "emotion the character is feeling indicated by \"Question:\"."
    ),
    "wound": (
        "You are a skilled fiction writer. You are performing writing exercises that ask "
        "you to complete short scenarios given a setting, character relationship, and past "
        "emotional wound. You need to write a scenario that does not explicitly mention the "
        "emotional wound but communicates that one of the characters likely experienced it "
        "in their past, while adhering to the setting and character relationship provided. "
        "First write the scenario indicated by \"Scenario:\", then pose the question of what "
        "emotional wound the character has experienced indicated by \"Question:\"."
    ),
    "fear": (
        "You are a skilled fiction writer. You are performing writing exercises that ask "
        "you to complete short scenarios given a setting, character relationship, and fear. "
        "You need to write a scenario that does not explicitly mention the fear but "
        "communicates that one of the characters has this core fear, while adhering to the "
        "setting and character relationship provided. "
        "First write the scenario indicated by \"Scenario:\", then pose the question of what "
        "core fear the character has indicated by \"Question:\"."
    ),
    "motivation": (
        "You are a skilled fiction writer. You are performing writing exercises that ask "
        "you to complete short scenarios given a setting, character relationship, and "
        "motivation. You need to write a scenario that does not explicitly mention the "
        "motivation but communicates that one of the characters has this core motivation, "
        "while adhering to the setting and character relationship provided. "
        "First write the scenario indicated by \"Scenario:\"d."
    ),
}

def build_combos(thesaurus, types, train_per_type, val_per_type, rng):
    combos = []
    for typ in types:
        answer_pool = thesaurus[TYPE_TO_THESAURUS_KEY[typ]]
        for i in range(train_per_type + val_per_type):
            answer = rng.choice(answer_pool)
            combos.append({
                "type":           typ,
                "setting":        rng.choice(thesaurus["settings"]),
                "relationship":   rng.choice(thesaurus["relationships"]),
                "answer":         answer,
                "answer_options": sample_answer_options(thesaurus, typ, [answer], rng),
                "split":          "train" if i < train_per_type else "val",
            })
    return combos


def build_prompt(combo):
    typ = combo["type"]
    return SYSTEM_PROMPTS[typ], VIGNETTE_PROMPT_TEMPLATE.format(
        setting=combo["setting"],
        relationship=combo["relationship"],
        type_cap=typ.capitalize(),
        answer=combo["answer"],
    )


def build_record(combo, scenario, question):
    return {
        "setting":        combo["setting"],
        "relationship":   combo["relationship"],
        "answer":         combo["answer"],
        "scenario":       scenario,
        "question":       question,
        "type":           combo["type"],
        "answer_options": combo["answer_options"],
        "split":          combo["split"],
        "style":          "implicit",
    }


async def main_async(args):
    rng       = random.Random(args.seed)
    thesaurus = load_json(args.thesaurus)
    types     = [args.type_filter] if args.type_filter else TYPES
    train_per, val_per = split_sizes(args.train_per_type, args.val_per_type,
                                     args.max_per_type)

    combos = build_combos(thesaurus, types, train_per, val_per, rng)
    print(f"Combos to generate: {len(combos)}  "
          f"(train={train_per}, val={val_per}, per type, concurrency={args.concurrency})")

    results = resume(args.output)
    pending = [(str(idx), combo) for idx, combo in enumerate(combos)
               if str(idx) not in results]

    await generate_vignettes(pending, build_prompt, build_record, args.output, results,
                             args.model, args.concurrency,
                             desc="generating implicit vignettes")
    print(f"\nDone. {len(results)} implicit vignettes saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_generation_args(parser, "tomstories_vignettes/implicit_vignettes.json")
    add_combo_args(parser)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
