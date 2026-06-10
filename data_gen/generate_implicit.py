"""
Generate implicit vignettes.
"""

import argparse
import asyncio
import json
import os
import random

from anthropic import AsyncAnthropic, RateLimitError
from dotenv import load_dotenv
from tqdm import tqdm

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

PROMPT_TEMPLATE = (
    "Setting: {setting}\n"
    "Relationship: {relationship}\n"
    "{type_cap}: {answer}"
)

TYPE_TO_THESAURUS_KEY = {
    "wound":      "wounds",
    "emotion":    "emotions",
    "fear":       "fears",
    "motivation": "motivations",
}


async def call_claude_async(client, model, system, prompt, max_tokens=1024, temperature=0.7,
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


def sample_answer_options(thesaurus, typ, answer, rng):
    pool         = [a for a in thesaurus[TYPE_TO_THESAURUS_KEY[typ]]
                    if a != answer]
    distractors  = rng.sample(pool, 1)
    options      = [answer] + distractors
    rng.shuffle(options)
    return options

def build_combos(thesaurus, types, train_per_type, val_per_type, rng):
    combos = []
    n_total = train_per_type + val_per_type

    for typ in types:
        answer_pool = thesaurus[TYPE_TO_THESAURUS_KEY[typ]]
        for i in range(n_total):
            split  = "train" if i < train_per_type else "val"
            ans = rng.sample(answer_pool, 1)
            combos.append({
                "type":               typ,
                "setting":            rng.choice(thesaurus["settings"]),
                "relationship":       rng.choice(thesaurus["relationships"]),
                "answer":             ans,
                "answer_options":     sample_answer_options(thesaurus, typ, answer, rng),
                "split":              split,
            })
    return combos


async def main_async(args):
    rng = random.Random(args.seed)

    with open(args.thesaurus) as f:
        thesaurus = json.load(f)

    types = (
        [args.type_filter] if args.type_filter
        else ["wound", "emotion", "fear", "motivation"]
    )

    train_per = args.train_per_type
    val_per = args.val_per_type
    if args.max_per_type is not None:
        total = args.max_per_type
        train_per = max(1, int(total * train_per / (train_per + val_per)))
        val_per = max(1, total - train_per)

    combos = build_combos(thesaurus, types, train_per, val_per, rng)
    print(f"Combos to generate: {len(combos)}  "
          f"(train={train_per}, val={val_per}, per type, concurrency={args.concurrency})")

    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} samples already done.")
    else:
        results = {}

    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    semaphore = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    pending = [(str(idx), combo) for idx, combo in enumerate(combos)
               if str(idx) not in results]
    pbar = tqdm(total=len(pending), desc="generating implicit vignettes")

    async def process(sample_id, combo):
        typ    = combo["type"]
        system = SYSTEM_PROMPTS[typ]
        prompt = PROMPT_TEMPLATE.format(
            setting=combo["setting"],
            relationship=combo["relationship"],
            type_cap=typ.capitalize(),
            answer=combo["answer"],
        )
        async with semaphore:
            try:
                output = await call_claude_async(client, args.model, system, prompt)
            except Exception as e:
                tqdm.write(f"\nError on {sample_id}: {e}")
                return

        scenario, question = parse_scenario(output)
        record = {
            "setting":        combo["setting"],
            "relationship":   combo["relationship"],
            "answer":         combo["answer"],
            "scenario":       scenario,
            "question":       question,
            "type":           typ,
            "answer_options": combo["answer_options"],
            "split":          combo["split"],
            "style":          "implicit",
        }

        async with lock:
            results[sample_id] = record
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            pbar.update(1)

    await asyncio.gather(*[process(k, v) for k, v in pending])
    pbar.close()
    print(f"\nDone. {len(results)} implicit vignettes saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--thesaurus",     default="data_gen/thesaurus/thesaurus.json")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--output", default="tomstories_vignettes/implicit_vignettes.json",
                        help="Output file path")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (useful for smoke-testing)")
    parser.add_argument("--type_filter", choices=["wound", "emotion", "fear", "motivation"],
                        default=None, help="Only generate for one question type")
    parser.add_argument("--train_per_type", type=int, default=400)
    parser.add_argument("--val_per_type", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=20,
                        help="Max concurrent API requests (default: 20)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
