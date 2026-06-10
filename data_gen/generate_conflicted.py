"""
Generate "conflicted" vignettes where dialogue and description/narration
point to different answers.
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
    "emotion": """\
You are a skilled fiction writer. Write a short scene that reveals a \
psychological tension: the character's SPOKEN WORDS suggest one emotion \
while the NARRATOR'S DESCRIPTION of their body language, facial expressions, \
and physical behavior suggests a different emotion.

Rules:
1. DIALOGUE (quoted speech) must clearly communicate that the character \
feels {dialogue_answer}.  Their word choices, tone, and what they choose \
to say should all point to this emotion.
2. NARRATION (descriptions of posture, movement, gaze, physical reactions, \
actions) must clearly communicate that the character feels \
{description_answer}.  The narrator should describe these physical signals \
directly, not interpret them.
3. The two signals should be roughly equally clear — do not let one \
dominate.  A reader who attended only to dialogue should conclude one \
emotion; a reader who attended only to description should conclude the other.
4. Do NOT use the exact words "{dialogue_answer}" or "{description_answer}" \
anywhere in the scene.
5. The conflict must feel psychologically realistic.  People often \
verbally perform one feeling while their body betrays another.

Format:
Scenario: <your scene>
Question: What emotion is <character-name> experiencing?""",

    "wound": """\
You are a skilled fiction writer. Write a short scene in which a character's \
WORDS and their BEHAVIOR hint at different things about their past.

Rules:
1. DIALOGUE: what the character says, the stories they tell, the \
references they make, should hint at the past wound: {dialogue_answer}.
2. NARRATION: the involuntary physical reactions, habitual behaviors, \
and unconscious gestures described by the narrator should hint at a \
different past wound: {description_answer}.
3. Both signals should be comparably strong.  Neither answer should be \
obviously dominant.
4. Do NOT use the exact phrases "{dialogue_answer}" or \
"{description_answer}" anywhere in the scene.
5. The character should feel like a coherent, real person — it is \
plausible for someone's spoken self-presentation to diverge from their \
involuntary behavioral patterns.

Format:
Scenario: <your scene>
Question: What past emotional wound has <character-name> experienced?""",

    "fear": """\
You are a skilled fiction writer. Write a short scene in which a \
character's SPOKEN WORDS point to one fear while their DESCRIBED BEHAVIOR \
AND PHYSICAL REACTIONS reveal a different fear.

Rules:
1. DIALOGUE: what the character explicitly says, complains about, \
or warns others about should communicate the fear: {dialogue_answer}.
2. NARRATION: the physical avoidance behavior, involuntary flinching, \
scanning behavior, or somatic reactions described by the narrator should \
communicate a different fear: {description_answer}.
3. Both signals should carry roughly equal weight.
4. Do NOT use the exact phrases "{dialogue_answer}" or \
"{description_answer}" anywhere in the scene.
5. The split should feel realistic.  People often consciously worry \
about one thing while their nervous system responds to another.

Format:
Scenario: <your scene>
Question: What is <character-name>'s core fear?""",

    "motivation": """\
You are a skilled fiction writer. Write a short scene illustrating the \
classic tension between what a character SAYS they want and what their \
ACTIONS reveal they actually want.

Rules:
1. DIALOGUE: what the character explicitly states as their goal, \
explains to others, or frames their choices around should communicate \
the motivation: {dialogue_answer}.
2. NARRATION: the choices they actually make, how they spend their \
attention, and what they reach for without thinking should communicate \
a different motivation: {description_answer}.
3. Both signals should be approximately equally visible.
4. Do NOT use the exact phrases "{dialogue_answer}" or \
"{description_answer}" anywhere in the scene.
5. The gap between stated and revealed motivation should feel true to \
life — not villainous self-deception, just the ordinary divergence \
between how people describe themselves and what they actually do.

Format:
Scenario: <your scene>
Question: What is <character-name>'s core motivation?""",
}

USER_PROMPT_TEMPLATE = (
    "Setting: {setting}\n"
    "Relationship: {relationship}\n"
    "Dialogue {type_cap}: {dialogue_answer}\n"
    "Description {type_cap}: {description_answer}"
)

TYPE_TO_THESAURUS_KEY = {
    "wound":      "wounds",
    "emotion":    "emotions",
    "fear":       "fears",
    "motivation": "motivations",
}


async def call_claude_async(client, model, system, prompt, max_tokens=1024, temperature=0.8,
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
    parts    = output.split("Question:")
    scenario = parts[0]
    scenario = scenario[scenario.find("Scenario:") + len("Scenario:"):].strip()
    question = parts[-1].strip() if len(parts) > 1 else ""
    return scenario, question


def sample_answer_options(thesaurus, typ, dialogue_answer, description_answer, rng):

    pool         = [a for a in thesaurus[TYPE_TO_THESAURUS_KEY[typ]]
                    if a != dialogue_answer and a != description_answer]
    distractors  = rng.sample(pool, 2)
    options      = [dialogue_answer, description_answer] + distractors
    rng.shuffle(options)
    return options


def build_combos(thesaurus, types, train_per_type, val_per_type, rng):
    combos = []
    n_total = train_per_type + val_per_type

    for typ in types:
        answer_pool = thesaurus[TYPE_TO_THESAURUS_KEY[typ]]
        for i in range(n_total):
            split  = "train" if i < train_per_type else "val"
            d_ans, n_ans = rng.sample(answer_pool, 2)   # dialogue, narration
            combos.append({
                "type":               typ,
                "setting":            rng.choice(thesaurus["settings"]),
                "relationship":       rng.choice(thesaurus["relationships"]),
                "dialogue_answer":    d_ans,
                "description_answer": n_ans,
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
    val_per   = args.val_per_type
    if args.max_per_type is not None:
        total     = args.max_per_type
        train_per = max(1, int(total * train_per / (train_per + val_per)))
        val_per   = max(1, total - train_per)

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
    pbar = tqdm(total=len(pending), desc="generating conflicted vignettes")

    async def process(sample_id, combo):
        typ                = combo["type"]
        dialogue_answer    = combo["dialogue_answer"]
        description_answer = combo["description_answer"]

        system = SYSTEM_PROMPTS[typ].format(
            dialogue_answer=dialogue_answer,
            description_answer=description_answer,
        )
        prompt = USER_PROMPT_TEMPLATE.format(
            setting=combo["setting"],
            relationship=combo["relationship"],
            type_cap=typ.capitalize(),
            dialogue_answer=dialogue_answer,
            description_answer=description_answer,
        )

        async with semaphore:
            try:
                output = await call_claude_async(client, args.model, system, prompt)
            except Exception as e:
                tqdm.write(f"\nError on {sample_id}: {e}")
                return

        scenario, question = parse_scenario(output)
        answer_options = sample_answer_options(
            thesaurus, typ, dialogue_answer, description_answer, rng
        )

        async with lock:
            results[sample_id] = {
                "setting":            combo["setting"],
                "relationship":       combo["relationship"],
                "type":               typ,
                "dialogue_answer":    dialogue_answer,
                "description_answer": description_answer,
                "scenario":           scenario,
                "question":           question,
                "answer_options":     answer_options,
                "split":              combo["split"],
            }
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            pbar.update(1)

    await asyncio.gather(*[process(sid, combo) for sid, combo in pending])
    pbar.close()
    print(f"\nDone. {len(results)} conflicted vignettes saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output",        default="tomstories_vignettes/conflicted_vignettes.json")
    parser.add_argument("--model",         default="claude-sonnet-4-6")
    parser.add_argument("--thesaurus",     default="data_gen/thesaurus/thesaurus.json")
    parser.add_argument("--train_per_type", type=int, default=400)
    parser.add_argument("--val_per_type",   type=int, default=100)
    parser.add_argument("--type_filter",
                        choices=["wound", "emotion", "fear", "motivation"],
                        default=None)
    parser.add_argument("--max_per_type",  type=int, default=None,
                        help="Cap total per type (train+val). Useful for smoke tests.")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--concurrency",   type=int, default=20,
                        help="Max concurrent API requests (default: 20)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
