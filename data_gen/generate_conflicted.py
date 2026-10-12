"""
Generate the conflicted vignettes: scenes whose dialogue points to one mental state while
the narrated behavior points to another, for measuring modality bias.

Outputs:
  tomstories_vignettes/conflicted_vignettes.json   (--output), rewritten as samples complete
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
from common.prompts import TYPE_TO_THESAURUS_KEY, TYPES

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

def build_combos(thesaurus, types, train_per_type, val_per_type, rng):
    combos = []
    for typ in types:
        answer_pool = thesaurus[TYPE_TO_THESAURUS_KEY[typ]]
        for i in range(train_per_type + val_per_type):
            d_ans, n_ans = rng.sample(answer_pool, 2)   # dialogue, narration
            combos.append({
                "type":               typ,
                "setting":            rng.choice(thesaurus["settings"]),
                "relationship":       rng.choice(thesaurus["relationships"]),
                "dialogue_answer":    d_ans,
                "description_answer": n_ans,
                "answer_options":     sample_answer_options(thesaurus, typ,
                                                            [d_ans, n_ans], rng),
                "split":              "train" if i < train_per_type else "val",
            })
    return combos


def build_prompt(combo):
    typ = combo["type"]
    system = SYSTEM_PROMPTS[typ].format(
        dialogue_answer=combo["dialogue_answer"],
        description_answer=combo["description_answer"],
    )
    prompt = USER_PROMPT_TEMPLATE.format(
        setting=combo["setting"],
        relationship=combo["relationship"],
        type_cap=typ.capitalize(),
        dialogue_answer=combo["dialogue_answer"],
        description_answer=combo["description_answer"],
    )
    return system, prompt


def build_record(combo, scenario, question):
    return {
        "setting":            combo["setting"],
        "relationship":       combo["relationship"],
        "type":               combo["type"],
        "dialogue_answer":    combo["dialogue_answer"],
        "description_answer": combo["description_answer"],
        "scenario":           scenario,
        "question":           question,
        "answer_options":     combo["answer_options"],
        "split":              combo["split"],
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
                             desc="generating conflicted vignettes", temperature=0.8)
    print(f"\nDone. {len(results)} conflicted vignettes saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_generation_args(parser, "tomstories_vignettes/conflicted_vignettes.json")
    add_combo_args(parser)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
