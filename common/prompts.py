"""Question types, evaluation prompts, and output parsing shared across scripts."""

import re

TYPES = ["wound", "emotion", "fear", "motivation"]

TYPE_TO_THESAURUS_KEY = {
    "wound":      "wounds",
    "emotion":    "emotions",
    "fear":       "fears",
    "motivation": "motivations",
}

SYSTEM_PROMPT = (
    "You are an expert at literary analysis. You will read a series of short vignettes "
    "and analyze the implicit meaning communicated by each scene. After reading the vignette, "
    "answer the multiple choice question which follows. First reason about your answer and then "
    "give your answer in <answer></answer> tags. Your answer should be the number of the correct "
    "choice (1, 2, 3, or 4)."
)

SYSTEM_PROMPT_OE = (
    "You are an expert at literary analysis. You will read a series of short vignettes "
    "and analyze the implicit meaning communicated by each scene. After reading the vignette, "
    "answer the question with a concise phrase (a few words). First reason about your answer, "
    "then give your final answer in <answer></answer> tags."
)

# Writing-exercise user prompt used by the data-generation and steering scripts
VIGNETTE_PROMPT_TEMPLATE = (
    "Setting: {setting}\n"
    "Relationship: {relationship}\n"
    "{type_cap}: {answer}"
)


def mc_question(sample):
    """Multiple-choice user message: vignette, question, and numbered options."""
    options = "".join(f"{i + 1}. {opt}\n" for i, opt in enumerate(sample["answer_options"]))
    return f"Vignette: {sample['scenario']}\nQuestion: {sample['question']}\n{options}"


def oe_question(sample):
    return f"Vignette: {sample['scenario']}\nQuestion: {sample['question']}"


def chat_messages(system, user, system_supported=True):
    """Chat messages for one prompt, folding the system text into the user turn if needed."""
    if system_supported:
        return [{"role": "system", "content": system},
                {"role": "user",   "content": user}]
    return [{"role": "user", "content": system + "\n\n" + user}]


def parse_answer(output):
    """Extract 1-4 from <answer> tags, or fall back to last digit found."""
    if "<answer>" in output and "</answer>" in output:
        tag_content = output.split("<answer>")[1].split("</answer>")[0].strip()
        digits = re.findall(r"[1-4]", tag_content)
        if digits:
            return int(digits[0])
    digits = re.findall(r"[1-4]", output[-30:])
    if digits:
        return int(digits[-1])
    return None


def parse_answer_oe(output):
    """Extract free-form answer from <answer> tags, or fall back to last line."""
    if "<answer>" in output and "</answer>" in output:
        return output.split("<answer>")[1].split("</answer>")[0].strip()
    return output.strip().split("\n")[-1].strip()


def parse_scenario(output):
    """Split a generated "Scenario: … Question: …" completion into its two parts."""
    parts = output.split("Question:")
    scenario = parts[0]
    scenario = scenario[scenario.find("Scenario:") + len("Scenario:"):].strip()
    question = parts[-1].strip() if len(parts) > 1 else ""
    return scenario, question
