import os
from typing import Iterable, Dict, Any

def generate_prompt_from_triples(
    triples: Iterable[Dict[str, Any]],
    instruction: str,
    template_path: str = "prompt_template.txt",
) -> str:
    """Fill in a prompt template using provided triples.

    Parameters
    ----------
    triples:
        Iterable of dictionaries with ``subject``, ``predicate`` and ``object``
        keys.
    instruction:
        Text instruction appended to the template.
    template_path:
        Location of the prompt template file.

    Returns
    -------
    str
        Completed prompt with triples and instruction inserted.
    """
    with open(template_path) as f:
        template = f.read()

    facts = ""
    for t in triples:
        facts += f"{t['subject']} — {t['predicate']} — {t['object']}\n"

    return template.replace("<<FACTS>>", facts.strip()).replace("<<INSTRUCTION>>", instruction)


def compute_reward(generated_input: str, ground_truth_input: str) -> float:
    """Compute a reward for generated content.

    Parameters
    ----------
    generated_input:
        Text produced by the language model.
    ground_truth_input:
        Reference `.in` file text used as ground truth.

    Returns
    -------
    float
        Jaccard similarity between generated and reference files.
    """
    gen_lines = set(generated_input.strip().splitlines())
    gt_lines = set(ground_truth_input.strip().splitlines())
    overlap = gen_lines.intersection(gt_lines)
    union = gen_lines.union(gt_lines)
    return len(overlap) / len(union) if union else 0.0


def generate_qe_input_from_prompt(
    prompt: str,
    openai_client: Any,
    temperature: float = 0.7,
) -> str:
    """Query a language model with ``prompt`` and return the output.

    Parameters
    ----------
    prompt:
        The text prompt to send to the language model.
    openai_client:
        Initialized OpenAI client with a ``chat.completions.create`` method.
    temperature:
        Sampling temperature for the model.

    Returns
    -------
    str
        Generated QE input file content or an empty string on error.
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[!] Prompt failed: {e}")
        return ""

