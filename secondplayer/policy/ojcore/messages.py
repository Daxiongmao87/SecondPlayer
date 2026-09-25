"""Choice-question prompt rendering for the baked-in OpenJev readout.

Option lines (``A. name: description``) and the
``Question:/Options:/Answer with one letter:`` suffix follow the OpenJev
Choice rendering shape; the system wording is reconstructed. Fidelity to the
oracle is settled by the agreement gate (SPEC-OPENJEV-INPROCESS.md section 4),
not by this file's comments. PROMPT_VERSION marks the template revision.
"""

from __future__ import annotations

from typing import Any

from ...errors import ModelError

PROMPT_VERSION = "ojcore-choice-v1"

LETTERS = [chr(code) for code in range(ord("A"), ord("P") + 1)]

SYSTEM_INSTRUCTIONS = (
    "You are a decision readout. Read the state and the question, then answer "
    "with exactly one letter and nothing else."
)


def options_block(options: list[dict[str, Any]]) -> str:
    """Lettered ``name: description`` lines; at most 16 options."""
    if len(options) > len(LETTERS):
        raise ModelError(f"{len(options)} options exceed the {len(LETTERS)} letter slots")
    lines = []
    for letter, opt in zip(LETTERS, options):
        name = opt.get("id", opt.get("name"))
        if not name:
            raise ModelError("option is missing its id")
        desc = (opt.get("description") or "").strip()
        lines.append(f"{letter}. {name}: {desc}" if desc else f"{letter}. {name}")
    return "\n".join(lines)


def render_suffix(question: str, options: list[dict[str, Any]]) -> str:
    """Question + lettered options + single-letter answer cue."""
    letters = ", ".join(LETTERS[: len(options)])
    return f"Question: {question}\n\nOptions:\n{options_block(options)}\n\nAnswer with one letter: {letters}."


def direct_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Row (id/state/question/options) -> chat messages for the readout."""
    state = row.get("state")
    question = row.get("question")
    options = row.get("options")
    if not isinstance(state, str) or not state.strip():
        raise ModelError("row is missing state text")
    if not isinstance(question, str) or not question.strip():
        raise ModelError("row is missing its question")
    if not isinstance(options, list) or not options:
        raise ModelError("row has no options")
    user = f"{state.strip()}\n\n{render_suffix(question.strip(), options)}"
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": user},
    ]
