"""
S-NIAH (Single Needle-in-a-Haystack) benchmark generator.

Based on the RULER benchmark (Hsieh et al., 2024) used in the RLM paper.
Generates synthetic tasks where a specific phrase/number is hidden in a large
body of unrelated text, and the model must find it.
"""

import hashlib
import random
import string
from dataclasses import dataclass
from pathlib import Path

ESSAY_SENTENCES = [
    "The most important thing is to be able to think clearly.",
    "It's not that people are dumb, it's that the world is hard.",
    "Technology changes fast, but people change slow.",
    "The best way to predict the future is to create it.",
    "When you have a hammer, everything looks like a nail.",
    "Simplicity is the ultimate sophistication in design.",
    "Great companies are built on solving real problems for people.",
    "The difference between a good idea and a great one is execution.",
    "Innovation often comes from connecting ideas across different fields.",
    "The hardest part of building something new is knowing what to build.",
    "Markets are efficient but entrepreneurs find the inefficiencies.",
    "Culture eats strategy for breakfast in any organization.",
    "The best code is the code you never have to write.",
    "Distribution is just as important as the product itself.",
    "Startups succeed not by being first but by being right.",
    "Every expert was once a beginner who refused to give up.",
    "The internet has made information free but wisdom is still scarce.",
    "Speed of iteration beats quality of iteration almost every time.",
    "Your network is your net worth in the knowledge economy.",
    "The future belongs to those who learn more skills and combine them creatively.",
    "Constraints breed creativity when you embrace them fully.",
    "Data without context is just noise in a crowded room.",
    "The map is not the territory and the model is not reality.",
    "Feedback loops are the most powerful force in any system.",
    "Trust is built in drops and lost in buckets over time.",
    "Complexity is the enemy of security in any software system.",
    "The most productive people are not the busiest ones.",
    "Automation should augment human capability not replace it.",
    "Good judgment comes from experience and experience comes from bad judgment.",
    "The cost of being wrong is usually less than the cost of doing nothing.",
]

ADJECTIVES = [
    "crimson", "azure", "golden", "silver", "emerald", "violet", "amber",
    "scarlet", "cobalt", "ivory", "obsidian", "copper", "jade", "coral",
    "sapphire", "ruby", "pearl", "onyx", "bronze", "crystal",
]

NOUNS = [
    "elephant", "falcon", "dolphin", "tiger", "phoenix", "dragon",
    "panther", "whale", "eagle", "wolf", "lion", "bear", "hawk",
    "serpent", "owl", "fox", "raven", "stag", "shark", "lynx",
]


@dataclass
class NIAHTask:
    context: str
    query: str
    answer: str
    needle_position: float
    context_lines: int


def generate_key() -> str:
    adj = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)
    return f"{adj}-{noun}"


def generate_value() -> str:
    return str(random.randint(1000000, 9999999))


def generate_haystack(num_lines: int, rng: random.Random) -> list[str]:
    lines = []
    for _ in range(num_lines):
        sent = rng.choice(ESSAY_SENTENCES)
        lines.append(sent)
    return lines


def generate_task(
    num_lines: int = 500,
    needle_position: float | None = None,
    seed: int | None = None,
) -> NIAHTask:
    rng = random.Random(seed)

    key = generate_key()
    value = generate_value()
    needle = f"The special magic number for {key} is: {value}."

    if needle_position is None:
        needle_position = rng.random()

    haystack = generate_haystack(num_lines, rng)
    insert_idx = int(needle_position * len(haystack))
    insert_idx = max(0, min(insert_idx, len(haystack) - 1))
    haystack.insert(insert_idx, needle)

    context = "\n".join(haystack)
    query = (
        f"A special magic number is hidden within the following text. "
        f"Find the special magic number for {key} mentioned in the text. "
        f"Give your final answer in the form 'Answer: <number>'."
    )

    return NIAHTask(
        context=context,
        query=query,
        answer=value,
        needle_position=needle_position,
        context_lines=len(haystack),
    )


def evaluate(predicted: str, expected: str) -> float:
    predicted_clean = predicted.strip().lower()
    expected_clean = expected.strip().lower()
    if expected_clean in predicted_clean:
        return 1.0
    return 0.0


def generate_dataset(
    num_tasks: int = 50,
    num_lines: int = 500,
    seed: int = 42,
) -> list[NIAHTask]:
    tasks = []
    for i in range(num_tasks):
        position = (i + 0.5) / num_tasks
        task = generate_task(
            num_lines=num_lines,
            needle_position=position,
            seed=seed + i,
        )
        tasks.append(task)
    return tasks
