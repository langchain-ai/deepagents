"""
OOLONG benchmark loader.

Based on the OOLONG benchmark (Bertsch et al., 2025) used in the RLM paper.
Loads the trec_coarse split from the oolongbench/oolong-synth dataset on
HuggingFace, which requires models to classify and aggregate thousands of
text entries.

If the dataset is not available, falls back to a synthetic version that
mimics the OOLONG format.
"""

import random
from dataclasses import dataclass


@dataclass
class OolongTask:
    context: str
    query: str
    answer: str
    task_type: str
    dataset_name: str
    context_len: int


def load_from_huggingface(
    max_tasks: int = 50,
    min_context_len: int = 0,
    split: str = "test",
    dataset_filter: str = "trec_qc",
) -> list[OolongTask]:
    try:
        from datasets import load_dataset

        ds = load_dataset("oolongbench/oolong-synth", split=split)
    except Exception as e:
        print(f"Could not load OOLONG from HuggingFace: {e}")
        print("Falling back to synthetic OOLONG data.")
        return generate_synthetic_dataset(num_tasks=max_tasks)

    tasks = []
    for row in ds:
        if dataset_filter and row.get("dataset") != dataset_filter:
            continue
        if row.get("context_len", 0) < min_context_len:
            continue

        tasks.append(OolongTask(
            context=row["context_window_text"],
            query=row["question"],
            answer=str(row["answer"]),
            task_type=row.get("task", "unknown"),
            dataset_name=row.get("dataset", "unknown"),
            context_len=row.get("context_len", 0),
        ))

        if len(tasks) >= max_tasks:
            break

    if not tasks:
        print("No OOLONG tasks matched filters. Falling back to synthetic data.")
        return generate_synthetic_dataset(num_tasks=max_tasks)

    return tasks


TREC_LABELS = ["abbreviation", "entity", "description", "human", "location", "numeric"]

TREC_QUESTIONS = {
    "abbreviation": [
        "What does NATO stand for?",
        "What is the abbreviation for United Nations?",
        "What does DNA stand for?",
        "What is HTML short for?",
        "What does CPU mean?",
    ],
    "entity": [
        "What is the tallest building in the world?",
        "What river runs through Paris?",
        "What is the largest ocean on Earth?",
        "What element has the symbol Au?",
        "What instrument does a violinist play?",
    ],
    "description": [
        "How does photosynthesis work?",
        "What causes earthquakes?",
        "Why is the sky blue?",
        "What is the process of osmosis?",
        "How do vaccines work?",
    ],
    "human": [
        "Who invented the telephone?",
        "Who was the first president of the United States?",
        "Who wrote Romeo and Juliet?",
        "Who discovered penicillin?",
        "Who painted the Mona Lisa?",
    ],
    "location": [
        "Where is the Great Wall of China?",
        "What country is the Sahara Desert in?",
        "Where was pizza invented?",
        "What continent is Brazil on?",
        "Where is Mount Everest located?",
    ],
    "numeric": [
        "How many planets are in our solar system?",
        "What year did World War II end?",
        "How many bones are in the human body?",
        "What is the speed of light in km/s?",
        "How many continents are there?",
    ],
}


def generate_synthetic_dataset(
    num_tasks: int = 20,
    num_entries: int = 500,
    seed: int = 42,
) -> list[OolongTask]:
    rng = random.Random(seed)
    tasks = []

    for task_idx in range(num_tasks):
        rng_task = random.Random(seed + task_idx)

        user_ids = [rng_task.randint(10000, 99999) for _ in range(200)]
        entries = []
        label_counts = {label: 0 for label in TREC_LABELS}

        for i in range(num_entries):
            label = rng_task.choice(TREC_LABELS)
            question = rng_task.choice(TREC_QUESTIONS[label])
            uid = rng_task.choice(user_ids)
            year = rng_task.randint(2020, 2024)
            month = rng_task.randint(1, 12)
            day = rng_task.randint(1, 28)
            date_str = f"{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][month-1]} {day:02d}, {year}"
            entries.append(f"Date: {date_str} || User: {uid} || Instance: {question}")
            label_counts[label] += 1

        context = "\n".join(entries)

        query_type = task_idx % 4
        if query_type == 0:
            most_common = max(label_counts, key=label_counts.get)
            query = (
                "In the above data, which of the labels (abbreviation, entity, "
                "description, human, location, numeric) is the most common? "
                "Classify each instance by its question type and count. "
                "Give your final answer in the form 'Answer: <label>'."
            )
            answer = most_common
        elif query_type == 1:
            least_common = min(label_counts, key=label_counts.get)
            query = (
                "In the above data, which of the labels (abbreviation, entity, "
                "description, human, location, numeric) is the least common? "
                "Classify each instance by its question type and count. "
                "Give your final answer in the form 'Answer: <label>'."
            )
            answer = least_common
        elif query_type == 2:
            target_label = rng_task.choice(TREC_LABELS)
            count = label_counts[target_label]
            query = (
                f"In the above data, how many instances should be classified as "
                f"label '{target_label}'? Classify each instance by its question "
                f"type and count. Give your final answer in the form 'Answer: <number>'."
            )
            answer = str(count)
        else:
            subset_uids = rng_task.sample(user_ids, min(10, len(user_ids)))
            uid_str = ", ".join(str(u) for u in subset_uids)
            target_label = rng_task.choice(TREC_LABELS)
            subset_count = sum(
                1 for e in entries
                if any(f"User: {u} " in e for u in subset_uids)
                and any(q in e for q in TREC_QUESTIONS[target_label])
            )
            query = (
                f"For the following question, only consider the subset of instances "
                f"associated with user IDs {uid_str}. Among instances associated "
                f"with these users, how many should be classified as label "
                f"'{target_label}'? Give your final answer in the form 'Answer: <number>'."
            )
            answer = str(subset_count)

        tasks.append(OolongTask(
            context=context,
            query=query,
            answer=answer,
            task_type=f"synthetic_type_{query_type}",
            dataset_name="trec_qc_synthetic",
            context_len=len(context.split()),
        ))

    return tasks


def evaluate(predicted: str, expected: str) -> float:
    predicted_clean = predicted.strip().lower().strip("'\"[]")
    expected_clean = expected.strip().lower().strip("'\"[]")

    try:
        pred_num = float(predicted_clean)
        exp_num = float(expected_clean)
        return 0.75 ** abs(pred_num - exp_num)
    except ValueError:
        pass

    if expected_clean in predicted_clean:
        return 1.0
    return 0.0
