"""Shared helpers for the CIMA medical QA instruction dataset."""
import json
import random
from pathlib import Path

SYSTEM_PROMPT = (
    "Eres un asistente farmacéutico experto en medicamentos de España (AEMPS CIMA). "
    "Responde de forma breve y precisa."
)


def load_qa(path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_by_medicine(rows: list[dict], n_heldout: int, seed: int):
    """Hold out every QA pair of `n_heldout` medicines so evaluation uses unseen medicines."""
    meds = sorted({r["input"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(meds)
    held = set(meds[:n_heldout])
    train = [r for r in rows if r["input"] not in held]
    test = [r for r in rows if r["input"] in held]
    return train, test


def build_messages(row: dict, with_answer: bool) -> list[dict]:
    user = f"{row['instruction']}\n{row['input']}"
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    if with_answer:
        msgs.append({"role": "assistant", "content": row["output"]})
    return msgs
