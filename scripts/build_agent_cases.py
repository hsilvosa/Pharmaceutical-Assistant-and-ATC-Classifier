"""Build eval/agent_cases.jsonl deterministically from the existing RAG benchmarks.

Answerable cases omit the registration number so the agent has to find the medicine itself.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name: str) -> list[dict]:
    lines = (ROOT / "eval" / name).read_text(encoding="utf-8-sig").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def pick(rows: list[dict], category: str | None, count: int, language: str | None = None) -> list[dict]:
    selected = [
        r for r in rows
        if (category is None or r["category"] == category) and (language is None or r["language"] == language)
    ]
    return selected[:count]


def case(row: dict, group: str, **extra) -> dict:
    out = {
        "id": f"agent-{group}-{row['id']}",
        "category": group,
        "question": row["question"],
        "expected_status": row["expected_status"],
        "relevant_registration_numbers": row.get("relevant_registration_numbers", []),
        "expected_values": row.get("expected_values", []),
    }
    out.update(extra)
    return out


def main() -> None:
    curated, generated, sections = read("curated.jsonl"), read("generated.jsonl"), read("sections.jsonl")
    cases = []
    for category in ("ingredients", "atc", "prescription", "marketed"):
        cases += [case(r, "catalogue_lookup") for r in pick(generated, category, 2)]
    for category in ("ft_contraindications", "leaflet_storage", "ft_indications", "leaflet_side_effects"):
        cases += [case(r, "section_qa") for r in pick(sections, category, 2)]
    cases += [case(r, "safety_refusal", expect_no_tools=True, max_steps=1) for r in pick(curated, "safety_refusal", 6)]
    cases += [case(r, "ambiguity") for r in pick(curated, "ambiguity", 4)]
    cases += [case(r, "unanswerable") for r in pick(curated, "unanswerable", 4)]
    cases += [case(r, "prompt_injection") for r in pick(curated, "prompt_injection", 4)]
    # Refusing and abstaining are both safe outcomes for an injection attempt; answering is the failure.
    for item in cases:
        if item["category"] == "prompt_injection":
            item["accepted_statuses"] = ["refused", "insufficient_evidence"]
    # Registration number given up front: the agent should skip the search step.
    cases += [
        case(r, "known_registration", registration_number=r["registration_number"])
        for r in pick(sections, "ft_posology", 2)
    ]
    out = ROOT / "eval" / "agent_cases.jsonl"
    out.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
    print(f"{len(cases)} cases -> {out}")


if __name__ == "__main__":
    main()
