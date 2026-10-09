"""
Compare the base model with the fine-tuned adapter on held-out medicines.

Metric: for each catalogue field in the reference answer (active ingredient, dosage form,
route, prescription status, first excipient), check whether the generated answer contains it.
The prompt gives only the medicine name, so this measures what the model can infer or
recall without any retrieved context. It does not measure answer safety or leaflet QA.
"""
from src.utils import setup_environment, setup_logger
setup_environment()

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import PROCESSED_DATA_DIR, LLM_MODEL_DIR, LLM_BASE_MODEL
from src.medical_qa import build_messages, load_qa

logger = setup_logger("evaluate_medical_llm")


def expected_fields(output: str) -> dict:
    """Extract checkable catalogue values from a templated reference answer."""
    f = {}
    if m := re.search(r"principio activo principal: (.+?)\. Su forma", output):
        f["active_ingredient"] = m.group(1)
    if m := re.search(r"Su forma farmacéutica es (.+?)\. ", output):
        f["form"] = m.group(1)
    if m := re.search(r"en forma de (.+?) y se administra por vía (.+?)\. Debe", output):
        f["form"], f["route"] = m.group(1), m.group(2).replace("VÍA ", "")
    if "No requiere receta" in output:
        f["prescription"] = "no requiere receta"
    elif "Requiere receta" in output:
        f["prescription"] = "requiere receta"
    if m := re.search(r"excipientes declarados: (.+?)(?: \(|,|\. Se recomienda)", output):
        f["first_excipient"] = m.group(1)
    return f


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def generate(model, tok, rows, batch_size, max_new_tokens):
    outs = []
    tok.padding_side = "left"
    for i in range(0, len(rows), batch_size):
        prompts = [
            tok.apply_chat_template(build_messages(r, False), tokenize=False, add_generation_prompt=True)
            for r in rows[i : i + batch_size]
        ]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


def score(rows, outs):
    hits, totals = {}, {}
    for r, o in zip(rows, outs):
        for k, v in expected_fields(r["output"]).items():
            totals[k] = totals.get(k, 0) + 1
            hits[k] = hits.get(k, 0) + int(norm(v) in norm(o))
    res = {k: round(hits[k] / totals[k], 3) for k in totals}
    res["overall"] = round(sum(hits.values()) / sum(totals.values()), 3)
    res["n_fields"] = sum(totals.values())
    return res


def main(adapter: str, n: int, batch_size: int):
    rows = load_qa(Path(PROCESSED_DATA_DIR) / "medical_qa_heldout.jsonl")[:n]
    tok = AutoTokenizer.from_pretrained(LLM_BASE_MODEL)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(LLM_BASE_MODEL, quantization_config=bnb, device_map={"": 0},
                                                dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(Path(LLM_MODEL_DIR) / adapter)).eval()
    with model.disable_adapter():
        base_outs = generate(model, tok, rows, batch_size, 120)
    tuned_outs = generate(model, tok, rows, batch_size, 120)
    report = {
        "adapter": adapter,
        "n_heldout_rows": len(rows),
        "base_model": score(rows, base_outs),
        "fine_tuned": score(rows, tuned_outs),
        "examples": [{"question": r["instruction"], "reference": r["output"], "base": b, "tuned": t}
                     for r, b, t in list(zip(rows, base_outs, tuned_outs))[:5]],
    }
    out = Path(LLM_MODEL_DIR) / f"eval_{adapter}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(json.dumps({k: report[k] for k in ("n_heldout_rows", "base_model", "fine_tuned")}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="final_adapter")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    main(a.adapter, a.n, a.batch_size)
