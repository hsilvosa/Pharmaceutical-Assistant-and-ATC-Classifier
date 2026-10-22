"""Public demo: ATC level-1 classifier (live) and agent traces (real runs, static).

Run locally: python space/app.py   (set ATC_MODEL to a local folder to avoid the Hub download)
"""
from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
MODEL_ID = os.environ.get("ATC_MODEL", "hsilvosa/BETO-ATC-Classifier")
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

TEMPERATURE = float(json.loads((HERE / "calibration.json").read_text(encoding="utf-8"))["temperature"])
GALLERY = json.loads((HERE / "gallery.json").read_text(encoding="utf-8"))
FORMS = json.loads((HERE / "data" / "forms.json").read_text(encoding="utf-8"))
NEIGHBORS = pd.read_csv(HERE / "data" / "cima_neighbors.csv.gz").fillna("")
_docs = (NEIGHBORS["name"] + " " + NEIGHBORS["ingredients"]).tolist()
VECTORIZER = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
MATRIX = VECTORIZER.fit_transform(_docs)

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
ID2LABEL = {int(k): v for k, v in MODEL.config.id2label.items()}

GROUPS = ["A", "B", "C", "D", "G", "H", "J", "L", "M", "N", "P", "R", "S", "V"]
GROUP_NAMES = {
    "en": ["Alimentary tract and metabolism", "Blood", "Cardiovascular system", "Dermatologicals",
           "Genito-urinary system and sex hormones", "Systemic hormonal preparations", "Anti-infectives",
           "Antineoplastic and immunomodulating", "Musculo-skeletal system", "Nervous system",
           "Antiparasitic products", "Respiratory system", "Sensory organs", "Various"],
    "es": ["Digestivo y metabolismo", "Sangre", "Cardiovascular", "Dermatológicos",
           "Genitourinario y hormonas sexuales", "Hormonales sistémicos", "Antiinfecciosos",
           "Antineoplásicos e inmuno", "Musculoesquelético", "Sistema nervioso",
           "Antiparasitarios", "Respiratorio", "Órganos de los sentidos", "Varios"],
}

T = {
    "en": {
        "kicker": "pharma-assistant · AEMPS CIMA data", "title": "The pharmacy cabinet",
        "sub": "Every medicine has its drawer. Describe one and see where the classifier would file it; next to it, the receipt the agent prints when you ask it a question.",
        "s1": "1 · Put the medicine on the tray", "name": "Medicine name", "ing": "Active ingredient", "form": "Pharmaceutical form",
        "go": "Classify", "spanish": "The classifier reads Spanish CIMA drug descriptions, so use the name and ingredient as sold in Spain. The more you give it, the better: on unseen ingredients top-1 accuracy is {a1} with the name only, {a2} with the ingredient only, {a3} with both and {a4} with the full record.",
        "s2": "2 · The cabinet decides", "empty": "Describe a medicine and press Classify.",
        "drawer": "Drawer", "occ": "Underlined: what weighed most (word occlusion on the calibrated probability).",
        "near": "Closest CIMA medicines (text similarity) and their real ATC code",
        "s3": "3 · The agent's receipt", "case": "Real run", "q": "QUESTION", "total": "TOTAL",
        "answer": "SHOWN TO THE USER", "sources": "SOURCES", "note_case": "Questions are shown as asked; quotes are always the original Spanish text.",
        "stamps": {"answered": "ANSWERED", "refused": "REFUSED", "insufficient_evidence": "NO EVIDENCE", "needs_disambiguation": "ASKS TO CLARIFY", "timeout": "TIMED OUT"},
        "need": "Enter at least a medicine name or an active ingredient.",
        "note1": "Only the anatomical group (ATC level 1, 14 groups) is predicted, not a full ATC code. It is a hint from a text classifier, not medical advice. Accuracy on active ingredients the model has not seen: {acc}% top-1 (13 of 14 groups present in the test set). Probabilities are temperature-scaled (T={t}): expected calibration error went from {e0} to {e1} on the test split.",
        "note2": "Agent evaluation: 33 of 36 benchmark cases passed (development result, rules tuned on those cases). Every receipt here is a real run on a local Qwen3-8B; citation text is verbatim from AEMPS documents. This is information from official leaflets, never personal medical advice.",
        "model": "model decision", "tool": "tool", "none": "no tool call",
    },
    "es": {
        "kicker": "pharma-assistant · datos AEMPS CIMA", "title": "La botica",
        "sub": "Cada medicamento tiene su cajón. Describe uno y mira dónde lo guardaría el clasificador; al lado, el ticket que imprime el agente cuando le haces una pregunta.",
        "s1": "1 · Deja el medicamento en la bandeja", "name": "Nombre del medicamento", "ing": "Principio activo", "form": "Forma farmacéutica",
        "go": "Clasificar", "spanish": "El clasificador lee descripciones CIMA en español: usa el nombre y el principio activo como se venden en España. Cuanto más le des, mejor: con principios activos no vistos la precisión top-1 es {a1} solo con el nombre, {a2} solo con el principio activo, {a3} con ambos y {a4} con el registro completo.",
        "s2": "2 · La botica decide", "empty": "Describe un medicamento y pulsa Clasificar.",
        "drawer": "Cajón", "occ": "Lo subrayado es lo que más pesó (oclusión de palabras sobre la probabilidad calibrada).",
        "near": "Medicamentos CIMA más parecidos (similitud de texto) y su código ATC real",
        "s3": "3 · El ticket del agente", "case": "Ejecución real", "q": "PREGUNTA", "total": "TOTAL",
        "answer": "LO QUE VE EL USUARIO", "sources": "FUENTES", "note_case": "Las preguntas se muestran tal como se hicieron; las citas son siempre el texto original en español.",
        "stamps": {"answered": "RESPONDIDA", "refused": "RECHAZADA", "insufficient_evidence": "SIN EVIDENCIA", "needs_disambiguation": "PIDE ACLARAR", "timeout": "TIEMPO AGOTADO"},
        "need": "Introduce al menos un nombre de medicamento o un principio activo.",
        "note1": "Solo se predice el grupo anatómico (ATC nivel 1, 14 grupos), no un código ATC completo. Es una pista de un clasificador de texto, no consejo médico. Precisión en principios activos que el modelo no ha visto: {acc} % top-1 (13 de 14 grupos presentes en el test). Las probabilidades están escaladas por temperatura (T={t}): el error de calibración esperado pasó de {e0} a {e1} en el test.",
        "note2": "Evaluación del agente: 33 de 36 casos del banco superados (resultado de desarrollo, reglas ajustadas con esos casos). Cada ticket es una ejecución real con un Qwen3-8B local; las citas son literales de documentos de la AEMPS. Es información de prospectos oficiales, nunca consejo médico personal.",
        "model": "decisión del modelo", "tool": "herramienta", "none": "sin herramienta",
    },
}
CAL = json.loads((HERE / "calibration.json").read_text(encoding="utf-8"))
ABL = json.loads((HERE / "data" / "atc_input_ablation.json").read_text(encoding="utf-8"))


def pct(key: str, lang: str) -> str:
    text = f"{ABL[key]['top1'] * 100:.0f}"
    return text + (" %" if lang == "es" else "%")


def esc(value: object) -> str:
    return html.escape(str(value))


def fmt_s(ms: float, lang: str) -> str:
    text = f"{ms / 1000:.1f}"
    return text.replace(".", ",") if lang == "es" else text


def compose(name: str, ing: str, form: str) -> str:
    parts = []
    if name.strip():
        parts.append(f"Medicamento: {name.strip()}")
    if form.strip():
        parts.append(f"Forma farmacéutica: {form.strip()}")
    if ing.strip():
        parts.append(f"Principios activos: {ing.strip()}")
    return " | ".join(parts)


@torch.no_grad()
def probabilities(texts: list[str]) -> np.ndarray:
    enc = TOKENIZER(texts, truncation=True, max_length=256, padding=True, return_tensors="pt")
    logits = MODEL(**enc).logits.float() / TEMPERATURE
    return torch.softmax(logits, dim=-1).numpy()


def occlusion(name: str, ing: str, form: str, top_index: int, base: float) -> list[tuple[str, int]]:
    """Weight of each word of the three fields: drop in the top class probability when it is removed."""
    fields = {"name": name.split(), "ing": ing.split(), "form": form.split()}
    variants, owners = [], []
    for key, words in fields.items():
        for i in range(len(words)):
            reduced = {k: list(v) for k, v in fields.items()}
            del reduced[key][i]
            variants.append(compose(" ".join(reduced["name"]), " ".join(reduced["ing"]), " ".join(reduced["form"])) or "Medicamento:")
            owners.append((key, i))
    if not variants:
        return []
    drops = base - probabilities(variants)[:, top_index]
    peak = max(float(drops.max()), 1e-9)
    weights = {}
    for (key, i), drop in zip(owners, drops):
        ratio = max(float(drop), 0.0) / peak
        weights[(key, i)] = 3 if ratio >= 0.66 else 2 if ratio >= 0.33 else 1 if ratio >= 0.1 else 0
    return [(w, weights[(key, i)]) for key, words in fields.items() for i, w in enumerate(words)]


def nearest(name: str, ing: str, k: int = 4) -> list[tuple[str, str]]:
    query = VECTORIZER.transform([f"{name} {ing}".strip()])
    scores = (MATRIX @ query.T).toarray().ravel()
    best = np.argsort(-scores)[:k]
    return [(NEIGHBORS.iloc[i]["name"], NEIGHBORS.iloc[i]["atc_code"]) for i in best if scores[i] > 0]


def cabinet_html(lang: str, preds: dict[str, float] | None) -> str:
    keys = list(preds) if preds else []
    cells = []
    for letter, label in zip(GROUPS, GROUP_NAMES[lang]):
        cls = "drawer"
        pc = ""
        if letter in keys:
            cls += " hit" + (" top" if letter == keys[0] else "")
            pc = f'<span class="pc">{preds[letter] * 100:.0f} %</span>'
        cells.append(
            f'<div class="{cls}" aria-label="{esc(T[lang]["drawer"])} {letter}: {esc(label)}">{pc}'
            f'<span class="lab">{letter}</span><span class="nm">{esc(label)}</span><span class="knob"></span></div>'
        )
    return f'<div class="cab{" has" if keys else ""}">{"".join(cells)}</div>'


def classify(name: str, ing: str, form: str, lang: str):
    t = T[lang]
    if not name.strip() and not ing.strip():
        return cabinet_html(lang, None), f'<div class="inside"><span class="hint">{esc(t["need"])}</span></div>'
    text = compose(name, ing, form)
    probs = probabilities([text])[0]
    order = np.argsort(-probs)[:3]
    preds = {ID2LABEL[int(i)]: float(probs[i]) for n, i in enumerate(order) if n == 0 or probs[i] >= 0.05}
    top = int(order[0])
    letter = ID2LABEL[top]
    weights = occlusion(name, ing, form, top, float(probs[top]))
    words = " ".join(f'<mark class="h{w}">{esc(word)}</mark>' if w else esc(word) for word, w in weights)
    rows = "".join(f"<tr><td>{esc(n)}</td><td class='c'>{esc(c)}</td></tr>" for n, c in nearest(name, ing))
    title = f'{t["drawer"]} {letter} · {GROUP_NAMES[lang][GROUPS.index(letter)] if letter in GROUPS else letter}'
    inside = (
        f'<div class="inside"><h3>{esc(title)}</h3><div class="occ">{words}</div>'
        f'<p class="hint">{esc(t["occ"])}</p><p class="hint"><b>{esc(t["near"])}</b></p><table>{rows}</table></div>'
    )
    return cabinet_html(lang, preds), inside


def stamp_for(entry: dict, lang: str) -> tuple[str, bool]:
    stamps = T[lang]["stamps"]
    if entry["termination"] == "timeout":
        return stamps["timeout"], False
    ok = entry["status"] in stamps
    return stamps.get(entry["status"], entry["status"].upper()), ok


def receipt_html(case_id: str, lang: str) -> str:
    t = T[lang]
    entry = next(e for e in GALLERY["entries"] if e["id"] == case_id)
    lines: list[str] = [f'<span class="q">{esc(t["q"])}</span>', esc(entry["question"]), "<hr>"]
    for step in entry["steps"]:
        label = t["model"] if step["kind"] == "model" else step["name"]
        if step["kind"] == "model":
            target = step["name"] if step["name"] != "no tool call" else t["none"]
            lines.append(f'{esc(label)} → {esc(target)} ({esc(fmt_s(step["ms"], lang))} s)')
        elif step["kind"] == "tool":
            lines.append(f'{esc(step["name"])} ({esc(fmt_s(step["ms"], lang))} s) · {esc(step.get("status", ""))}')
        else:
            lines.append(esc(step["name"]))
    if not entry["steps"]:
        lines.append(esc(entry["termination"].replace("_", " ")))
    else:
        lines.append(f'<span class="dim">{esc(entry["termination"].replace("_", " "))}</span>')
    lines += ["<hr>", f'{esc(t["total"])} {esc(fmt_s(entry["total_ms"], lang))} s', f'<span class="q">{esc(t["answer"])}</span>']
    lines.append(esc(entry["answer"]))
    if entry["citations"]:
        lines.append(f'<span class="q">{esc(t["sources"])}</span>')
        for c in entry["citations"][:2]:
            quote = c["quote"][:320] + ("…" if len(c["quote"]) > 320 else "")
            lines.append(f'<span class="cite">[{esc(c["id"])}] {esc(c["medicine"])}, §{esc(c["section"])}<br>“{esc(quote)}”</span>')
    if entry.get("note"):
        lines.append(f'<span class="dim">{esc(entry["note"])}</span>')
    stamp, ok = stamp_for(entry, lang)
    body = "".join(f'<span class="ln" style="--i:{i}">{line}</span>' for i, line in enumerate(lines))
    return f'<div class="printer"></div><div class="slip">{body}<span class="stamp{"" if ok else " bad"}" style="--i:{len(lines)}">{esc(stamp)}</span></div>'


def case_choices(lang: str) -> list[tuple[str, str]]:
    out = []
    for e in GALLERY["entries"]:
        stamp, ok = stamp_for(e, lang)
        out.append((f'{"✓" if ok and e["termination"] != "timeout" else "✗"} {e["question"][:70]}', e["id"]))
    return out


def notes(lang: str) -> tuple[str, str]:
    t = T[lang]
    acc = "79,3" if lang == "es" else "79.3"
    n1 = t["note1"].format(acc=acc, t=CAL["temperature"], e0=CAL["test_before"]["ece"], e1=CAL["test_after"]["ece"])
    return f'<p class="note">{esc(n1)}</p>', f'<p class="note">{esc(t["note2"])} {esc(t["note_case"])}</p>'


CSS = (HERE / "style.css").read_text(encoding="utf-8")
DEFAULT_CASE = GALLERY["entries"][0]["id"]
EXAMPLES = [("Sertralina 50 mg comprimidos", "SERTRALINA", "COMPRIMIDO RECUBIERTO CON PELÍCULA"),
            ("Ventolin 100 microgramos", "SALBUTAMOL", "SUSPENSIÓN PARA INHALACIÓN EN ENVASE A PRESIÓN"),
            ("Atorvastatina 20 mg", "ATORVASTATINA", "COMPRIMIDO RECUBIERTO CON PELÍCULA")]


def static_blocks(lang: str):
    t = T[lang]
    head = (f'<header><div class="cross"></div><h1><small>{esc(t["kicker"])}</small>{esc(t["title"])}</h1>'
            f'<p class="sub">{esc(t["sub"])}</p></header>')
    n1, n2 = notes(lang)
    return head, f'<div class="panel-t">{esc(t["s1"])}</div>', f'<p class="hint">{esc(t["spanish"].format(a1=pct("name only", lang), a2=pct("ingredient only", lang), a3=pct("name + ingredient", lang), a4=pct("full training format (all fields)", lang)))}</p>', f'<div class="panel-t">{esc(t["s2"])}</div>', f'<div class="panel-t">{esc(t["s3"])}</div>', n1, n2


with gr.Blocks(css=CSS, title="La botica · pharma-assistant", theme=gr.themes.Base()) as demo:
    with gr.Row(elem_id="top"):
        head = gr.HTML()
        lang = gr.Radio([("EN", "en"), ("ES", "es")], value="en", label="", show_label=False, container=False, elem_id="lang", scale=0, min_width=110)
    with gr.Row(elem_id="cols"):
        with gr.Column(scale=3):
            s1 = gr.HTML()
            name = gr.Textbox(label=T["en"]["name"], value=EXAMPLES[0][0])
            ing = gr.Textbox(label=T["en"]["ing"], value=EXAMPLES[0][1])
            form = gr.Dropdown(FORMS, value=EXAMPLES[0][2], label=T["en"]["form"], allow_custom_value=True)
            go = gr.Button(T["en"]["go"], variant="primary", elem_id="go")
            with gr.Row(elem_id="chips"):
                chips = [gr.Button(e[1].lower(), size="sm", elem_classes="chip") for e in EXAMPLES]
            spanish = gr.HTML()
            s2 = gr.HTML()
            cab = gr.HTML()
            inside = gr.HTML()
            note1 = gr.HTML()
        with gr.Column(scale=2):
            s3 = gr.HTML()
            case = gr.Radio(case_choices("en"), value=DEFAULT_CASE, label=T["en"]["case"], elem_id="cases")
            slip = gr.HTML()
            note2 = gr.HTML()

    def refresh(lang_value, name_v, ing_v, form_v, case_v):
        t = T[lang_value]
        h, a, b, c, d, n1, n2 = static_blocks(lang_value)
        cab_html, inside_html = classify(name_v, ing_v, form_v, lang_value)
        return (h, a, b, c, d, n1, n2, cab_html, inside_html, receipt_html(case_v, lang_value),
                gr.update(label=t["name"]), gr.update(label=t["ing"]), gr.update(label=t["form"]), gr.update(value=t["go"]),
                gr.update(choices=case_choices(lang_value), value=case_v, label=t["case"]))

    outputs = [head, s1, spanish, s2, s3, note1, note2, cab, inside, slip, name, ing, form, go, case]
    inputs = [lang, name, ing, form, case]
    lang.change(refresh, inputs, outputs, show_progress="hidden")
    demo.load(refresh, inputs, outputs, show_progress="hidden")
    go.click(lambda n, i, f, l: classify(n, i, f, l), [name, ing, form, lang], [cab, inside])
    case.change(lambda c, l: receipt_html(c, l), [case, lang], slip, show_progress="hidden")
    for chip, example in zip(chips, EXAMPLES):
        chip.click(lambda l, e=example: (*e, *classify(e[0], e[1], e[2], l)), [lang], [name, ing, form, cab, inside])

if __name__ == "__main__":
    demo.launch()
