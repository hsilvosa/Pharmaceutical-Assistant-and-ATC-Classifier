// La botica: ATC classifier running in the browser (Transformers.js, int8 ONNX) and receipts of real agent runs.
import { AutoTokenizer, AutoModelForSequenceClassification, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3";

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = new URL("./models/", location.href).href;
env.backends.onnx.wasm.proxy = true; // run inference in a worker so the page stays responsive

const $ = (id) => document.getElementById(id);
const esc = (v) => String(v).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const GROUPS = ["A", "B", "C", "D", "G", "H", "J", "L", "M", "N", "P", "R", "S", "V"];
const GN = {
  en: ["Alimentary tract and metabolism", "Blood", "Cardiovascular system", "Dermatologicals", "Genito-urinary system and sex hormones", "Systemic hormonal preparations", "Anti-infectives", "Antineoplastic and immunomodulating", "Musculo-skeletal system", "Nervous system", "Antiparasitic products", "Respiratory system", "Sensory organs", "Various"],
  es: ["Digestivo y metabolismo", "Sangre", "Cardiovascular", "Dermatológicos", "Genitourinario y hormonas sexuales", "Hormonales sistémicos", "Antiinfecciosos", "Antineoplásicos e inmuno", "Musculoesquelético", "Sistema nervioso", "Antiparasitarios", "Respiratorio", "Órganos de los sentidos", "Varios"],
};
const T = {
  en: {
    kicker: "pharma-assistant · AEMPS CIMA data", title: "The pharmacy cabinet",
    sub: "Every medicine has its drawer. Describe one and see where the classifier would file it (it runs in your browser); next to it, the receipt the agent prints when you ask it a question.",
    s1: "1 · Put the medicine on the tray", name: "Medicine name", ing: "Active ingredient", form: "Pharmaceutical form",
    go: "Classify", loading: "Loading the model…", ready: "Classify", failed: "The model could not be loaded",
    spanish: "The classifier reads Spanish CIMA drug descriptions, so use the name and ingredient as sold in Spain. The more you give it, the better: on unseen ingredients top-1 accuracy is {a1} with the name only, {a2} with the ingredient only, {a3} with both and {a4} with the full record.",
    s2: "2 · The cabinet decides", empty: "Describe a medicine and press Classify.", need: "Enter at least a medicine name or an active ingredient.",
    drawer: "Drawer", occ: "Underlined: what weighed most (word occlusion on the calibrated probability).",
    near: "Closest CIMA medicines (text similarity) and their real ATC code",
    s3: "3 · The agent's receipt", q: "QUESTION", total: "TOTAL", answer: "SHOWN TO THE USER", sources: "SOURCES",
    stamps: { answered: "ANSWERED", refused: "REFUSED", insufficient_evidence: "NO EVIDENCE", needs_disambiguation: "ASKS TO CLARIFY", timeout: "TIMED OUT" },
    model: "model decision", none: "no tool call",
    note1: "Only the anatomical group (ATC level 1, 14 groups) is predicted, not a full ATC code. It is a hint from a text classifier, not medical advice. Accuracy on active ingredients the model has not seen: {acc} top-1 (13 of 14 groups present in the test set). Probabilities are temperature-scaled (T={t}): expected calibration error went from {e0} to {e1}. The model here is the int8-quantised export (110 MB): on 600 held-out rows it agreed with the full model on 98.2% of predictions.",
    note2: "Agent evaluation: 33 of 36 benchmark cases passed (development result, rules tuned on those cases). Every receipt is a real run on a local Qwen3-8B recorded on {date}; citation text is verbatim from AEMPS documents and always in Spanish. This is information from official leaflets, never personal medical advice.",
    footer: "All numbers come from the repository's evaluation files.",
  },
  es: {
    kicker: "pharma-assistant · datos AEMPS CIMA", title: "La botica",
    sub: "Cada medicamento tiene su cajón. Describe uno y mira dónde lo guardaría el clasificador (se ejecuta en tu navegador); al lado, el ticket que imprime el agente cuando le haces una pregunta.",
    s1: "1 · Deja el medicamento en la bandeja", name: "Nombre del medicamento", ing: "Principio activo", form: "Forma farmacéutica",
    go: "Clasificar", loading: "Cargando el modelo…", ready: "Clasificar", failed: "No se pudo cargar el modelo",
    spanish: "El clasificador lee descripciones CIMA en español: usa el nombre y el principio activo como se venden en España. Cuanto más le des, mejor: con principios activos no vistos la precisión top-1 es {a1} solo con el nombre, {a2} solo con el principio activo, {a3} con ambos y {a4} con el registro completo.",
    s2: "2 · La botica decide", empty: "Describe un medicamento y pulsa Clasificar.", need: "Introduce al menos un nombre de medicamento o un principio activo.",
    drawer: "Cajón", occ: "Lo subrayado es lo que más pesó (oclusión de palabras sobre la probabilidad calibrada).",
    near: "Medicamentos CIMA más parecidos (similitud de texto) y su código ATC real",
    s3: "3 · El ticket del agente", q: "PREGUNTA", total: "TOTAL", answer: "LO QUE VE EL USUARIO", sources: "FUENTES",
    stamps: { answered: "RESPONDIDA", refused: "RECHAZADA", insufficient_evidence: "SIN EVIDENCIA", needs_disambiguation: "PIDE ACLARAR", timeout: "TIEMPO AGOTADO" },
    model: "decisión del modelo", none: "sin herramienta",
    note1: "Solo se predice el grupo anatómico (ATC nivel 1, 14 grupos), no un código ATC completo. Es una pista de un clasificador de texto, no consejo médico. Precisión en principios activos que el modelo no ha visto: {acc} top-1 (13 de 14 grupos presentes en el test). Las probabilidades están escaladas por temperatura (T={t}): el error de calibración esperado pasó de {e0} a {e1}. El modelo aquí es la exportación cuantizada a int8 (110 MB): en 600 filas retenidas coincidió con el modelo completo en el 98,2 % de las predicciones.",
    note2: "Evaluación del agente: 33 de 36 casos del banco superados (resultado de desarrollo, reglas ajustadas con esos casos). Cada ticket es una ejecución real con un Qwen3-8B local grabada el {date}; las citas son literales de documentos de la AEMPS y siempre en español. Es información de prospectos oficiales, nunca consejo médico personal.",
    footer: "Todas las cifras salen de los ficheros de evaluación del repositorio.",
  },
};
const EXAMPLES = [
  ["Sertralina 50 mg comprimidos", "SERTRALINA", "COMPRIMIDO RECUBIERTO CON PELÍCULA"],
  ["Ventolin 100 microgramos", "SALBUTAMOL", "SUSPENSIÓN PARA INHALACIÓN EN ENVASE A PRESIÓN"],
  ["Atorvastatina 20 mg", "ATORVASTATINA", "COMPRIMIDO RECUBIERTO CON PELÍCULA"],
];

let lang = localStorage.getItem("lang") === "es" ? "es" : "en";
let tokenizer = null, model = null, temperature = 1, data = {}, lastResult = null, currentCase = null, neighbors = null;
const fmtNum = (x) => (lang === "es" ? String(x).replace(".", ",") : String(x));
const pct = (key) => fmtNum((data.ablation[key].top1 * 100).toFixed(0)) + (lang === "es" ? " %" : "%");

function compose(name, ing, form) {
  const parts = [];
  if (name.trim()) parts.push(`Medicamento: ${name.trim()}`);
  if (form.trim()) parts.push(`Forma farmacéutica: ${form.trim()}`);
  if (ing.trim()) parts.push(`Principios activos: ${ing.trim()}`);
  return parts.join(" | ");
}

async function probabilities(texts) {
  const inputs = tokenizer(texts, { padding: true, truncation: true, max_length: 256 });
  const { logits } = await model(inputs);
  const k = logits.dims[1], out = [];
  for (let i = 0; i < logits.dims[0]; i++) {
    const row = Array.from(logits.data.slice(i * k, (i + 1) * k), (v) => v / temperature);
    const peak = Math.max(...row), exps = row.map((v) => Math.exp(v - peak)), sum = exps.reduce((a, b) => a + b, 0);
    out.push(exps.map((v) => v / sum));
  }
  return out;
}

async function occlusion(fields, topIndex, base) {
  const keys = ["name", "ing", "form"], variants = [], owners = [];
  keys.forEach((key) => fields[key].split(/\s+/).filter(Boolean).forEach((_, i) => {
    const reduced = {}; keys.forEach((k) => (reduced[k] = fields[k].split(/\s+/).filter(Boolean)));
    reduced[key].splice(i, 1);
    variants.push(compose(reduced.name.join(" "), reduced.ing.join(" "), reduced.form.join(" ")) || "Medicamento:");
    owners.push([key, i]);
  }));
  if (!variants.length) return [];
  const probs = await probabilities(variants);
  const drops = probs.map((p) => base - p[topIndex]), peak = Math.max(...drops, 1e-9);
  const weight = {};
  owners.forEach(([key, i], n) => { const r = Math.max(drops[n], 0) / peak; weight[key + i] = r >= 0.66 ? 3 : r >= 0.33 ? 2 : r >= 0.1 ? 1 : 0; });
  return keys.flatMap((key) => fields[key].split(/\s+/).filter(Boolean).map((w, i) => [w, weight[key + i]]));
}

function grams(text) {
  const t = ` ${text.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "")} `, set = new Set();
  for (let i = 0; i < t.length - 2; i++) set.add(t.slice(i, i + 3));
  return set;
}
function nearest(name, ing, k = 4) {
  if (!neighbors) neighbors = data.neighbors.map((r) => ({ r, g: grams(`${r[0]} ${r[2]}`) }));
  const q = grams(`${name} ${ing}`.trim());
  return neighbors.map(({ r, g }) => {
    let inter = 0; q.forEach((x) => { if (g.has(x)) inter++; });
    return [inter / (q.size + g.size - inter || 1), r];
  }).sort((a, b) => b[0] - a[0]).slice(0, k).filter((x) => x[0] > 0).map((x) => [x[1][0], x[1][1]]);
}

function cabinet(preds) {
  const keys = preds ? Object.keys(preds) : [];
  $("cab").className = "cab" + (keys.length ? " has" : "");
  $("cab").innerHTML = GROUPS.map((g, i) => {
    const hit = keys.includes(g), top = keys[0] === g;
    return `<div class="drawer${hit ? " hit" : ""}${top ? " top" : ""}" role="listitem" aria-label="${esc(T[lang].drawer)} ${g}: ${esc(GN[lang][i])}">${hit ? `<span class="pc">${Math.round(preds[g] * 100)} %</span>` : ""}<span class="lab">${g}</span><span class="nm">${esc(GN[lang][i])}</span><span class="knob"></span></div>`;
  }).join("");
}

async function classify() {
  $("go").disabled = true; $("go").textContent = "…";
  try { await classifyNow(); } finally { $("go").disabled = false; $("go").textContent = T[lang].ready; }
}

async function classifyNow() {
  const name = $("name").value, ing = $("ing").value, form = $("form").value, t = T[lang];
  if (!name.trim() && !ing.trim()) { cabinet(null); $("inside").innerHTML = `<span class="hint">${esc(t.need)}</span>`; lastResult = null; return; }
  const probs = (await probabilities([compose(name, ing, form)]))[0];
  const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
  const letter = model.config.id2label[order[0][1]];
  const preds = {};
  order.forEach(([p, i], n) => { if (n === 0 || p >= 0.05) preds[model.config.id2label[i]] = p; });
  const plain = ["name", "ing", "form"].flatMap((k) => ({ name, ing, form })[k].split(/\s+/).filter(Boolean).map((w) => [w, 0]));
  lastResult = { preds, letter, words: plain, near: nearest(name, ing) };
  renderInside(); cabinet(preds);
  // The highlights need one extra pass per word: show the result first, refine afterwards.
  const mine = lastResult;
  const words = await occlusion({ name, ing, form }, order[0][1], order[0][0]);
  if (lastResult === mine) { mine.words = words; renderInside(); }
}

function renderInside() {
  const t = T[lang];
  if (!lastResult) { $("inside").innerHTML = `<span class="hint">${esc(t.empty)}</span>`; return; }
  const { letter, words, near } = lastResult, idx = GROUPS.indexOf(letter);
  const marked = words.map(([w, h]) => (h ? `<mark class="h${h}">${esc(w)}</mark>` : esc(w))).join(" ");
  $("inside").innerHTML = `<h3>${esc(t.drawer)} ${esc(letter)} · ${esc(idx >= 0 ? GN[lang][idx] : letter)}</h3><div class="occ">${marked}</div><p class="hint">${esc(t.occ)}</p><p class="hint"><b>${esc(t.near)}</b></p><table>${near.map(([n, c]) => `<tr><td>${esc(n)}</td><td class="c">${esc(c)}</td></tr>`).join("")}</table>`;
}

function stampFor(e) {
  if (e.termination === "timeout") return [T[lang].stamps.timeout, false];
  const s = T[lang].stamps[e.status];
  return [s || e.status.toUpperCase(), Boolean(s)];
}
const sec = (ms) => fmtNum((ms / 1000).toFixed(1));

function receipt(id) {
  const t = T[lang], e = data.gallery.entries.find((x) => x.id === id);
  const lines = [`<span class="q">${esc(t.q)}</span>`, esc(e.question), "<hr>"];
  e.steps.forEach((s) => {
    if (s.kind === "model") lines.push(`${esc(t.model)} → ${esc(s.name === "no tool call" ? t.none : s.name)} (${sec(s.ms)} s)`);
    else if (s.kind === "tool") lines.push(`${esc(s.name)} (${sec(s.ms)} s) · ${esc(s.status || "")}`);
    else lines.push(esc(s.name));
  });
  lines.push(`<span class="dim">${esc(e.termination.replace(/_/g, " "))}</span>`, "<hr>", `${esc(t.total)} ${sec(e.total_ms)} s`, `<span class="q">${esc(t.answer)}</span>`, esc(e.answer));
  if (e.citations.length) {
    lines.push(`<span class="q">${esc(t.sources)}</span>`);
    e.citations.slice(0, 2).forEach((c) => {
      const quote = c.quote.length > 320 ? c.quote.slice(0, 320) + "…" : c.quote;
      lines.push(`<span class="cite">[${esc(c.id)}] ${esc(c.medicine)}, §${esc(c.section)}<br>“${esc(quote)}”</span>`);
    });
  }
  if (e.note) lines.push(`<span class="dim">${esc(e.note)}</span>`);
  const [stamp, ok] = stampFor(e);
  $("slip").innerHTML = lines.map((l, i) => `<span class="ln" style="--i:${i}">${l}</span>`).join("") + `<span class="stamp${ok ? "" : " bad"}" style="--i:${lines.length}">${esc(stamp)}</span>`;
}

function renderCases() {
  $("cases").innerHTML = data.gallery.entries.map((e) => {
    const ok = e.status in T[lang].stamps;
    return `<button class="case" aria-pressed="${e.id === currentCase}" data-id="${esc(e.id)}"><i class="${ok ? "" : "bad"}">${ok ? "OK" : "!"}</i><span>${esc(e.question.length > 90 ? e.question.slice(0, 90) + "…" : e.question)}</span></button>`;
  }).join("");
}

function applyLang() {
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-t]").forEach((n) => { const v = T[lang][n.dataset.t]; if (v) n.textContent = v; });
  document.querySelectorAll(".lang button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.l === lang));
  if (!model) $("go").textContent = T[lang].loading; else $("go").textContent = T[lang].ready;
  $("spanish").textContent = T[lang].spanish.replace("{a1}", pct("name only")).replace("{a2}", pct("ingredient only")).replace("{a3}", pct("name + ingredient")).replace("{a4}", pct("full training format (all fields)"));
  const c = data.calibration;
  $("note1").textContent = T[lang].note1.replace("{acc}", fmtNum("79.3") + (lang === "es" ? " %" : "%")).replace("{t}", fmtNum(c.temperature)).replace("{e0}", fmtNum(c.test_before.ece)).replace("{e1}", fmtNum(c.test_after.ece));
  $("note2").textContent = T[lang].note2.replace("{date}", data.gallery.meta.generated);
  $("chips").innerHTML = EXAMPLES.map((e, i) => `<button type="button" class="chip" data-i="${i}">${esc(e[1].toLowerCase())}</button>`).join("");
  document.title = `${T[lang].title} · pharma-assistant`;
  cabinet(lastResult ? lastResult.preds : null); renderInside(); renderCases(); receipt(currentCase);
}

async function main() {
  const load = (f) => fetch(f).then((r) => r.json());
  const [gallery, calibration, ablation, nb, forms] = await Promise.all([load("data/gallery.json"), load("data/calibration.json"), load("data/atc_input_ablation.json"), load("data/neighbors.json"), load("data/forms.json")]);
  data = { gallery, calibration, ablation, neighbors: nb };
  temperature = calibration.temperature;
  currentCase = gallery.entries[0].id;
  $("forms").innerHTML = forms.map((f) => `<option value="${esc(f)}">`).join("");
  applyLang();
  $("cases").onclick = (e) => { const b = e.target.closest(".case"); if (b) { currentCase = b.dataset.id; renderCases(); receipt(currentCase); } };
  $("chips").onclick = (e) => { const b = e.target.closest(".chip"); if (!b) return; const ex = EXAMPLES[+b.dataset.i]; $("name").value = ex[0]; $("ing").value = ex[1]; $("form").value = ex[2]; classify(); };
  document.querySelectorAll(".lang button").forEach((b) => (b.onclick = () => { lang = b.dataset.l; localStorage.setItem("lang", lang); applyLang(); }));
  $("f").onsubmit = (e) => { e.preventDefault(); if (model) classify(); };
  try {
    tokenizer = await AutoTokenizer.from_pretrained("atc");
    model = await AutoModelForSequenceClassification.from_pretrained("atc", { dtype: "q8" });
    $("go").disabled = false; $("go").textContent = T[lang].ready;
    await classify();
  } catch (err) {
    console.error(err);
    $("go").textContent = T[lang].failed;
  }
}
main();
