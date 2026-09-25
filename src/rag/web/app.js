const state = { registrationNumber: null, question: "" };
const qs = (selector) => document.querySelector(selector);

function setHidden(selector, hidden) { qs(selector).hidden = hidden; }
function escapeHtml(value) {
  const div = document.createElement("div"); div.textContent = value ?? ""; return div.innerHTML;
}

async function checkHealth() {
  const health = qs("#health");
  try {
    const data = await fetch("/api/v1/health").then((response) => response.json());
    health.className = `health ${data.ready ? "ready" : "degraded"}`;
    health.innerHTML = `<span></span>${data.ready ? "Listo" : "Sin índice"}`;
  } catch { health.className = "health degraded"; health.innerHTML = "<span></span>No disponible"; }
}

document.querySelectorAll(".tab").forEach((button) => button.addEventListener("click", async () => {
  document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
  button.classList.add("active");
  const isSearch = button.dataset.tab === "search";
  setHidden("#search-view", !isSearch); setHidden("#evaluation-view", isSearch);
  if (!isSearch) await loadEvaluation();
}));

qs("#query-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  state.question = qs("#question").value.trim();
  await runQuery();
});

async function runQuery() {
  setHidden("#empty-state", true); setHidden("#result", true); setHidden("#disambiguation", true);
  setHidden("#loading", false); qs("#submit").disabled = true;
  try {
    const response = await fetch("/api/v1/query", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: state.question, registration_number: state.registrationNumber, language: "auto" })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "No se pudo completar la consulta");
    if (data.status === "needs_disambiguation") renderCandidates(data.candidates);
    else renderResult(data);
  } catch (error) {
    renderResult({ status: "error", answer: error.message, citations: [], latency_ms: {} });
  } finally { setHidden("#loading", true); qs("#submit").disabled = false; }
}

function renderCandidates(candidates) {
  setHidden("#disambiguation", false);
  qs("#candidates").innerHTML = candidates.map((item, index) => `
    <button class="candidate" data-index="${index}"><strong>${escapeHtml(item.name)}</strong>
    <span>${escapeHtml(item.presentation || "Catálogo general")}</span>
    <span>Registro ${escapeHtml(item.registration_number)}${item.national_code ? ` · CN ${escapeHtml(item.national_code)}` : ""}</span></button>`).join("");
  qs("#candidates").querySelectorAll("button").forEach((button) => button.addEventListener("click", async () => {
    const item = candidates[Number(button.dataset.index)]; state.registrationNumber = item.registration_number;
    qs("#medicine-filter").hidden = false;
    qs("#medicine-filter").textContent = `Filtro: ${item.name} · registro ${item.registration_number}`;
    await runQuery();
  }));
}

function renderResult(data) {
  setHidden("#result", false);
  qs("#status-label").textContent = data.status.replaceAll("_", " ");
  qs("#latency").textContent = data.latency_ms?.total ? `${Math.round(data.latency_ms.total)} ms` : "";
  qs("#answer").textContent = data.answer;
  const first = data.citations?.[0];
  if (first) {
    setHidden("#medicine-context", false);
    qs("#medicine-context").innerHTML = `<div><strong>${escapeHtml(first.medicine_name)}</strong><br><span>Registro ${escapeHtml(first.registration_number)}</span></div>`;
  } else setHidden("#medicine-context", true);
  qs("#sources").innerHTML = (data.citations || []).map((source) => `
    <details class="source"><summary>[${escapeHtml(source.citation_id)}] Sección ${escapeHtml(source.section)} · ${escapeHtml(source.title)}</summary>
    <blockquote>${escapeHtml(source.quote)}</blockquote>
    ${source.url ? `<a href="${escapeHtml(source.url)}" target="_blank" rel="noreferrer">Abrir documento oficial</a>` : ""}</details>`).join("") || "<p>No hay fuentes verificadas para esta respuesta.</p>";
}

async function loadEvaluation() {
  const data = await fetch("/api/v1/evaluations/latest").then((response) => response.json());
  qs("#eval-status").textContent = data.status || "Pendiente";
  qs("#eval-revision").textContent = data.source_revision ? `Fuente ${data.source_revision}` : "Sin ejecución registrada";
  const metrics = data.metrics || {};
  qs("#metrics").innerHTML = Object.entries(metrics).slice(0, 12).map(([name, value]) => `
    <div class="metric"><strong>${typeof value === "number" ? (value <= 1 ? (value * 100).toFixed(1) + "%" : value.toFixed(1)) : escapeHtml(String(value))}</strong><span>${escapeHtml(name.replaceAll("_", " "))}</span></div>`).join("") || "<div class='metric'><strong>—</strong><span>Ejecute cima-rag evaluate</span></div>";
  const thresholds = data.thresholds || {};
  qs("#threshold-list").innerHTML = Object.entries(thresholds).map(([name, threshold]) => {
    const value = metrics[name]; const passed = typeof value === "number" && value >= threshold;
    return `<div class="threshold-row"><span>${escapeHtml(name.replaceAll("_", " "))}</span><span>${typeof value === "number" ? value.toFixed(3) : "—"}</span><strong class="${passed ? "pass" : "fail"}">${threshold.toFixed(2)}</strong></div>`;
  }).join("");
}

checkHealth();


