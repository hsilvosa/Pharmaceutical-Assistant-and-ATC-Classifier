"""
Interactive Streamlit Web UI for Spanish Pharmaceutical Assistant & ATC Classifier.
"""
from src.utils import setup_environment

setup_environment()

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import ATC_LEVEL_1_MAP
from src.inference import ATCClassifierEngine, CIMAMedicalAssistantEngine

st.set_page_config(
    page_title="CIMA Pharmaceutical Assistant & ATC Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark/glassmorphic modern aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-lbl {
        font-size: 0.9rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .atc-badge {
        background-color: #0284C7;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engines():
    atc_eng = ATCClassifierEngine()
    qa_eng = CIMAMedicalAssistantEngine()
    return atc_eng, qa_eng

def render_rag_response(response, question: str) -> None:
    if response.status == "answered":
        st.markdown(response.answer)
    elif response.status == "needs_disambiguation":
        st.info(response.answer)
        options = {
            f"{item.name} | {item.presentation or item.registration_number}": item
            for item in response.candidates
        }
        selected_label = st.selectbox("Medicamento", options)
        if st.button("Consultar esta presentación"):
            selected = options[selected_label]
            resolved = qa_engine.query_evidence(
                question,
                registration_number=selected.registration_number,
                language="auto",
            )
            render_rag_response(resolved, question)
        return
    elif response.status == "refused":
        st.warning(response.answer)
    else:
        st.info(response.answer)

    if response.citations:
        st.markdown("#### Fuentes oficiales")
    for citation in response.citations:
        label = f"{citation.medicine_name} · {citation.title or citation.section}"
        with st.expander(label):
            st.write(citation.quote)
            st.caption(
                f"Registro {citation.registration_number} · "
                f"Documento {citation.document_type} · Sección {citation.section}"
            )
            st.link_button("Abrir documento oficial AEMPS", citation.url)

    st.caption(
        f"Fuente {response.source_revision[:12]} · "
        f"{response.latency_ms.get('total', 0):.0f} ms"
    )

atc_engine, qa_engine = load_engines()

# Header
st.markdown('<div class="main-header">🏥 Spanish Pharmaceutical Assistant & ATC Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Modelos de IA aplicados al Dataset Oficial AEMPS CIMA (1.17M registros de medicamentos, prospectos y códigos ATC)</div>', unsafe_allow_html=True)

tabs = st.tabs([
    "💬 Asistente Médico QA & Síntomas",
    "Evidencia CIMA RAG",
    "🏷️ Clasificador Jerárquico ATC",
    "💊 Buscador de Medicamentos AEMPS",
    "📊 Benchmarks & Métricas"
])

# --- TAB 1: Existing symptom and medication assistant ---
with tabs[0]:
    st.markdown("### Consulta médica específica o síntomas")
    st.write(
        "Describa un síntoma o problema de salud para consultar las categorías ATC, "
        "principios activos y medicamentos registrados en el catálogo CIMA."
    )

    st.markdown("#### Ejemplos rápidos")
    examples = [
        "no puedo ir al baño",
        "tengo dolor al orinar",
        "tengo una quemadura en el brazo",
        "dolor muscular por golpe",
        "dolor de cabeza fuerte",
        "ojo rojo y picores",
    ]
    columns = st.columns(3)
    for index, example in enumerate(examples):
        if columns[index % 3].button(example.capitalize(), key=f"symptom-{index}"):
            st.session_state["symptom_query"] = example

    symptom_query = st.text_input(
        "Describa su problema de salud o busque un medicamento:",
        value=st.session_state.get("symptom_query", ""),
        key="symptom-query-input",
    )
    if st.button("Consultar asistente", type="primary", key="symptom-submit"):
        if symptom_query:
            with st.spinner("Consultando el catálogo CIMA..."):
                response = qa_engine.answer_question(symptom_query)
                st.markdown(response["answer"])
        else:
            st.warning("Introduce una consulta o síntoma.")

# --- TAB 2: Grounded CIMA RAG ---
with tabs[1]:
    st.markdown("### Evidencia oficial CIMA")
    st.caption("Información documental con citas verificables. No ofrece diagnóstico ni tratamiento personal.")

    query_input = st.text_input(
        "Pregunta sobre un medicamento:",
        value=st.session_state.get("user_query", ""),
        placeholder="Ej.: ¿Qué contraindicaciones recoge la ficha técnica de este medicamento?"
    )
    
    if st.button("Consultar documentación AEMPS", type="primary"):
        if query_input:
            with st.spinner("Buscando evidencia oficial..."):
                try:
                    response = qa_engine.query_evidence(query_input, language="auto")
                    render_rag_response(response, query_input)
                except (FileNotFoundError, RuntimeError, ValueError) as exc:
                    st.error(
                        "El índice RAG todavía no está disponible. "
                        "Ejecuta `cima-rag index build` antes de consultar."
                    )
                    st.caption(str(exc))
        else:
            st.warning("Introduce una pregunta.")

# --- TAB 3: ATC Classifier ---
with tabs[2]:
    st.markdown("### 🏷️ Clasificación de Taxonomía ATC (Nivel 1-5)")
    st.write("Introduzca la descripción del medicamento, principios activos o fragmento del prospecto para predecir su categoría anatómica y terapéutica ATC.")
    
    text_to_classify = st.text_area(
        "Texto del medicamento / prospecto:",
        height=140,
        value="Medicamento: Omeprazol 20 mg | Forma farmacéutica: COMPRIMIDO | Principios activos: Omeprazol | Vía de administración: ORAL | Indicación: tratamiento del reflujo gastroesofágico y úlceras duodenales."
    )
    
    col_btn, col_topk = st.columns([2, 1])
    with col_topk:
        top_k = st.slider("Top Predictions:", min_value=1, max_value=5, value=3)
    
    if st.button("⚡ Predecir Código ATC", type="primary"):
        if text_to_classify:
            with st.spinner("Ejecutando BETO Hierarchical Multi-Label Classifier..."):
                res = atc_engine.predict(text_to_classify, top_k=top_k)
                
                if "error" in res:
                    st.error(res["error"])
                else:
                    top_pred = res["top_prediction"]
                    st.success(f"**Categoría Principal Predicha**: `{top_pred['atc_level1_code']}` - {top_pred['atc_level1_name']} ({top_pred['percentage']} confianza)")
                    
                    st.markdown("#### Predicciones Top-K:")
                    for pred in res["all_predictions"]:
                        st.write(f"**Nivel 1 `{pred['atc_level1_code']}`**: {pred['atc_level1_name']}")
                        st.progress(pred["confidence"])
                        st.caption(f"Confianza: {pred['percentage']}")
                        st.divider()
        else:
            st.warning("Introduzca texto para clasificar.")

# --- TAB 4: CIMA Medication Explorer ---
with tabs[3]:
    st.markdown("### 💊 Buscador de Medicamentos CIMA")
    search_q = st.text_input("Buscar por Nombre, Principio Activo o Código ATC:", value="Fosfomicina")
    
    if search_q:
        results = qa_engine.search_medications(search_q, limit=20)
        if results:
            df_res = pd.DataFrame(results)
            st.dataframe(df_res, use_container_width=True)
        else:
            st.info("No se encontraron registros coincidentes.")

# --- TAB 5: Evaluation Benchmarks ---
with tabs[4]:
    st.markdown("### Evaluación reproducible del sistema RAG")
    report_paths = sorted(
        (Path("artifacts/evaluations")).glob("*/report.json"), reverse=True
    )
    report_path = report_paths[0] if report_paths else Path("eval/baseline.json")
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    metrics = report.get("metrics", {})

    if metrics:
        selected_metrics = (
            ("Recall@5", "recall_at_5"),
            ("MRR", "mrr"),
            ("Precisión de citas", "citation_precision"),
            ("Cobertura de citas", "citation_coverage"),
            ("Recall de valores esperados", "answer_recall"),
            ("Exactitud de rechazo", "refusal_accuracy"),
        )
        columns = st.columns(3)
        for index, (label, key) in enumerate(selected_metrics):
            columns[index % 3].metric(label, f"{metrics.get(key, 0):.3f}")
        st.caption(
            f"Estado: {report.get('status', 'unknown')} · "
            f"Fuente: {report.get('source_revision', 'sin registrar')}"
        )
    else:
        st.info("La evaluación completa fijada todavía no se ha ejecutado.")

    st.markdown("#### Categorías Anatómicas Principales ATC (Nivel 1)")
    atc_df = pd.DataFrame(list(ATC_LEVEL_1_MAP.items()), columns=["Código ATC", "Nombre del Grupo Anatómico"])
    st.table(atc_df)
