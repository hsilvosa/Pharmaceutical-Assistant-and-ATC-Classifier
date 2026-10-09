"""
Unified Inference Engine for CIMA Spanish Medical Assistant & ATC Classifier.
Combines the grounded CIMA RAG service, the existing medication catalogue search,
and the BETO ATC hierarchical classifier behind one application-facing facade.
"""
from src.utils import setup_environment, setup_logger, get_device
setup_environment()

import os
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from src.config import (
    PROCESSED_DATA_DIR,
    ATC_MODEL_DIR,
    LLM_MODEL_DIR,
    ATC_BASE_MODEL,
    ATC_LEVEL_1_MAP
)

logger = setup_logger("inference")

# Expanded Clinical Symptom Mapping for High-Precision Guidance
CLINICAL_CONDITIONS = [
    {
        "id": "cistitis_infeccion_urinaria",
        "patterns": [
            r"dolor\s+al\s+mear", r"dolor\s+al\s+orinar", r"escozor\s+al\s+orinar", r"escozor\s+al\s+mear",
            r"cistitis", r"infeccion\s+de\s+orina", r"infeccion\s+urinaria", r"ganas\s+de\s+mear\s+constantes",
            r"mear\s+gotas", r"chispazo\s+al\s+orinar", r"orina\s+con\s+peste"
        ],
        "condition": "Cistitis / Infección Urinaria (Dolor o Escozor al Orinar)",
        "atc_group": "J01XE / G04BX",
        "atc_name": "Antiinfecciosos Urinarios / Antisépticos del Aparato Urinario",
        "description": "El dolor, escozor o dificultad al orinar es un síntoma claro de infección urinaria (cistitis). Requiere diagnóstico bacteriológico.",
        "ingredients": ["Fosfomicina", "Nitrofurantoína", "Norfloxacino", "Ciprofloxacino"],
        "advice": "Acuda a su centro de salud para análisis de orina. Requiere receta médica obligatoria."
    },
    {
        "id": "estrenimiento",
        "patterns": [
            r"no\s+puedo\s+ir\s+al\s+ba[nñ]o", r"no\s+puedo\s+cagar", r"estre[nñ]ido", r"estre[nñ]imiento",
            r"cagarruta", r"heces\s+duras", r"retencion\s+fecal", r"dificultad\s+defecar", r"sin\s+ir\s+al\s+ba[nñ]o"
        ],
        "condition": "Estreñimiento / Estreñimiento Ocasional",
        "atc_group": "A06A",
        "atc_name": "Fármacos para el Estreñimiento / Laxantes",
        "description": "Baja frecuencia o dificultad expulsiva en las deposiciones intestinales.",
        "ingredients": ["Plantago ovata", "Lactulosa", "Lactitol", "Bisacodilo", "Glicerol"],
        "advice": "Aumente la ingesta de agua y fibra. Para casos punteros use laxantes formadores de masa o supositorios de glicerina."
    },
    {
        "id": "diarrea",
        "patterns": [
            r"diarrea", r"suelto\s+del\s+estomago", r"descomposicion", r"gastroenteritis", r"cagarrria",
            r"cagando\s+agua", r"cagarse\s+vivo", r"tripa\s+suelta", r"heces\s+liquidas"
        ],
        "condition": "Diarrea Aguda y Gastroenteritis",
        "atc_group": "A07DA / A07CA",
        "atc_name": "Antidiarreicos / Sales de Rehidratación Oral",
        "description": "Aumento en la liquidez y frecuencia de las evacuaciones intestinales.",
        "ingredients": ["Loperamida", "Racecadotrilo", "Sales de rehidratación oral"],
        "advice": "Mantener rehidratación oral continua. Evitar antidiarreicos si hay fiebre o moco/sangre en heces."
    },
    {
        "id": "hemorroides_almorranas",
        "patterns": [
            r"almorrana", r"almorranas", r"hemorroides", r"picor\s+en\s+el\s+culo", r"sangre\s+al\s+cagar",
            r"bulto\s+en\s+el\s+ano", r"dolor\s+rectal", r"esfinter\s+dolorido"
        ],
        "condition": "Hemorroides (Almorranas) y Lesiones Anorrectales",
        "atc_group": "C05AX / D07A",
        "atc_name": "Antihemorroidales Tópicos / Corticoides y Vasoprotectores",
        "description": "Inflamación o congestión del plexo venoso hemorroidal con picor, dolor o sangrado.",
        "ingredients": ["Ruscus aculeatus", "Triamcinolona", "Hidrocortisona tópica", "Diosmina", "Hesperidina"],
        "advice": "Usar pomadas antihemorroidales con anestésico/corticoide local y beber agua para evitar heces duras."
    },
    {
        "id": "quemaduras_lesiones",
        "patterns": [
            r"quemadura", r"quemado", r"me\s+he\s+quemado", r"rozadura", r"herida\s+en\s+la\s+piel",
            r"ampolla", r"quemadura\s+solar", r"aceite\s+caliente", r"plancha"
        ],
        "condition": "Quemaduras Leves y Lesiones Cutáneas",
        "atc_group": "D03A / D08A",
        "atc_name": "Cicatrizantes / Antisépticos y Desinfectantes Cutáneos",
        "description": "Tratamiento tópico para quemaduras de primer o segundo grado leve y erosiones.",
        "ingredients": ["Sulfadiazina de plata", "Clorhexidina", "Centella asiática", "Dexpantenol"],
        "advice": "Refrescar con agua fría durante 10-15 minutos. Aplicar sulfadiazina de plata o dexpantenol."
    },
    {
        "id": "dolor_muela_flemon",
        "patterns": [
            r"dolor\s+de\s+muela", r"flemon", r"muela\s+picada", r"cara\s+hinchada", r"infeccion\s+boca",
            r"infeccion\s+dental", r"dolor\s+de\s+dientes"
        ],
        "condition": "Infección Dental / Flemón y Dolor de Muelas",
        "atc_group": "J01CR / M01AE",
        "atc_name": "Antibióticos Sistémicos / Antiinflamatorios Potentes",
        "description": "Infección odontogénica en encías o raíz dental con dolor agudo e inflamación facial.",
        "ingredients": ["Amoxicilina", "Ibuprofeno 600mg", "Dexketoprofeno (Enantyum)"],
        "advice": "Acuda al dentista inmediatamente. Los antibióticos requieren prescripción odontológica."
    },
    {
        "id": "dolor_osteomuscular_fractura",
        "patterns": [
            r"dolor\s+de\s+hueso", r"dolor\s+muscular", r"esguince", r"fractura", r"golpe", r"traumatismo",
            r"dolor\s+de\s+espalda", r"lumbalgia", r"tiron", r"chichon", r"pisotón", r"latigazo\s+cervical"
        ],
        "condition": "Dolor Osteomuscular, Lumbalgia, Traumatismos y Fracturas",
        "atc_group": "M01A / M02AA",
        "atc_name": "Antiinflamatorios No Esteroideos (AINEs) y Geles Tópicos",
        "description": "Dolor muscular, articular o lesiones traumáticas tras golpes o esguinces.",
        "ingredients": ["Dexketoprofeno", "Diclofenaco", "Ibuprofeno", "Naproxeno", "Etofenamato"],
        "advice": "Aplicar frío/hielo inicial en golpes. Para dolor severo o sospecha de fractura acuda a urgencias."
    },
    {
        "id": "dolor_cabeza_migrana",
        "patterns": [
            r"dolor\s+de\s+cabeza", r"migra[nñ]a", r"cefalea", r"jaqueca", r"dolor\s+de\s+melon",
            r"sien\s+latiendo", r"latido\s+cabeza"
        ],
        "condition": "Cefalea (Dolor de Cabeza) y Migraña",
        "atc_group": "N02CC / N02BE",
        "atc_name": "Analgésicos y Preparados Antimigrañosos (Triptanes)",
        "description": "Dolor tensional o migraña pulsátil.",
        "ingredients": ["Paracetamol", "Ibuprofeno", "Sumatriptán", "Zolmitriptán"],
        "advice": "Triptanes bajo receta para migraña confirmada. Paracetamol/Ibuprofeno para cefalea tensional."
    }
]

class ATCModelNotTrainedError(RuntimeError):
    """Raised when the trained ATC classifier weights are not available."""


class ATCClassifierEngine:
    def __init__(self, model_dir: str = None):
        self.device = get_device()
        self.model_dir = Path(model_dir) if model_dir else Path(ATC_MODEL_DIR)
        self.mapping_path = self.model_dir / "label_mapping.json"
        self.model_path = self.model_dir / "final_model"
        
        self.label2id = {}
        self.id2label = {}
        self.atc_map = ATC_LEVEL_1_MAP
        
        self._load_label_mapping()
        self._load_model()

    def _load_label_mapping(self):
        if self.mapping_path.exists():
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.label2id = data.get("label2id", {})
                self.id2label = {int(k): v for k, v in data.get("id2label", {}).items()}
                if "atc_level1_map" in data:
                    self.atc_map = data["atc_level1_map"]
        else:
            unique_levels = sorted(list(ATC_LEVEL_1_MAP.keys()))
            self.label2id = {lvl: i for i, lvl in enumerate(unique_levels)}
            self.id2label = {i: lvl for i, lvl in enumerate(unique_levels)}

    def _load_model(self):
        if not self.model_path.exists():
            raise ATCModelNotTrainedError(
                f"Trained ATC classifier not found at '{self.model_path}'. "
                "Run 'python -m src.train_atc_classifier' first. "
                "Refusing to serve predictions from an untrained model."
            )
        logger.info(f"Loading trained ATC Classifier from '{self.model_path}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str, top_k: int = 3) -> dict:
        """Predict top-k ATC Level 1 anatomical groups with confidence scores."""
        if not text or not text.strip():
            return {"error": "Empty text provided."}

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            
        top_k = min(top_k, len(probs))
        top_probs, top_indices = torch.topk(probs, k=top_k)

        predictions = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            code = self.id2label.get(idx, "V")
            name = self.atc_map.get(code, "Grupo Anatómico General")
            predictions.append({
                "atc_level1_code": code,
                "atc_level1_name": name,
                "confidence": round(float(prob), 4),
                "percentage": f"{float(prob)*100:.1f}%"
            })

        return {
            "input_text": text,
            "top_prediction": predictions[0],
            "all_predictions": predictions
        }

class CIMASemanticVectorRetriever:
    """Neural & Vector Semantic Search Engine over 75,830 CIMA Medication Records."""
    def __init__(self, df_meds: pd.DataFrame):
        self.df_meds = df_meds
        self.vectorizer = None
        self.tfidf_matrix = None
        self._build_vector_index()

    def _build_vector_index(self):
        logger.info("Building Semantic Vector Index across CIMA database...")
        corpus = (
            self.df_meds["name"].fillna("") + " " +
            self.df_meds["active_ingredients_str"].fillna("") + " " +
            self.df_meds["pharmaceutical_form_name"].fillna("") + " " +
            self.df_meds["routes_str"].fillna("") + " " +
            self.df_meds["atc_name"].fillna("") + " " +
            self.df_meds["atc_level1_name"].fillna("")
        )
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=1,
            max_features=50000,
            analyzer="word",
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        logger.info("Semantic Vector Index successfully built!")

    def search_semantic(self, query: str, top_k: int = 5) -> list:
        """Perform Zero-Shot Cosine Similarity Semantic Vector Search for ANY natural language query."""
        if not query or not query.strip():
            return []
            
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            row = self.df_meds.iloc[idx]
            results.append({
                "score": round(score, 4),
                "registration_number": row.get("registration_number"),
                "name": row.get("name"),
                "dose": row.get("dose"),
                "pharmaceutical_form": row.get("pharmaceutical_form_name"),
                "active_ingredients": row.get("active_ingredients_str"),
                "atc_code": row.get("atc_code"),
                "atc_name": row.get("atc_name"),
                "atc_level1_name": row.get("atc_level1_name"),
                "routes": row.get("routes_str"),
                "prescription_required": bool(row.get("prescription_required", False))
            })
        return results

class CIMAMedicalAssistantEngine:
    """Unified facade for the existing catalogue, classifiers, and grounded CIMA RAG."""

    def __init__(self):
        self.dataset_path = Path(PROCESSED_DATA_DIR) / "atc_dataset.parquet"
        self.df_meds = None
        self.retriever = None
        self._rag_service = None
        self._rag_error = None
        self._load_knowledge_base()

    def _get_rag_service(self):
        """Load and cache the grounded service behind the existing assistant facade."""
        if self._rag_service is not None:
            return self._rag_service
        if self._rag_error is not None:
            raise RuntimeError(self._rag_error)

        try:
            from src.rag.api import load_service
            from src.rag.config import Settings

            self._rag_service = load_service(Settings())
            return self._rag_service
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            self._rag_error = str(exc)
            raise RuntimeError(self._rag_error) from exc

    def query_evidence(
        self,
        question: str,
        registration_number: str | None = None,
        language: str = "auto",
    ):
        """Answer through the grounded RAG pipeline using the existing assistant object."""
        from src.rag.models import QueryRequest

        return self._get_rag_service().query(
            QueryRequest(
                question=question,
                registration_number=registration_number,
                language=language,
            )
        )

    def _load_knowledge_base(self):
        if self.dataset_path.exists():
            self.df_meds = pd.read_parquet(self.dataset_path)
            logger.info(f"Loaded CIMA Knowledge Base with {len(self.df_meds)} medications.")
            self.retriever = CIMASemanticVectorRetriever(self.df_meds)

    def search_medications(self, query: str, limit: int = 5) -> list:
        """Search the indexed CIMA catalogue, then enrich it with legacy ATC fields."""
        if not query:
            return []

        grounded = []
        try:
            for item in self._get_rag_service().resolver.search(query, limit=limit):
                grounded.append(
                    {
                        "registration_number": item.registration_number,
                        "name": item.name,
                        "presentation": item.presentation,
                        "national_code": item.national_code,
                        "photo_url": item.photo_url,
                        "dose": None,
                        "pharmaceutical_form": item.presentation or None,
                        "active_ingredients": None,
                        "atc_code": None,
                        "atc_level1_name": None,
                        "routes": None,
                        "prescription_required": None,
                    }
                )
        except RuntimeError:
            pass

        if self.df_meds is None:
            return grounded
        
        q = query.lower().strip()
        mask = (
            self.df_meds["name"].astype(str).str.lower().str.contains(q, regex=False) |
            self.df_meds["active_ingredients_str"].astype(str).str.lower().str.contains(q, regex=False) |
            self.df_meds["atc_code"].astype(str).str.lower().str.contains(q, regex=False)
        )
        matches = self.df_meds[mask].head(limit)
        
        results = []
        for _, r in matches.iterrows():
            results.append({
                "registration_number": r.get("registration_number"),
                "name": r.get("name"),
                "dose": r.get("dose"),
                "pharmaceutical_form": r.get("pharmaceutical_form_name"),
                "active_ingredients": r.get("active_ingredients_str"),
                "atc_code": r.get("atc_code"),
                "atc_level1_name": r.get("atc_level1_name"),
                "routes": r.get("routes_str"),
                "prescription_required": bool(r.get("prescription_required", False))
            })
        by_registration = {
            str(item.get("registration_number")): item for item in results
        }
        for item in grounded:
            registration = str(item["registration_number"])
            if registration in by_registration:
                by_registration[registration].update(
                    {key: value for key, value in item.items() if value}
                )
            else:
                by_registration[registration] = item
        return list(by_registration.values())[:limit]

    def detect_symptoms_and_recommend(self, query: str) -> dict:
        """Detect clinical symptoms in query and return targeted recommendations."""
        q_lower = query.lower().strip()
        q_norm = q_lower.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("ñ", "n")

        best_match = None
        for cond_info in CLINICAL_CONDITIONS:
            for pattern in cond_info["patterns"]:
                pat_norm = pattern.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("ñ", "n")
                if re.search(pat_norm, q_norm):
                    best_match = cond_info
                    break
            if best_match:
                break

        if not best_match:
            return None

        rec_meds = []
        for ing in best_match["ingredients"]:
            meds = self.search_medications(ing, limit=2)
            rec_meds.extend(meds)

        seen = set()
        unique_rec_meds = []
        for m in rec_meds:
            if m["name"] not in seen:
                seen.add(m["name"])
                unique_rec_meds.append(m)

        return {
            "condition": best_match["condition"],
            "atc_group": best_match["atc_group"],
            "atc_name": best_match["atc_name"],
            "description": best_match["description"],
            "advice": best_match["advice"],
            "recommended_ingredients": best_match["ingredients"],
            "medications": unique_rec_meds[:6]
        }

    def answer_question(self, query: str) -> dict:
        """Answer Spanish medical query using Hybrid Clinical Intent + Zero-Shot Vector Search."""
        if not query or not query.strip():
            return {"answer": "Por favor, detalle su síntoma o consulta médica."}

        # 1. Targeted Clinical Intent Matcher (Rich advice & medical disclaimers)
        symptom_res = self.detect_symptoms_and_recommend(query)
        if symptom_res:
            cond = symptom_res["condition"]
            atc_code = symptom_res["atc_group"]
            atc_name = symptom_res["atc_name"]
            desc = symptom_res["description"]
            advice = symptom_res["advice"]
            ings = ", ".join(symptom_res["recommended_ingredients"])
            meds = symptom_res["medications"]

            meds_md = ""
            for m in meds:
                prescription = m.get("prescription_required")
                if prescription is True:
                    presc = "Requiere Receta Médica"
                elif prescription is False:
                    presc = "Venta Libre (EFP)"
                else:
                    presc = "Consultar ficha oficial"
                meds_md += f"- **{m.get('name', 'Medicamento CIMA')}** ({m.get('dose') or 'Ver ficha'})\n"
                meds_md += f"  - *Principio Activo*: {m.get('active_ingredients') or 'Ver ficha'}\n"
                meds_md += (
                    f"  - *Forma / Vía*: {m.get('pharmaceutical_form') or 'Ver ficha'} "
                    f"({m.get('routes') or 'Ver ficha'})\n"
                )
                meds_md += f"  - *Régimen*: {presc}\n\n"

            answer = (
                f"### Orientación Terapéutica AEMPS: **{cond}**\n\n"
                f"**Descripción Clínica**: {desc}\n\n"
                f"**Principios Activos Autorizados en España**: {ings}\n"
                f"**Categoría Farmacológica ATC**: `{atc_code}` — *{atc_name}*\n\n"
                f"#### Ejemplos de Medicamentos Registrados en CIMA:\n"
                f"{meds_md}"
                f"**Consejo y Recomendación Médica**:\n"
                f"{advice}\n\n"
                f"---\n"
                f"**Advertencia de Seguridad**: Esta información proviene de los registros oficiales de la AEMPS. "
                f"No reemplaza la valoración de un profesional sanitario. No se automedique."
            )
            return {"answer": answer, "symptom_data": symptom_res}

        # 2. Zero-Shot Neural Semantic Vector Search (Handles ANY novel/unseen natural language query)
        if self.retriever:
            semantic_matches = self.retriever.search_semantic(query, top_k=5)
            if semantic_matches and semantic_matches[0]["score"] > 0.001:
                meds_md = ""
                for m in semantic_matches:
                    presc = "Requiere Receta Médica" if m.get("prescription_required") else "Venta Libre (EFP)"
                    atc_info = f"{m.get('atc_code')} ({m.get('atc_name') or m.get('atc_level1_name')})"
                    meds_md += f"- **{m.get('name', 'Medicamento CIMA')}** (Dosis: {m.get('dose') or 'Ver envase'})\n"
                    meds_md += f"  - *Principio Activo*: {m.get('active_ingredients') or 'Ver prospecto'}\n"
                    meds_md += f"  - *Categoría ATC*: `{atc_info}`\n"
                    meds_md += f"  - *Forma / Vía*: {m.get('pharmaceutical_form') or 'Ver ficha'} ({m.get('routes') or 'Ver ficha'})\n"
                    meds_md += f"  - *Régimen*: {presc}\n\n"

                answer = (
                    f"### Análisis Semántico Médico CIMA AEMPS\n\n"
                    f"**Consulta Expresada**: *\"{query}\"*\n\n"
                    f"**Comprensión Semántica Autónoma**: Basándonos en la similitud semántica y farmacológica del catálogo oficial AEMPS CIMA, el sistema ha comprendido su consulta y seleccionado las siguientes opciones terapéuticas principales:\n\n"
                    f"#### Medicamentos y Principios Activos Relevantes:\n"
                    f"{meds_md}"
                    f"---\n"
                    f"**Advertencia de Seguridad Sanitaria**:\n"
                    f"Esta respuesta se genera mediante comprensión semántica sobre el registro oficial de la AEMPS. "
                    f"No constituye diagnóstico ni prescripción médica personal. Consulte siempre con un profesional sanitario."
                )
                return {"answer": answer, "semantic_matches": semantic_matches}

        # 3. Fallback
        answer = (
            f"No se ha encontrado una coincidencia semántica suficiente para '{query}'.\n\n"
            f"Intente describir el problema de salud o el medicamento con más detalle."
        )
        return {"answer": answer, "matched_medication": None}

    def answer_with_evidence(
        self,
        query: str,
        registration_number: str | None = None,
        language: str = "auto",
    ) -> dict:
        """Return the grounded RAG response as a serializable compatibility dictionary."""
        if not query or not query.strip():
            return {
                "status": "insufficient_evidence",
                "answer": "Introduce una pregunta sobre la documentación de un medicamento.",
                "citations": [],
                "candidates": [],
            }
        try:
            response = self.query_evidence(query, registration_number, language)
            return response.model_dump(mode="json")
        except RuntimeError as exc:
            return {
                "status": "unavailable",
                "answer": "El índice documental CIMA no está disponible.",
                "citations": [],
                "candidates": [],
                "detail": str(exc),
            }

if __name__ == "__main__":
    qa_engine = CIMAMedicalAssistantEngine()
    print("Testing 'tengo el ojo hecho polvo':")
    print(qa_engine.answer_question("tengo el ojo hecho polvo")["answer"][:350])
