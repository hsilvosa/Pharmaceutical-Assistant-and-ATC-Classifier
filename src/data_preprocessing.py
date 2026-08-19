"""
CIMA Dataset Preprocessing & ETL Pipeline.
Ingests AEMPS CIMA relational dataset and prepares formatted benchmarks for:
1. BETO ATC Hierarchical Classifier (multi-label / hierarchical ATC prediction)
2. CIMA Spanish Medical QA Assistant (grounded instruction-tuning dataset)
"""
from src.utils import setup_environment, setup_logger
# Execute SSL environment fix before loading datasets
setup_environment()

import os
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import load_dataset

from src.config import (
    HF_DATASET_NAME,
    PROCESSED_DATA_DIR,
    ATC_LEVEL_1_MAP
)

logger = setup_logger("data_preprocessing")

def clean_html_text(text: str) -> str:
    """Strip HTML tags and clean up whitespace."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_atc_levels(atc_code: str) -> dict:
    """Extract hierarchical ATC levels (1 to 5) from standard ATC code."""
    if not atc_code or not isinstance(atc_code, str):
        return {}
    atc_code = atc_code.strip().upper()
    return {
        "atc_level1": atc_code[0] if len(atc_code) >= 1 else "",
        "atc_level2": atc_code[:3] if len(atc_code) >= 3 else "",
        "atc_level3": atc_code[:4] if len(atc_code) >= 4 else "",
        "atc_level4": atc_code[:5] if len(atc_code) >= 5 else "",
        "atc_level5": atc_code if len(atc_code) >= 7 else ""
    }

class CIMADatasetBuilder:
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self.output_dir = Path(PROCESSED_DATA_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_cima_tables(self):
        """Load relational tables from HuggingFace hsilvosa/aemps-cima."""
        logger.info(f"Loading dataset tables from HF repository: '{HF_DATASET_NAME}'...")
        
        meds_ds = load_dataset(HF_DATASET_NAME, "medications", split="train")
        atc_ds = load_dataset(HF_DATASET_NAME, "atc_codes", split="train")
        active_ds = load_dataset(HF_DATASET_NAME, "active_ingredients", split="train")
        excip_ds = load_dataset(HF_DATASET_NAME, "excipients", split="train")
        routes_ds = load_dataset(HF_DATASET_NAME, "administration_routes", split="train")
        
        logger.info(f"Loaded {len(meds_ds)} medications, {len(atc_ds)} ATC records, {len(active_ds)} active ingredients.")
        
        df_meds = meds_ds.to_pandas()
        df_atc = atc_ds.to_pandas()
        df_active = active_ds.to_pandas()
        df_excip = excip_ds.to_pandas()
        df_routes = routes_ds.to_pandas()
        
        return df_meds, df_atc, df_active, df_excip, df_routes

    def build_atc_dataset(self, df_meds, df_atc, df_active, df_routes):
        """Construct dataset for ATC Hierarchical Multi-Label Classifier."""
        logger.info("Building ATC Hierarchical Classification dataset...")
        
        active_grouped = (
            df_active.groupby("registration_number")["ingredient_name"]
            .apply(lambda x: ", ".join(x.dropna().unique()))
            .reset_index()
            .rename(columns={"ingredient_name": "active_ingredients_str"})
        )
        
        routes_grouped = (
            df_routes.groupby("registration_number")["route_name"]
            .apply(lambda x: ", ".join(x.dropna().unique()))
            .reset_index()
            .rename(columns={"route_name": "routes_str"})
        )
        
        df_merged = df_meds.merge(df_atc, on="registration_number", how="inner")
        df_merged = df_merged.merge(active_grouped, on="registration_number", how="left")
        df_merged = df_merged.merge(routes_grouped, on="registration_number", how="left")
        
        df_merged["active_ingredients_str"] = df_merged["active_ingredients_str"].fillna("")
        df_merged["routes_str"] = df_merged["routes_str"].fillna("")
        df_merged["pharmaceutical_form_name"] = df_merged["pharmaceutical_form_name"].fillna("")
        df_merged["dose"] = df_merged["dose"].fillna("")
        
        atc_levels_df = df_merged["atc_code"].apply(extract_atc_levels).apply(pd.Series)
        df_merged = pd.concat([df_merged, atc_levels_df], axis=1)
        
        df_merged = df_merged[df_merged["atc_level1"].str.len() > 0].copy()
        df_merged["atc_level1_name"] = df_merged["atc_level1"].map(ATC_LEVEL_1_MAP).fillna("Desconocido")
        
        def build_text_input(row):
            parts = [f"Medicamento: {row['name']}"]
            if row["dose"]:
                parts.append(f"Dosis: {row['dose']}")
            if row["pharmaceutical_form_name"]:
                parts.append(f"Forma farmacéutica: {row['pharmaceutical_form_name']}")
            if row["active_ingredients_str"]:
                parts.append(f"Principios activos: {row['active_ingredients_str']}")
            if row["routes_str"]:
                parts.append(f"Vía de administración: {row['routes_str']}")
            return " | ".join(parts)

        df_merged["text_input"] = df_merged.apply(build_text_input, axis=1)
        
        output_file = self.output_dir / "atc_dataset.parquet"
        df_merged.to_parquet(output_file, index=False)
        logger.info(f"Saved ATC dataset with {len(df_merged)} rows to '{output_file}'")
        return df_merged

    def build_medical_qa_dataset(self, df_meds, df_active, df_excip, df_routes, sample_size: int = 15000):
        """Construct grounded Spanish Medical QA dataset for SLM fine-tuning."""
        logger.info("Building Spanish Medical Instruction QA dataset...")
        
        excip_grouped = (
            df_excip.groupby("registration_number")
            .apply(lambda g: [
                f"{r['excipient_name']} ({r['quantity']} {r['unit']})" if r['quantity'] else r['excipient_name']
                for _, r in g.iterrows() if r['excipient_name']
            ], include_groups=False)
            .reset_index(name="excipients_list")
        )
        
        active_grouped = (
            df_active.groupby("registration_number")["ingredient_name"]
            .apply(lambda x: ", ".join(x.dropna().unique()))
            .reset_index(name="active_ingredients_str")
        )
        
        routes_grouped = (
            df_routes.groupby("registration_number")["route_name"]
            .apply(lambda x: ", ".join(x.dropna().unique()))
            .reset_index(name="routes_str")
        )

        df_base = df_meds.merge(active_grouped, on="registration_number", how="left")
        df_base = df_base.merge(excip_grouped, on="registration_number", how="left")
        df_base = df_base.merge(routes_grouped, on="registration_number", how="left")
        
        df_base = df_base.sample(n=min(sample_size, len(df_base)), random_state=42)
        
        qa_pairs = []
        
        for _, row in df_base.iterrows():
            med_name = row["name"]
            act = row.get("active_ingredients_str", "") or row.get("virtual_therapeutic_moiety_name", "") or "No especificado"
            route = row.get("routes_str", "") or "No especificada"
            form = row.get("pharmaceutical_form_name", "") or "No especificada"
            dose = row.get("dose", "") or "Ver prospecto"
            presc = "Requiere receta médica." if row.get("prescription_required") else "No requiere receta médica."
            excip_list = row.get("excipients_list")
            
            is_valid_list = isinstance(excip_list, (list, np.ndarray)) and len(excip_list) > 0
            excip_str = ", ".join(excip_list[:5]) if is_valid_list else "Sin excipientes de declaración obligatoria señalados."
            
            qa1 = {
                "instruction": f"¿Cuál es la composición y uso principal del medicamento {med_name}?",
                "input": f"Medicamento: {med_name}",
                "output": f"El medicamento {med_name} (dosis: {dose}) contiene como principio activo principal: {act}. Su forma farmacéutica es {form}. {presc}"
            }
            
            qa2 = {
                "instruction": f"¿Cómo debe administrarse {med_name} y cuál es su vía de administración?",
                "input": f"Medicamento: {med_name}",
                "output": f"{med_name} se presenta en forma de {form} y se administra por vía {route}. Debe seguirse la pauta fijada en la ficha técnica oficial de la AEMPS."
            }
            
            qa3 = {
                "instruction": f"¿Qué excipientes o precauciones de seguridad contiene {med_name}?",
                "input": f"Medicamento: {med_name}",
                "output": f"{med_name} contiene los siguientes excipientes declarados: {excip_str}. Se recomienda verificar posibles alergias o intolerancias según la información oficial de la AEMPS."
            }
            
            qa_pairs.extend([qa1, qa2, qa3])
            
        output_file = self.output_dir / "medical_qa_dataset.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for item in qa_pairs:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
        logger.info(f"Saved {len(qa_pairs)} Spanish Medical QA instruction pairs to '{output_file}'")
        return qa_pairs

    def run_pipeline(self):
        """Execute full preprocessing workflow."""
        df_meds, df_atc, df_active, df_excip, df_routes = self.load_cima_tables()
        self.build_atc_dataset(df_meds, df_atc, df_active, df_routes)
        self.build_medical_qa_dataset(df_meds, df_active, df_excip, df_routes)
        logger.info("Data preprocessing pipeline completed successfully.")

if __name__ == "__main__":
    builder = CIMADatasetBuilder()
    builder.run_pipeline()
