"""Build the small data files the public demo needs (neighbour index and calibration).

Usage: python scripts/build_space_assets.py
Reads data/processed/atc_dataset.parquet and models/atc_classifier/calibration.json.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    out_dir = ROOT / "space" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(
        ROOT / "data" / "processed" / "atc_dataset.parquet",
        columns=["name", "active_ingredients_str", "pharmaceutical_form_name", "atc_code", "atc_level1"],
    )
    df = df.dropna(subset=["name", "atc_code"])
    df["atc_code"] = df["atc_code"].astype(str)
    # One row per medicine name: keep the most specific ATC code it has.
    df["depth"] = df["atc_code"].str.len()
    df = df.sort_values("depth", ascending=False).drop_duplicates("name").drop(columns="depth")
    df = df.rename(columns={"active_ingredients_str": "ingredients", "pharmaceutical_form_name": "form"})
    df["ingredients"] = df["ingredients"].fillna("")
    df["form"] = df["form"].fillna("")
    path = out_dir / "cima_neighbors.csv.gz"
    df.to_csv(path, index=False, compression="gzip")
    calibration = ROOT / "models" / "atc_classifier" / "calibration.json"
    shutil.copy(calibration, ROOT / "space" / "calibration.json")
    shutil.copy(ROOT / "eval" / "atc_input_ablation.json", out_dir / "atc_input_ablation.json")
    forms = df["form"].value_counts().head(30).index.tolist()
    (out_dir / "forms.json").write_text(json.dumps(forms, ensure_ascii=False), encoding="utf-8")
    static = ROOT / "static-space" / "data"
    static.mkdir(parents=True, exist_ok=True)
    for name in ("calibration.json",):
        shutil.copy(ROOT / "space" / name, static / name)
    shutil.copy(ROOT / "space" / "gallery.json", static / "gallery.json")
    shutil.copy(out_dir / "atc_input_ablation.json", static / "atc_input_ablation.json")
    (static / "forms.json").write_text(json.dumps(forms, ensure_ascii=False), encoding="utf-8")
    compact = df[["name", "atc_code", "ingredients"]].values.tolist()
    (static / "neighbors.json").write_text(json.dumps(compact, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{len(df)} medicines -> {path} ({path.stat().st_size / 1e6:.1f} MB), top forms: {forms[:3]}", file=sys.stderr)


if __name__ == "__main__":
    main()
