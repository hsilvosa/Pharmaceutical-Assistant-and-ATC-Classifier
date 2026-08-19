"""
Unit tests for data preprocessing and ATC level extraction.
"""
import pytest
from src.data_preprocessing import extract_atc_levels, clean_html_text

def test_extract_atc_levels():
    res = extract_atc_levels("A02BC01")
    assert res["atc_level1"] == "A"
    assert res["atc_level2"] == "A02"
    assert res["atc_level3"] == "A02B"
    assert res["atc_level4"] == "A02BC"
    assert res["atc_level5"] == "A02BC01"

def test_clean_html_text():
    raw_html = "<div><p>Omeprazol <b>20mg</b> comprimidos.</p></div>"
    cleaned = clean_html_text(raw_html)
    assert cleaned == "Omeprazol 20mg comprimidos."
