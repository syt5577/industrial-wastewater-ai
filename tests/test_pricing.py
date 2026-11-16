import pandas as pd
from src.pricing import analyze_wastewater_price
import pytest

def test_rule_price():
    water_quality = {
        "COD_mg_L": 8000,
        "NH4_N_mg_L": 500,
        "TP_mg_L": 30
    }

    limits = {
        "COD_mg_L": 600,
        "NH4_N_mg_L": 30,
        "TP_mg_L": 3.2
    }

    history_df = pd.DataFrame()

    result = analyze_wastewater_price(
        water_quality, history_df, limits, 8.93
    )

    assert result["rule_price"] == 148.83
    assert result["max_indicator"] == "NH4_N_mg_L"
def test_historical_matching():
    history_df = pd.DataFrame([
        {
            "case_id": 1,
            "customer_name": "案例A",
            "COD_mg_L": 8000,
            "NH4_N_mg_L": 500,
            "TP_mg_L": 30,
            "initial_quote_yuan_per_t": 150,
            "agreed_price_yuan_per_t": 140,
            "actual_settlement_price_yuan_per_t": 140
        },
        {
            "case_id": 2,
            "customer_name": "案例B",
            "COD_mg_L": 1000,
            "NH4_N_mg_L": 50,
            "TP_mg_L": 5,
            "initial_quote_yuan_per_t": 50,
            "agreed_price_yuan_per_t": 45,
            "actual_settlement_price_yuan_per_t": 45
        }
    ])

    water_quality = {
        "COD_mg_L": 8000,
        "NH4_N_mg_L": 500,
        "TP_mg_L": 30
    }

    limits = {
        "COD_mg_L": 600,
        "NH4_N_mg_L": 30,
        "TP_mg_L": 3.2
    }

    result = analyze_wastewater_price(
        water_quality, history_df, limits, 8.93
    )

    assert result["historical_cases"][0]["case_id"] == 1
    assert result["historical_cases"][0]["distance"] == 0

def test_empty_water_quality():
    with pytest.raises(ValueError):
        analyze_wastewater_price(
                {}, pd.DataFrame(), {}, 8.93
            )

def test_unpriced_indicator():
    water_quality = {"FLUORIDE_mg_L": 20}

    limits = {
        "COD_mg_L": 600,
        "NH4_N_mg_L": 30,
        "TP_mg_L": 3.2
    }

    result = analyze_wastewater_price(
        water_quality,
        pd.DataFrame(),
        limits,
        8.93
    )

    assert result["rule_price"] is None
    assert result["historical_cases"] == []
    assert result["unpriced_indicators"] == ["FLUORIDE_mg_L"]
def test_partial_pricing():
    water_quality = {
        "COD_mg_L": 8000,
        "FLUORIDE_mg_L": 20
    }

    limits = {"COD_mg_L": 600}

    result = analyze_wastewater_price(
        water_quality,
        pd.DataFrame(),
        limits,
        8.93
    )

    assert result["rule_price"] == 119.07
    assert result["max_indicator"] == "COD_mg_L"
    assert result["unpriced_indicators"] == ["FLUORIDE_mg_L"]