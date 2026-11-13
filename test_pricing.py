from src.pricing import analyze_wastewater_price
import pandas as pd

print("报价模块导入成功")
history_df = pd.read_excel("data/structured/price_history.xlsx")

limits = {
    "COD_mg_L": 600,
    "NH4_N_mg_L": 30,
    "TP_mg_L": 3.2
}

water_quality = {
    "COD_mg_L": 8000,
    "NH4_N_mg_L": 500,
    "TP_mg_L": 30
}

base_cost = 8.93
analyze_wastewater_price(
    water_quality,
    history_df,
    limits,
    base_cost
)