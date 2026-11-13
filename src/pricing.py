import pandas as pd 
def analyze_wastewater_price(
    water_quality: dict,
    history_df: pd.DataFrame,
    limits: dict,
    base_cost: float
) -> dict:
    if not water_quality:
        raise ValueError("缺水质数据，无法进行报价")
    if any(value < 0 for value in water_quality.values()):
        raise ValueError("水质数据中存在负值，无法进行报价" )
    pricing_indicator = [indicator for indicator in water_quality if indicator in limits]
    unpriced_indicators = [indicator for indicator in water_quality if indicator not in limits] 
    ratios = {}
    for indicator in pricing_indicator:
        ratios[indicator] = water_quality[indicator] / limits[indicator]
    if ratios:
        max_indicator = max(ratios,key=ratios.get)
        max_ratio = ratios[max_indicator]
        rule_price =round(base_cost * max(max_ratio,1.2),2)
    else:
        max_indicator = None
        max_ratio = None
        rule_price = None
    features = [feature for feature in water_quality if feature in history_df.columns and feature in limits]
    if features:
        diff_df = pd.DataFrame(index=history_df.index)
        for column in features:
            diff_column = column.replace('_mg_L','_diff')
            diff_df[diff_column] = (history_df[column] - water_quality[column]).abs()/max(water_quality[column],1e-6)
        diff_df['distance'] = diff_df.mean(axis=1) 
        top3 = (diff_df.sort_values('distance').head(3))
        price_columns = [
        'case_id',
        'customer_name',
        'initial_quote_yuan_per_t',
        'agreed_price_yuan_per_t',
        'actual_settlement_price_yuan_per_t'
            ]
        historical_cases = history_df.loc[top3.index, price_columns].copy()
        historical_cases['distance'] = top3['distance']
        historical_cases = historical_cases.astype(object).where(pd.notna(historical_cases), None)
        historical_cases = historical_cases.to_dict(orient = 'records')
    else:
        historical_cases = []
    return {
    'input_water_quality': water_quality,
    'ratios': ratios,
    'max_indicator': max_indicator,
    'max_ratio': max_ratio,
    'rule_price': rule_price,
    'historical_cases': historical_cases,
    'unpriced_indicators': unpriced_indicators
}

                
    


    