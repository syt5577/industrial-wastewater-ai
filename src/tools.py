from langchain_core.tools import tool
from src.pricing import analyze_wastewater_price
from src.schemas import PricingInput
def create_pricing_tool(history_df,limits,base_cost)->dict:
    @tool(args_schema=PricingInput)
    def wastewater_price_tool(water_quality):
        '''
        工业污水报价工具。
        根据用户提供的水质指标，计算制度报价并匹配历史相似案例。
        指标包括 COD、氨氮、总磷等，浓度单位为 mg/L。
        只提取用户明确提供的指标，不得编造数据。
        '''
        return analyze_wastewater_price(water_quality,history_df,limits,base_cost)
    return wastewater_price_tool
