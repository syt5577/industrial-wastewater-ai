from src.pricing import analyze_wastewater_price
import pandas as pd
from src.agent import pricing_agent
print("报价模块导入成功")
history_df = pd.read_excel("data/construct/price_history.xlsx")

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
# result = analyze_wastewater_price(
#     water_quality,
#     history_df,
#     limits,
#     base_cost
# )
# print(result["rule_price"])
# print(result["max_indicator"])
# print(result["historical_cases"])


from src.tools import create_pricing_tool

pricing_tool = create_pricing_tool(history_df,limits,base_cost)

# print(pricing_tool.name)
# print(pricing_tool.args)

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("SILICONFLOW_API_KEY")
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V4.1-Flash",
    base_url="https://api.siliconflow.cn/v1",
    api_key=api_key)
# llm_with_tools = llm.bind_tools([pricing_tool])
# response = llm_with_tools.invoke(
#     "请帮我报价：COD 8000mg/L，氨氮500mg/L，总磷30mg/L。请调用报价工具。"
# )
# print(response.tool_calls)
#
# tool_calls = response.tool_calls[0]
# result = pricing_tool.invoke(tool_calls['args'])
# print(result)


import json
from langchain_core.messages import HumanMessage, ToolMessage



answer = pricing_agent(
    "氟化物 FLUORIDE_mg_L 为20mg/L，帮我报价。",
    llm,
    pricing_tool
)
print(answer)