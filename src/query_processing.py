from  dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
load_dotenv()
api_key = os.getenv('SILICONFLOW_API_KEY')
def rewrite_query(query, llm):
    prompt = (
    f"请将用户问题改写为更适合企业知识库检索的查询。\n"
    f"要求：\n"
    f"1. 保留原始问题的真实意图，不回答问题。\n"
    f"2. 不只是把语言变正式，而要尽量转换为企业制度、SOP、业务手册中可能出现的标准术语。\n"
    f"3. 优先使用简洁、明确的业务概念，例如“接收前评估”“可行性评估”“合同审批”“资质审核”“水质异常处理”等类型的表达。\n"
    f"4. 不添加用户问题中不存在的具体事实。\n"
    f"5. 只输出一个改写后的检索问题。\n"
    f"原问题：{query}"
)
    response = llm.invoke(prompt)
    rewritten_query = response.content.strip()
    return rewritten_query