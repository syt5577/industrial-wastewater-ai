import json

from langchain_core.messages import ToolMessage,HumanMessage,SystemMessage
from openai.resources.beta.threads import messages



def pricing_agent(question: str, llm, pricing_tool) -> str:
    """
    工业污水报价 Agent。

    question: 用户自然语言问题
    llm: 大语言模型
    pricing_tool: 污水报价工具

    返回：中文报价分析文本
    """
    llm_with_tool = llm.bind_tools([pricing_tool])
    messages = [SystemMessage(content= "你是工业污水报价分析助手。"
        "严格依据工具返回的数据回答。"
        "规则计算价格不等于最终商务报价。"
        "不得自行编造最低成交价、折扣比例、"
        "客户议价空间或处理成本。"
        "历史相似案例仅供参考，不代表价格必然合理。"
        "如果数据不足，明确说明缺少哪些依据。"
        "本工具用于现有污水处理业务的制度报价测算，"
        "不要求每次重新开展完整工艺设计或成本核算。"
        "应明确给出工具计算的制度价格及其依据，"
        "不得因缺少处理规模、工艺路线等信息而否定制度报价结果。"
        "对实际接收和商务价格的最终确认，应说明必要的复核条件。"
                              )]
    messages.append(HumanMessage(content= question))
    response = llm_with_tool.invoke(messages)
    messages.append(response)
    if not response.tool_calls:
        return response.content
    tool_call = response.tool_calls[0]
    tool_id = tool_call['id']
    tool_args = tool_call['args']
    result = pricing_tool.invoke(tool_args)
    tool_message = ToolMessage(content=json.dumps(result, ensure_ascii=False),
                               tool_call_id=tool_id)
    messages.append(tool_message)
    llm_result = llm.invoke(messages)
    return llm_result.content



    raise NotImplementedError("Agent流程尚未实现")