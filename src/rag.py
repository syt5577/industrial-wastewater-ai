def build_context(rrf_results, top_k=3):
    context_parts = []

    # 1. 取 RRF 排名前 top_k 的结果

    # 2. 遍历结果，获取 Document

    # 3. 提取 metadata 中的 chunk_id

    # 4. 将 chunk_id 和 page_content 拼成一段文本

    # 5. 用 "\n\n---\n\n" 连接所有文本
    top_doc = rrf_results[:top_k]
    for item in top_doc:
        doc = item['document']
        chunk_id = doc.metadata['chunk_id']
        content = doc.page_content
        context_parts.append(
            f'来源：{chunk_id}\n内容：{content}'
        )
    context = '\n\n----------\n\n'.join(context_parts)
    return context
def generate_answer(query, context, llm):
    # 1. 构建 Prompt
    # 要求 LLM：
    # - 仅依据 context 回答
    # - 如果资料不足，明确说明
    # - 引用来源 chunk_id

    # 2. 调用 LLM
    # response = llm.invoke(...)

    # 3. 提取回答文本
    # return response.content
    prompt = f"""
    你是一名工业污水处理业务助手。

    请严格依据以下参考资料回答用户问题。
    如果参考资料不足以回答，请明确说明，不要编造。
    回答时引用对应的来源编号（chunk_id）。

    参考资料：
    {context}

    用户问题：
    {query}
    """
    response = llm.invoke(prompt)
    return response.content
