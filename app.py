import streamlit as st
import requests

st.title('工业污水智能业务助手')
mode = st.radio('请选择功能',['知识库问答','智能报价'],horizontal=True)
api_url = ("http://127.0.0.1:8000/ask"
    if mode == "知识库问答"
    else "http://127.0.0.1:8000/price")
question = st.text_input('请输出问题')
if st.button('提交问题'):
    if not question.strip():
        st.warning('请输入内容')
    else:
        try:
            with st.spinner(f'正在处理{mode}请求......'):
                response = requests.post(
                url=api_url,
                json={'question': question},
                timeout=60)
            response.raise_for_status()
            data = response.json()
            st.markdown(data['answer'])
        except requests.exceptions.RequestException as e:
            st.error(f'请求失败{e}')