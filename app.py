import streamlit as st
import requests

st.title('工业污水智能业务助手')
question = st.text_input('请输出问题')
if st.button('提交问题'):
    if not question.strip():
        st.warning('请输入内容')
    else:
        try:
            with st.spinner('正在检索知识库'):
                response = requests.post(
                'http://127.0.0.1:8000/ask',
                json={'question': question},
                timeout=60)
            response.raise_for_status()
            data = response.json()
            st.markdown(data['answer'])
        except requests.exceptions.RequestException as e:
            st.error(f'请求失败{e}')