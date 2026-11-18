import streamlit as st
import requests

st.title('工业污水智能业务助手')
mode = st.radio('请选择功能',['知识库问答','智能报价','合同审查'],horizontal=True)
if mode == '合同审查':
    uploaded_file = st.file_uploader(
        '上传待审查的合同',
        type=['docx']
    )
    if st.button('开始审查'):
        if uploaded_file is None:
            st.warning('请先上传docx合同')
        else:
            try:
                with st.spinner('正在审查合同中....'):
                    response = requests.post(
                        "http://127.0.0.1:8000/contract-review",
                        files={'file': (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

                        )},
                        timeout=180
                    )
                response.raise_for_status()
                data = response.json()
                st.success(f'审查完成，共{data["total"]}项')
                for result in data["results"]:
                    st.subheader(
                        f"{result['rule_id']}：{result['status']}"
                    )
                    st.write("合同原文：", result["evidence"])
                    st.write("存在问题：", result["issue"])
                    st.write("修改建议：", result["suggestion"])
            except requests.exceptions.RequestException as e:
                st.error(f'合同审查失败：{e}')











else:
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