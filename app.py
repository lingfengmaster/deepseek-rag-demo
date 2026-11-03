"""Streamlit 界面：调用 FastAPI 接口，提供带出处的问答交互。

启动：
    pip install streamlit
    streamlit run app.py
"""

import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="DeepSeek 文档问答", page_icon="📚")
st.title("📚 DeepSeek 文档问答")
st.caption("基于 RAG 的技术文档问答 · 答案附带原文出处")

if "history" not in st.session_state:
    st.session_state.history = []


def render_sources(sources):
    with st.expander(f"查看 {len(sources)} 段原文"):
        for i, s in enumerate(sources, 1):
            st.markdown(f"**[{i}]** {s}")


# ── 侧边栏：服务状态 ──
with st.sidebar:
    st.subheader("服务状态")
    try:
        if requests.get(f"{API}/health", timeout=3).json().get("status") == "ok":
            st.success("接口正常")
        else:
            st.warning("接口返回异常")
    except Exception:
        st.error("连不上接口\n\n先在另一个窗口启动：\n`uvicorn api:app --port 8000`")

    st.divider()
    if st.button("清空对话", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ── 历史消息 ──
for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["q"])
    with st.chat_message("assistant"):
        st.write(item["a"])
        render_sources(item["sources"])

# ── 输入框 ──
question = st.chat_input("问点什么，比如：deepseek-flash 的并发限制是多少？")
if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("检索中…"):
            try:
                resp = requests.post(
                    f"{API}/ask", json={"question": question}, timeout=120
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                st.error(f"请求失败：{e}")
                st.stop()

        st.write(data["answer"])
        render_sources(data["sources"])

    st.session_state.history.append(
        {"q": question, "a": data["answer"], "sources": data["sources"]}
    )
