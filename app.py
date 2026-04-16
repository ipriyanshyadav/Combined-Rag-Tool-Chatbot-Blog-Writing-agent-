import streamlit as st

st.set_page_config(page_title="LangGraph Apps", layout="wide")

blog_page = st.Page("pages/blog_writing_agent.py", title="Blog Writing Agent", icon="✍️")
rag_page  = st.Page("pages/rag_with_tools.py",    title="RAG With Tools",      icon="🤖")

pg = st.navigation([blog_page, rag_page], position="hidden")

col1, col2 = st.columns(2)
with col1:
    if st.button("✍️ Blog Writing Agent", use_container_width=True,
                 type="primary" if pg == blog_page else "secondary"):
        st.switch_page(blog_page)
with col2:
    if st.button("🤖 RAG With Tools", use_container_width=True,
                 type="primary" if pg == rag_page else "secondary"):
        st.switch_page(rag_page)

st.divider()
pg.run()
