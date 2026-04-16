from __future__ import annotations

import uuid
from typing import Dict

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langraph_rag_backend import chatbot, ingest_pdf, retrieve_all_threads, delete_thread, thread_document_metadata

# ── Session state ──
if "rag_message_history" not in st.session_state:
    st.session_state["rag_message_history"] = []
if "rag_thread_id" not in st.session_state:
    st.session_state["rag_thread_id"] = uuid.uuid4()
if "rag_chat_threads" not in st.session_state:
    st.session_state["rag_chat_threads"] = retrieve_all_threads()
if "rag_ingested_docs" not in st.session_state:
    st.session_state["rag_ingested_docs"] = {}

rag_thread_key = str(st.session_state["rag_thread_id"])
if st.session_state["rag_thread_id"] not in st.session_state["rag_chat_threads"]:
    st.session_state["rag_chat_threads"].append(st.session_state["rag_thread_id"])
rag_thread_docs = st.session_state["rag_ingested_docs"].setdefault(rag_thread_key, {})


def rag_reset_chat():
    new_id = uuid.uuid4()
    st.session_state["rag_thread_id"] = new_id
    st.session_state["rag_chat_threads"].append(new_id)
    st.session_state["rag_message_history"] = []


def _is_tool_call_json(content: str) -> bool:
    """Return True if the content is a raw JSON tool call that should be hidden."""
    stripped = content.strip()
    return stripped.startswith("{") and '"name"' in stripped and '"arguments"' in stripped


selected_thread = None

# ── Left sidebar ──
with st.sidebar:
    st.markdown("### 🗂️ PDF Chatbot")
    st.markdown(f"**Thread:** `{rag_thread_key[:8]}…`")

    if st.button("➕ New Chat", use_container_width=True, key="rag_new_chat"):
        rag_reset_chat()
        st.rerun()

    if rag_thread_docs:
        latest_doc = list(rag_thread_docs.values())[-1]
        st.success(f"`{latest_doc.get('filename')}` — {latest_doc.get('chunks')} chunks")
    else:
        st.info("No PDF indexed yet.")

    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"], key="rag_pdf")
    if uploaded_pdf:
        if uploaded_pdf.name in rag_thread_docs:
            st.info(f"`{uploaded_pdf.name}` already processed.")
        else:
            with st.status("Indexing…", expanded=True) as sb:
                summary = ingest_pdf(uploaded_pdf.getvalue(), thread_id=rag_thread_key, filename=uploaded_pdf.name)
                rag_thread_docs[uploaded_pdf.name] = summary
                sb.update(label="✅ Indexed", state="complete", expanded=False)

    st.markdown("---")
    st.markdown("**Past conversations**")
    rag_threads = st.session_state["rag_chat_threads"][::-1]
    if not rag_threads:
        st.caption("No past conversations yet.")
    else:
        for tid in rag_threads:
            label = str(tid)[:8] + "…"
            col_t, col_d = st.columns([3, 1])
            with col_t:
                if st.button(label, key=f"rag-thread-{tid}", use_container_width=True):
                    selected_thread = tid
            with col_d:
                if st.button("🗑️", key=f"rag-del-{tid}", use_container_width=True):
                    delete_thread(str(tid))
                    st.session_state["rag_chat_threads"].remove(tid)
                    st.session_state["rag_ingested_docs"].pop(str(tid), None)
                    if str(tid) == rag_thread_key:
                        rag_reset_chat()
                    st.rerun()

# ── Main chat area ──
st.title("Multi Utility Chatbot")
st.subheader("This chatbot can:\n1. Chat with your PDFs\n2. Search the web\n3. Calculate math problems\n4. Get stock prices")

for message in st.session_state["rag_message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Ask about your document or use tools", key="rag_chat_input")

if user_input:
    st.session_state["rag_message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": rag_thread_key},
        "metadata": {"thread_id": rag_thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder: Dict = {"box": None}

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(f"🔧 Using `{tool_name}` …", expanded=True)
                    else:
                        status_holder["box"].update(label=f"🔧 Using `{tool_name}` …", state="running", expanded=True)
                if isinstance(message_chunk, AIMessage):
                    content = message_chunk.content
                    if not content or _is_tool_call_json(content):
                        continue
                    yield content

        ai_message = st.write_stream(ai_only_stream())
        if status_holder["box"] is not None:
            status_holder["box"].update(label="✅ Tool finished", state="complete", expanded=False)

    st.session_state["rag_message_history"].append({"role": "assistant", "content": ai_message})

    doc_meta = thread_document_metadata(rag_thread_key)
    if doc_meta:
        st.caption(f"Document: {doc_meta.get('filename')} — chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')}")

if selected_thread:
    st.session_state["rag_thread_id"] = selected_thread
    messages = chatbot.get_state(config={"configurable": {"thread_id": selected_thread}}).values.get("messages", [])
    st.session_state["rag_message_history"] = [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in messages
    ]
    st.session_state["rag_ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()
