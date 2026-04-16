from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import uuid
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests

load_dotenv()

# -------------------
# 1. LLM + embeddings
# -------------------
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational"
)
llm = ChatHuggingFace(llm=llm)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """Build a FAISS retriever for the uploaded PDF and store it for the thread."""
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
        return _THREAD_METADATA[str(thread_id)].copy()
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tool call parser
# -------------------
def _extract_json_objects(text: str) -> list:
    """Extract all top-level JSON objects using bracket-matching to handle nesting."""
    objects = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objects.append(json.loads(text[start:i + 1]))
                except json.JSONDecodeError:
                    pass
                start = None
    return objects


def _parse_tool_calls(response: AIMessage) -> AIMessage:
    """
    Llama-3.1 via HuggingFaceEndpoint emits tool calls as raw JSON text rather
    than structured tool_calls. Detect that pattern and convert it into a proper
    AIMessage with tool_calls so tools_condition routes correctly.
    """
    if response.tool_calls:
        return response

    content = response.content if isinstance(response.content, str) else ""
    if not content.strip():
        return response

    tool_calls = []
    for data in _extract_json_objects(content):
        name = data.get("name")
        if not name and data.get("type") == "function":
            name = data.get("function", {}).get("name")
        args = data.get("arguments") or data.get("parameters") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if name and isinstance(args, dict):
            tool_calls.append({"name": name, "args": args, "id": str(uuid.uuid4()), "type": "tool_call"})

    if not tool_calls:
        return response

    return AIMessage(content="", tool_calls=tool_calls)


# -------------------
# 4. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform arithmetic on two numbers. Use this for ANY math or calculation question.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest real-time stock price for a ticker symbol (e.g. 'AAPL', 'TSLA', 'GOOGL').
    Use this tool for ANY question about a stock price or company share value.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()


def _make_rag_tool(thread_id: str):
    """Returns a rag_tool with thread_id baked in so the LLM never needs to pass it."""
    @tool
    def rag_tool(query: str) -> dict:
        """
        Search and retrieve relevant passages from the user's uploaded PDF document.
        Use this tool for ANY question about the content of the uploaded document or PDF.
        """
        retriever = _get_retriever(thread_id)
        if retriever is None:
            return {"error": "No document indexed for this chat. Please upload a PDF first.", "query": query}
        result = retriever.invoke(query)
        return {
            "query": query,
            "context": [doc.page_content for doc in result],
            "metadata": [doc.metadata for doc in result],
            "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
        }
    return rag_tool


# -------------------
# 5. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 6. Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    has_doc = bool(thread_id and str(thread_id) in _THREAD_RETRIEVERS)
    rag_tool = _make_rag_tool(str(thread_id) if thread_id else "")
    all_tools = [search_tool, get_stock_price, calculator, rag_tool]
    llm_with_tools = llm.bind_tools(all_tools)

    doc_instruction = (
        "A PDF IS uploaded. You MUST call `rag_tool` for any question about the document."
        if has_doc
        else "No PDF uploaded yet. Ask the user to upload one if they ask about a document."
    )

    system_message = SystemMessage(content=(
        "You are a helpful assistant with these tools:\n"
        "- `rag_tool`: " + doc_instruction + "\n"
        "- `duckduckgo_search`: Use ONLY for real-time or post-2023 information such as "
        "today's news, live scores, or very recent events. Do NOT use for general knowledge.\n"
        "- `get_stock_price`: Get real-time stock prices. Use for ANY stock price question.\n"
        "- `calculator`: Do math. Use for ANY arithmetic or calculation.\n\n"
        "RULES:\n"
        "1. NEVER answer stock prices from memory — always call `get_stock_price`.\n"
        "2. NEVER do math yourself — always call `calculator`.\n"
        "3. NEVER answer document questions without calling `rag_tool`.\n"
        "4. Answer general knowledge questions (geography, history, science, definitions, etc.) "
        "DIRECTLY from your own knowledge. Do NOT call any tool for these.\n"
        "5. Only call `duckduckgo_search` when the answer requires information from after 2023 "
        "or live/real-time data that is NOT a stock price."
    ))

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    response = _parse_tool_calls(response)
    return {"messages": [response]}


def tool_node_fn(state: ChatState, config=None):
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")
    rag_tool = _make_rag_tool(str(thread_id) if thread_id else "")
    all_tools = [search_tool, get_stock_price, calculator, rag_tool]
    return ToolNode(all_tools).invoke(state, config=config)


# -------------------
# 7. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 8. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node_fn)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 9. Helpers
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})


def delete_thread(thread_id: str) -> None:
    """Remove a thread's retriever, metadata, and checkpointed messages."""
    _THREAD_RETRIEVERS.pop(str(thread_id), None)
    _THREAD_METADATA.pop(str(thread_id), None)
    conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (str(thread_id),))
    conn.execute("DELETE FROM writes WHERE thread_id = ?", (str(thread_id),))
    conn.commit()
