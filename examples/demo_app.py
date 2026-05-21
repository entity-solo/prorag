"""
Streamlit demo - upload documents, ask questions, and inspect entity-graph stats.

Run:
    GROQ_API_KEY=your_key streamlit run examples/demo_app.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

from prorag import ProRAG


st.set_page_config(page_title="ProRAG Demo", page_icon=":brain:", layout="wide")
st.title("ProRAG - Entity Graph RAG")
st.caption("Upload documents -> ask questions -> retrieve grounded evidence from the graph")

if "rag" not in st.session_state:
    st.session_state.rag = ProRAG()
if "history" not in st.session_state:
    st.session_state.history = []

rag: ProRAG = st.session_state.rag

with st.sidebar:
    st.header("Knowledge Base")

    uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded:
        content = uploaded.read().decode("utf-8")
        with st.spinner(f"Extracting knowledge from {uploaded.name}..."):
            n = rag.ingest(content, source=uploaded.name)
        st.success(f"{n} triples extracted from {uploaded.name}")

    st.divider()
    st.subheader("Or paste text directly")
    paste = st.text_area("Paste knowledge here", height=150)
    if st.button("Ingest text") and paste.strip():
        with st.spinner("Extracting..."):
            n = rag.ingest(paste, source="pasted_text")
        st.success(f"{n} triples extracted")

    st.divider()
    stats = rag.stats()
    st.metric("Nodes", stats["nodes"])
    st.metric("Edges", stats["edges"])

    if st.button("Clear graph"):
        st.session_state.rag = ProRAG()
        st.session_state.history = []
        st.rerun()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Ask a question")
    question = st.text_input("Question", placeholder="What do you want to know?")

    if st.button("Ask", type="primary") and question.strip():
        if stats["nodes"] == 0:
            st.warning("Graph is empty - please ingest some documents first.")
        else:
            with st.spinner("Querying graph..."):
                result = rag.ask(question)
            st.session_state.history.append({"q": question, "r": result})

    for item in reversed(st.session_state.history):
        q, r = item["q"], item["r"]
        with st.container(border=True):
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {r['answer']}")
            cols = st.columns(2)
            cols[0].caption(f"Triples used: {r['triples_used']}")
            if r["sources"]:
                cols[1].caption(f"Sources: {', '.join(r['sources'])}")
            if r["has_contradictions"]:
                st.warning("Conflicting information exists in the graph for this topic.")

with col2:
    st.subheader("Graph explorer")
    if stats["nodes"] > 0:
        keyword = st.text_input("Search nodes", placeholder="e.g. Einstein")
        if keyword:
            triples = rag.graph.query([keyword], top_k=20)
            for t in triples:
                neg = "NOT " if t["negated"] else ""
                cond = f" [{t['condition']}]" if t["condition"] else ""
                conf = f" ({t['confidence']:.1f})" if t["confidence"] < 0.9 else ""
                st.markdown(f"- `{t['subject']}` -> **{neg}{t['relation']}** -> `{t['object']}`{cond}{conf}")
    else:
        st.info("Ingest documents to explore the graph.")
