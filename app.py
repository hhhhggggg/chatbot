# app.py
import os
import re
import time
import json
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple


# --- deps from notebook ---
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI


# ----------------------------
# Settings & Secrets handling
# ----------------------------


def get_secret(key: str, default: str = "") -> str:
# 1) Streamlit secrets → 2) env → 3) default
return (
st.secrets.get(key) # type: ignore
if hasattr(st, "secrets") and key in st.secrets # type: ignore
else os.getenv(key, default)
)


DEFAULTS = {
"EMBEDDING_MODEL_NAME": get_secret("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
"LLM_MODEL_NAME": get_secret("LLM_MODEL_NAME", "gpt-4o-mini"),
"PINECONE_INDEX_NAME": get_secret("PINECONE_INDEX_NAME", "YOUR_INDEX_NAME"),
"DEFAULT_VECTOR_WEIGHT": float(get_secret("DEFAULT_VECTOR_WEIGHT", "0.7")),
"DEFAULT_TOP_K": int(get_secret("DEFAULT_TOP_K", "50")),
"DEFAULT_CONTEXT_TOP_N": int(get_secret("DEFAULT_CONTEXT_TOP_N", "6")),
"DEFAULT_CONTEXT_CHARS": int(get_secret("DEFAULT_CONTEXT_CHARS", "2400")),
}


# ----------------------------
# UI — Sidebar
# ----------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 RAG Chatbot (Pinecone + BM25 + OpenAI)")


with st.sidebar:
st.header("⚙️ 설정")
# API Keys (masked input; kept in-memory session only)
if "OPENAI_API_KEY" not in st.session_state:
st.session_state.OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
if "PINECONE_API_KEY" not in st.session_state:
st.session_state.PINECONE_API_KEY = get_secret("PINECONE_API_KEY")


st.session_state.OPENAI_API_KEY = st.text_input(
"OpenAI API Key", value=st.session_state.OPENAI_API_KEY, type="password"
)
st.session_state.PINECONE_API_KEY = st.text_input(
"Pinecone API Key", value=st.session_state.PINECONE_API_KEY, type="password"
)


st.error(f"오류: {e}")
