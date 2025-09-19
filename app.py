# app.py
import os
import re
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
    # 오직 Streamlit secrets.toml 만 사용
    return st.secrets.get(key, default)  # type: ignore

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

    # --- 키 로드(화면에 입력칸 노출 없음) ---
    openai_secret = get_secret("OPENAI_API_KEY")
    pinecone_secret = get_secret("PINECONE_API_KEY")

    # 세션에 주입
    st.session_state.OPENAI_API_KEY = openai_secret
    st.session_state.PINECONE_API_KEY = pinecone_secret

    # 상태만 표시 (값은 출력하지 않음)
    if openai_secret:
        st.markdown("✅ **OpenAI API Key**: 설정됨")
    else:
        st.error("❌ OpenAI API Key가 없습니다. `.streamlit/secrets.toml` 또는 환경변수로 설정하세요.")

    if pinecone_secret:
        st.markdown("✅ **Pinecone API Key**: 설정됨")
    else:
        st.error("❌ Pinecone API Key가 없습니다. `.streamlit/secrets.toml` 또는 환경변수로 설정하세요.")

    # base_url도 secrets.toml 에서만 가져옴
    base_url = get_secret("OPENAI_BASE_URL", "")

    st.divider()
    # 비민감 설정만 노출
    EMBEDDING_MODEL_NAME = st.text_input("Embedding 모델", value=DEFAULTS["EMBEDDING_MODEL_NAME"])
    LLM_MODEL_NAME = st.text_input("LLM 모델", value=DEFAULTS["LLM_MODEL_NAME"])
    PINECONE_INDEX_NAME = st.text_input("Pinecone 인덱스", value=DEFAULTS["PINECONE_INDEX_NAME"])

    st.subheader("RAG 파라미터")
    vec_w = st.slider("벡터 가중치", 0.0, 1.0, float(DEFAULTS["DEFAULT_VECTOR_WEIGHT"]))
    bm25_w = 1.0 - vec_w
    top_k = st.number_input("Vector TopK", 1, 200, int(DEFAULTS["DEFAULT_TOP_K"]))
    ctx_n = st.number_input("Context TopN", 1, 20, int(DEFAULTS["DEFAULT_CONTEXT_TOP_N"]))
    max_ctx_chars = st.number_input("Context 길이(문자)", 200, 8000, int(DEFAULTS["DEFAULT_CONTEXT_CHARS"]))

# 키가 없으면 실행 중단 (민감정보 입력창 노출 방지)
if not st.session_state.get("OPENAI_API_KEY") or not st.session_state.get("PINECONE_API_KEY"):
    st.stop()

# ----------------------------
# Caches
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_embedder(name: str):
    return SentenceTransformer(name, device="cpu")

@st.cache_resource(show_spinner=True)
def init_pinecone(_api_key: str):
    """언더스코어로 시작하는 인자는 Streamlit이 해시하지 않음"""
    if not _api_key:
        raise ValueError("Pinecone API 키가 필요합니다.")
    return Pinecone(api_key=_api_key)

@st.cache_resource(show_spinner=False)
def get_index(_pc: Pinecone, index_name: str):
    """언더스코어로 시작하는 인자는 Streamlit이 해시하지 않음"""
    return _pc.Index(index_name)

# ----------------------------
# RAG Core
# ----------------------------
def simple_tokenize(s: str):
    return re.findall(r"[A-Za-z0-9가-힣]+", (s or "").lower())

def vector_search(index, embedder, query: str, top_k: int = 50, meta_filter=None):
    q_vec = embedder.encode([f"query: {query}"], convert_to_numpy=True, normalize_embeddings=True)[0]
    kwargs = {
        "vector": q_vec.tolist(),
        "top_k": int(top_k),
        "include_values": False,
        "include_metadata": True,
    }
    if meta_filter:
        kwargs["filter"] = meta_filter
    res = index.query(**kwargs)
    candidates = []
    for match in res.get("matches", []):
        cid = match.get("id")
        score = float(match.get("score") or 0.0)
        meta = match.get("metadata") or {}
        candidates.append((cid, score, meta))
    return candidates

def bm25_rescore(query: str, candidates: List[Tuple[str, float, Dict]]):
    ids, docs = [], []
    for cid, _, meta in candidates:
        text = (meta or {}).get("text_content") or ""
        if not text:
            title = (meta or {}).get("title") or ""
            keywords = (meta or {}).get("keywords") or ""
            text = f"{title}\n{keywords}"
        ids.append(cid)
        docs.append(simple_tokenize(text))
    if not docs:
        return {}
    bm25 = BM25Okapi(docs)
    scores = bm25.get_scores(simple_tokenize(query)) if query else np.zeros(len(ids))
    max_b = float(np.max(scores)) if len(scores) else 0.0
    return {ids[i]: (float(scores[i]) / max_b if max_b > 0 else 0.0) for i in range(len(ids))}

def build_context(query: str, candidates: List[Tuple[str, float, Dict]], vec_w: float, bm25_w: float, top_n: int, max_chars: int):
    bm25_scores = bm25_rescore(query, candidates)
    scored = []
    for cid, v_score, meta in candidates:
        b_score = bm25_scores.get(cid, 0.0)
        combo = vec_w * float(v_score) + bm25_w * float(b_score)
        scored.append((combo, cid, meta))
    scored.sort(reverse=True, key=lambda x: x[0])

    picked, used = [], 0
    for _, cid, meta in scored[: max(1, int(top_n) * 3)]:
        text = (meta or {}).get("text_content") or (meta or {}).get("title") or ""
        if not text:
            continue
        if used + len(text) > max_chars:
            continue
        picked.append({
            "id": cid,
            "title": (meta or {}).get("title"),
            "source": (meta or {}).get("source"),
            "url": (meta or {}).get("url"),
            "chunk": text,
        })
        used += len(text)
        if len(picked) >= int(top_n):
            break
    return picked

# ----------------------------
# LLM Call
# ----------------------------
def call_openai(api_key: str, model: str, messages: List[Dict], base_url: str = "") -> str:
    if not api_key:
        raise ValueError("OpenAI API 키가 필요합니다.")
    client = OpenAI(api_key=api_key, base_url=base_url or None)
    r = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    return r.choices[0].message.content or ""

SYSTEM_PROMPT = (
    "너는 RAG 기반 도우미야. 제공된 컨텍스트를 우선 활용해서 간결하고 정확하게 답해.\n"
    "근거가 없으면 솔직히 모른다고 말해.\n"
    "출처를 bullet로 함께 제공해."
)

# ----------------------------
# Chat UI
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("질문을 입력하세요…")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("검색 및 생성 중…"):
            try:
                embedder = load_embedder(EMBEDDING_MODEL_NAME)
                pc = init_pinecone(st.session_state.OPENAI_API_KEY and st.session_state.PINECONE_API_KEY)  # 키 존재 보장됨
                index = get_index(pc, PINECONE_INDEX_NAME)

                candidates = vector_search(index, embedder, user_input, top_k=top_k)
                contexts = build_context(user_input, candidates, vec_w, bm25_w, ctx_n, max_ctx_chars)

                context_text = "\n\n".join(
                    [f"[#{i+1}] {c['chunk']}" for i, c in enumerate(contexts)]
                )
                citations = "\n".join(
                    [
                        f"- [#{i+1}] {c.get('title') or c.get('source') or c.get('url') or c['id']}"
                        for i, c in enumerate(contexts)
                    ]
                )

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"질문: {user_input}\n\n컨텍스트:\n{context_text}\n\n"
                            "컨텍스트를 기반으로 답하세요. 답 끝에 '출처' 섹션을 넣어 아래 목록에서 근거를 인용하세요.\n"
                            f"출처 목록:\n{citations}"
                        ),
                    },
                ]

                answer = call_openai(
                    st.session_state.OPENAI_API_KEY,
                    LLM_MODEL_NAME,
                    messages,
                    base_url=base_url,
                )

                final = answer.strip()
                st.markdown(final)
                st.session_state.messages.append({"role": "assistant", "content": final})
            except Exception as e:
                st.error(f"오류: {e}")
