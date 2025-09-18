# RAG Chatbot (Streamlit)


Pinecone + sentence-transformers + BM25 + OpenAI를 사용하는 RAG 챗봇의 Streamlit 버전입니다.


## 빠른 시작


```bash
# 1) 설치
python -m venv .venv && source .venv/bin/activate # Windows는 .venv\Scripts\activate
pip install -r requirements.txt


# 2) secrets 준비
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# 파일 열어서 OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME 채우기


# 3) 실행
streamlit run app.py
