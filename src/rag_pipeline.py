# src/rag_pipeline.py
import os, re, csv
from pathlib import Path
from typing import Tuple, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from src.config import settings
from FlagEmbedding import FlagReranker





def _drop_think(text: str) -> str:
    return re.sub(r"<\s*think\b[^>]*>.*?</\s*think\s*>\s*", "", text, flags=re.S | re.I)

def _load_docs_from_csv(path: Path) -> List[Document]:
    docs: List[Document] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            key = (row.get("항목") or "").strip()
            val = (row.get("내용") or "").strip()
            if not key and not val:
                continue
            text = f"{key}: {val}" if key else val
            docs.append(Document(page_content=text, metadata={"source": str(path), "field": key}))
    return docs

def _format_docs(doc_list: List[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in doc_list)

def _has_faiss_files(idx: Path) -> bool:
    return (idx / "index.faiss").exists() and ((idx / "index.pkl").exists() or (idx / "index.json").exists())

def _empty_faiss(emb: HuggingFaceEmbeddings) -> FAISS:
    dim = len(emb.embed_query("dim-probe"))
    index = faiss.IndexFlatL2(dim)
    return FAISS(embedding_function=emb, index=index, docstore=InMemoryDocstore({}), index_to_docstore_id={})

def build_rag_chain(data_path: str = "data/민생.csv",
                    index_dir: str = "faiss_index") -> Tuple:
    load_dotenv(override=True)

    # 1) LLM
    llm = ChatOpenAI(
        base_url=settings.OPENAI_BASE_URL.rstrip("/"),
        api_key=settings.OPENAI_API_KEY,
        model=settings.OPENAI_MODEL,
        temperature=0,
    )

    # 2) 데이터 → Document (CSV가 없어도 지나가도록 완화)
    docs: List[Document] = []
    p = Path(data_path)
    if p.exists():
        docs = _load_docs_from_csv(p)

    # 2-1) 청킹 —  하드코딩 → 설정값 사용
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    chunked_docs = []
    for d in docs:
        parts = splitter.split_text(d.page_content)
        for c in parts:
            chunked_docs.append(Document(page_content=c, metadata=d.metadata))

    # 3) 임베딩
    emb = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": settings.EMBEDDING_DEVICE},
        encode_kwargs={"batch_size": 64, "normalize_embeddings": True}
    )

    # 3-1) 벡터스토어
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    if _has_faiss_files(index_path):
        vectorstore = FAISS.load_local(index_path, emb, allow_dangerous_deserialization=True)
    elif chunked_docs:
        vectorstore = FAISS.from_documents(chunked_docs, emb)
        vectorstore.save_local(index_path)
    else:
        vectorstore = _empty_faiss(emb)
        vectorstore.save_local(index_path)

    # 3-2) 리트리버 —  파라미터화
    retriever = vectorstore.as_retriever(
        search_type=settings.SEARCH_TYPE,
        search_kwargs={"k": settings.RETRIEVER_K, "fetch_k": settings.RETRIEVER_FETCH_K}
    )

    # (옵션) 3-3) 재랭커 준비
    reranker = None
    if settings.ENABLE_RERANKER:
        try:
            reranker = FlagReranker(settings.RERANKER_MODEL, use_fp16=True)
        except Exception as e:
            print(f"[RAG] Reranker init failed: {e}; fallback to no rerank")
            reranker = None

    def _retrieve_and_format(q: str) -> str:
        docs0 = retriever.invoke(q)
        docs1 = docs0
        if reranker:
            pairs = [(q, d.page_content) for d in docs0]
            try:
                scores = reranker.compute_score(pairs, normalize=True)
                ranked = [d for _, d in sorted(zip(scores, docs0), key=lambda x: x[0], reverse=True)]
                docs1 = ranked[: settings.RERANKER_TOP_N]
            except Exception as e:
                print(f"[RAG] Rerank failed: {e}; using raw retriever")
                docs1 = docs0
        return _format_docs(docs1)

    # 4) 프롬프트
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "너의 사고과정은 절대로 출력하지 말고, 다음 컨텍스트로만 간결하고 정확하게 한국어로 답하라. "
         "모르면 정확히 다음 문장을 출력하라: 주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다\n\n"
         "컨텍스트:\n{context}"),
        ("user", "{question}")
    ])

    # 5) 체인 —  컨텍스트 생성 함수를 교체
    rag_chain = (
        {"context": RunnableLambda(_retrieve_and_format),
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever, vectorstore, _drop_think