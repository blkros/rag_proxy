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

    # 1) LLM (vLLM/OpenAI 호환)
    llm = ChatOpenAI(
        base_url=os.environ["OPENAI_BASE_URL"].rstrip("/"),
        api_key=os.environ.get("OPENAI_API_KEY", "local-any"),
        model=os.environ["OPENAI_MODEL"],      # 예: qwen3-30b-a3b-fp8
        temperature=0,
    )

    # 2) 데이터 → Document (CSV가 없어도 지나가도록 완화)
    docs: List[Document] = []
    p = Path(data_path)
    if p.exists():
        docs = _load_docs_from_csv(p)

    # (선택) 추가 청킹
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunked_docs = []
    for d in docs:
        parts = splitter.split_text(d.page_content)
        for c in parts:
            chunked_docs.append(Document(page_content=c, metadata=d.metadata))

    # 3) 임베딩 + 벡터스토어
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")
    emb = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"batch_size": 64, "normalize_embeddings": True}
    )

    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    if _has_faiss_files(index_path):
        # 실제 인덱스 파일이 있을 때만 로드
        vectorstore = FAISS.load_local(index_path, emb, allow_dangerous_deserialization=True)
    elif chunked_docs:
        # 문서가 있으면 새로 만들어 저장
        vectorstore = FAISS.from_documents(chunked_docs, emb)
        vectorstore.save_local(index_path)
    else:
        # 문서도, 인덱스 파일도 없으면 '빈 인덱스'로 시작
        vectorstore = _empty_faiss(emb)
        vectorstore.save_local(index_path)

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 40})

    # 4) 프롬프트
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "너의 사고과정은 절대로 출력하지 말고, 다음 컨텍스트로만 간결하고 정확하게 한국어로 답하라. "
         "모르면 정확히 다음 문장을 출력하라: 주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다\n\n"
         "컨텍스트:\n{context}"),
        ("user", "{question}")
    ])

    # 5) 체인
    rag_chain = (
        {"context": retriever | RunnableLambda(_format_docs),
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever, vectorstore, _drop_think