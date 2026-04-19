# Streamlit Cloud의 sqlite3 버전 우회 (chromadb requires >= 3.35.0)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import tempfile
import os
from threading import Thread

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks.base import BaseCallbackHandler

from dotenv import load_dotenv
load_dotenv()


class StreamHandler(BaseCallbackHandler):
    """
    LLM이 토큰을 생성할 때마다 Streamlit UI에 실시간으로 출력하는 핸들러.

    LangChain의 BaseCallbackHandler를 상속받아 on_llm_new_token 이벤트를
    가로챈 후, st.empty() 컨테이너에 누적된 텍스트를 점진적으로 렌더링한다.
    """

    def __init__(self, container: st.delta_generator.DeltaGenerator):
        """
        :param container: 토큰을 스트리밍할 Streamlit empty 컨테이너
        """
        self.container = container
        self.text = ""  # 지금까지 누적된 전체 답변 텍스트

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """
        LLM이 새 토큰을 생성할 때마다 호출되는 콜백.

        토큰을 누적 버퍼에 추가하고 컨테이너를 즉시 업데이트하여
        사용자가 타이핑되는 것처럼 답변을 볼 수 있게 한다.

        :param token: 새로 생성된 단일 토큰 문자열
        """
        self.text += token
        self.container.markdown(self.text)


def build_rag_chain(pdf_file):
    """
    업로드된 PDF 파일로부터 RAG(Retrieval-Augmented Generation) 체인을 구성한다.

    처리 순서:
      1. PDF → 페이지 단위 로드
      2. RecursiveCharacterTextSplitter → 청크 분할
      3. OpenAIEmbeddings → 벡터 임베딩
      4. ChromaDB → 벡터 저장
      5. MultiQueryRetriever → 다중 쿼리 기반 검색
      6. RAG 프롬프트 + LLM + StrOutputParser → 최종 체인 반환

    :param pdf_file: Streamlit의 UploadedFile 객체
    :return: LangChain LCEL 체인 (question 문자열을 입력받아 답변 문자열 반환)
    """
    # 업로드된 파일을 임시 경로에 저장 (PyPDFLoader는 파일 경로 필요)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    try:
        # 1. PDF 로드
        loader = PyPDFLoader(tmp_path)
        pages = loader.load_and_split()

        # 2. 텍스트 청크 분할
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        texts = splitter.split_documents(pages)

        # 3. 임베딩 모델 초기화
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

        # 4. ChromaDB 인메모리 벡터 저장소 생성
        db = Chroma.from_documents(texts, embeddings_model)

        # 5. MultiQueryRetriever: 하나의 질문을 여러 각도의 쿼리로 변환해 검색 품질 향상
        llm_for_retriever = ChatOpenAI(temperature=0)
        retriever = MultiQueryRetriever.from_llm(
            retriever=db.as_retriever(),
            llm=llm_for_retriever,
        )

        # LangChain Hub에서 표준 RAG 프롬프트 로드
        prompt = hub.pull("rlm/rag-prompt")

        def format_docs(docs):
            """검색된 문서 목록을 하나의 컨텍스트 문자열로 합친다."""
            return "\n\n".join(
                [f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
            )

        # 6. LCEL 체인 조립 (streaming=True는 ChatOpenAI 생성 시 설정)
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | ChatOpenAI(temperature=0, streaming=True)  # 스트리밍 활성화
            | StrOutputParser()
        )
        return chain
    finally:
        # 임시 파일 정리
        os.unlink(tmp_path)


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Chat PDF", page_icon="📄", layout="centered")
st.title("📄 Chat PDF")

# ── PDF 업로드 영역 ────────────────────────────────────────────────────────────
st.subheader("PDF 업로드")
uploaded_file = st.file_uploader(
    "PDF 파일을 드래그하거나 클릭하여 선택하세요.",
    type=["pdf"],
    label_visibility="collapsed",
)

if uploaded_file:
    st.success(f"{uploaded_file.name} 업로드 완료")

    # 파일이 바뀐 경우에만 RAG 체인을 새로 빌드 (불필요한 재처리 방지)
    if "rag_chain" not in st.session_state or st.session_state.get("last_file") != uploaded_file.name:
        with st.spinner("PDF를 분석하는 중..."):
            st.session_state.rag_chain = build_rag_chain(uploaded_file)
            st.session_state.last_file = uploaded_file.name

    # ── 질문 영역 ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("PDF에게 질문해보세요!!")

    question = st.text_input("질문을 입력하세요.", placeholder="예: 이 문서의 핵심 내용은 무엇인가요?")

    ask = st.button("질문하기", type="primary", use_container_width=True)

    # ── 답변 출력 영역 (스트리밍) ──────────────────────────────────────────────
    if ask:
        if not question.strip():
            st.warning("질문을 입력해주세요.")
        else:
            st.markdown("**답변**")
            # 토큰이 스트리밍될 빈 컨테이너 생성
            answer_container = st.empty()

            # StreamHandler에 컨테이너를 전달 → 토큰 생성마다 실시간 렌더링
            handler = StreamHandler(answer_container)

            # chain.invoke 대신 callbacks 파라미터로 핸들러 주입
            st.session_state.rag_chain.invoke(
                question,
                config={"callbacks": [handler]},
            )
