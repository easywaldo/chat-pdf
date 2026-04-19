import tempfile
import os

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
load_dotenv()


def build_rag_chain(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load_and_split()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        texts = splitter.split_documents(pages)

        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
        db = FAISS.from_documents(texts, embeddings_model)

        llm = ChatOpenAI(temperature=0)
        retriever = MultiQueryRetriever.from_llm(
            retriever=db.as_retriever(),
            llm=llm,
        )

        prompt = hub.pull("rlm/rag-prompt")

        def format_docs(docs):
            return "\n\n".join(
                [f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
            )

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain
    finally:
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

    if "rag_chain" not in st.session_state or st.session_state.get("last_file") != uploaded_file.name:
        with st.spinner("PDF를 분석하는 중..."):
            st.session_state.rag_chain = build_rag_chain(uploaded_file)
            st.session_state.last_file = uploaded_file.name

    # ── 질문 영역 ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("PDF에게 질문해보세요!!")

    question = st.text_input("질문을 입력하세요.", placeholder="예: 이 문서의 핵심 내용은 무엇인가요?")

    ask = st.button("질문하기", type="primary", use_container_width=True)

    # ── 답변 출력 영역 ─────────────────────────────────────────────────────────
    answer_box = st.empty()

    if ask:
        if not question.strip():
            st.warning("질문을 입력해주세요.")
        else:
            with answer_box.container():
                with st.spinner("답변을 생성하는 중..."):
                    answer = st.session_state.rag_chain.invoke(question)
                st.markdown("**답변**")
                st.write(answer)
