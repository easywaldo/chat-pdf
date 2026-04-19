from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
load_dotenv()

# Loader
loader = PyPDFLoader("unsu.pdf")
pages = loader.load_and_split()

print(f"총 페이지 수: {len(pages)}")
print(f"첫 페이지 내용:\n{pages[0].page_content[:500]}")  # 첫 페이지의 처음 500자 출력

# Splitter
text_splitter = RecursiveCharacterTextSplitter(
    #Set a really small chunk size just to show.
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.split_documents(pages)
print(f"총 텍스트 청크 수: {len(texts)}")
print(texts[0])  # 첫 번째 텍스트 청크 출력

#Embeddings
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    # With the `text-embedding-3-large` class of models, you can specify size of the embeddings you want returned. dimensions=1024
)

# Chroma DB
db = Chroma.from_documents(texts, embeddings_model)

# Retriever
llm = ChatOpenAI(temperature=0)

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(),
    llm=llm,
)

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# Generate response
def format_docs(docs):
    return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

rag_chain = (
    {
        "context": retriever_from_llm | format_docs, "question": RunnablePassthrough()
    } | prompt | llm | StrOutputParser()
)

# Question
result = rag_chain.invoke("아내가 먹고 싶어하는 음식은 무엇인가요?")
print("질문에 대한 답변:")
print(result)