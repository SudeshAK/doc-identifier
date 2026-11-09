import logging
import os

# 1. Silence the false alarm Hugging Face warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_docling import DoclingLoader

loader = DoclingLoader("pdfs/sample.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = text_splitter.split_documents(docs)

for chunk in chunks:
    for key, value in list(chunk.metadata.items()):
        if isinstance(value, (dict, list)):
            chunk.metadata[key] = str(value)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings,
    persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
llm = ChatOllama(model="llama3.2:latest", temperature=0)

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        # Extracts just the string text from the dictionary for the retriever
        "context": (lambda x: x["question"]) | retriever | format_docs, 
        # Passes the whole original dictionary down so the prompt can access {question}
        "question": RunnablePassthrough() 
    }
    | prompt
    | llm
    | StrOutputParser()
)

question = "What is the main topic of the document?"
answer = rag_chain.invoke({"question": question})
print("Answer:", answer)