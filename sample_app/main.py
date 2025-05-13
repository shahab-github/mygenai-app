from fastapi import FastAPI, Query
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama  # Note: from langchain-community
import os

app = FastAPI()

# Load and split documents
loader = TextLoader("documents/sample.txt")
docs = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
documents = text_splitter.split_documents(docs)

# Create embeddings and FAISS index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(documents, embeddings)
retriever = db.as_retriever()

# Use Ollama with gemma
llm = Ollama(model="gemma:2b")

# Create the RAG chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

@app.get("/ask")
def ask_question(question: str = Query(..., description="Your question")):
    answer = qa.run(question)
    return {"question": question, "answer": answer}
