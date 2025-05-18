import os 
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

# setting environment variables
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_PROJECT'] = os.getenv('LANGSMITH_PROJECT')
os.environ['LANGSMITH_TRACING'] = 'true'
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')


models = ["gemma2-9b-it", "llama3-8b-8192"]

# prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the following question from the context."),
    ("user", "Question: {question}"),
])

loader = PyPDFLoader("data/RelievingLetter.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = Chroma.from_documents(texts, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

def get_answer(question):
    llm = ChatGroq(model="gemma2-9b-it", temperature=0.1, max_tokens=1000)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question, "context": retriever})
    return answer

# Streamlit app
st.title("Langsmith Chatbot")
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "question" not in st.session_state:
    st.session_state.question = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""
if "model" not in st.session_state:
    st.session_state.model = "gemma2-9b-it"
if "model_options" not in st.session_state:
    st.session_state.model_options = models
if "model_selected" not in st.session_state:
    st.session_state.model_selected = st.session_state.model_options[0]

# Sidebar for model selection
st.sidebar.title("Model Selection")
st.sidebar.selectbox("Select a model", st.session_state.model_options, key="model_selected", on_change=lambda: st.session_state.update(model=st.session_state.model_selected))
# User input
st.session_state.question = st.text_input("Ask a question:", value=st.session_state.question)
# Submit button
if st.button("Submit"):
    st.session_state.answer = get_answer(st.session_state.question)
    st.session_state.messages.append({"role": "user", "content": st.session_state.question})
    st.session_state.messages.append({"role": "assistant", "content": st.session_state.answer})
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# Display answer
with st.chat_message("assistant"):
    st.markdown(st.session_state.answer)
# Footer
st.markdown("---")
st.markdown("Made with ❤️ by [LangSmith](https://langsmith.com)")
