import os 
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM

load_dotenv()

# setting environment variables
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_PROJECT'] = os.getenv('LANGSMITH_PROJECT')
os.environ['LANGSMITH_TRACING'] = 'true'


# prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the following question."),
    ("user", "Question: {question}"),
])

def generate_response(question, llm, temperature, max_tokens):
    llm = OllamaLLM(model="gemma:2b")
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer

# Title of the app
st.title("Ollama Generative AI")
# Sidebar for user input
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Ollama API Key", type="password")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)
# drop down for model selection
llm = st.sidebar.selectbox("Select Model", ["gemma:2b", "gemini-1.5-turbo"])
# main interface for the user input
st.write("## Ask a question")
user_input = st.text_input("Enter your question here")
if user_input:
    # Generate response
    response = generate_response(user_input, llm, temperature, max_tokens)
    st.write("### Answer:")
    st.write(response)
