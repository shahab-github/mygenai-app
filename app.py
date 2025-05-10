import os 
from dotenv import load_dotenv

from langchain_community.llms import Ollama 
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

# Langsmith Tracking 
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

## Prompt Template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please answer the user's questions."),
    ("user", "{input}")
])


# Streamlit App
st.title("Langchain demo with Gemma LLM")
st.write("This is a simple demo of using Langchain with the Ollama LLM.")

# User input
user_input = st.text_input("Enter your question:")

## Ollama LLM
llm = Ollama(model="gemma:2b")
output_parser = StrOutputParser()

chain = prompt_template | llm | output_parser
if user_input:
    with st.spinner("Generating response..."):
        # Call the chain with the user input
        response = chain.invoke({"input": user_input})
        st.write("Response:")
        st.write(response)