import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os 
from dotenv import load_dotenv

load_dotenv()

# langsmith tracking
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_PROJECT'] = os.getenv('LANGSMITH_PROJECT')
os.environ['LANGSMITH_TRACING'] = 'true'
# google api key
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')


# prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the following question."),
    ("user", "Question: {question}"),
])

def generate_response(question, api_key, llm, temperature, max_tokens):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer


# Title of the app
st.title("Google Generative AI")

# Sidebar for user input
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Google API Key", type="password")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# drop down for model selection
llm = st.sidebar.selectbox("Select Model", ["gemini-2.0-flash", "gemini-1.5-turbo"])

# main interface for the user input
st.write("## Ask a question")
user_input = st.text_input("Enter your question here")

if user_input:
    # Generate response
    response = generate_response(user_input, api_key, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please enter a question to get a response.")