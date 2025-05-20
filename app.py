import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_utils import load_chunks, vector_store, create_chain


HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
GROQ_TOKEN = st.secrets["GROQ_API_KEY"]

def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.05)

st.set_page_config(
    page_title="💭TTYD",
    page_icon="📑",
    layout="wide"
)
st.title("📝 Talk to Your Doc")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("Reset Session",help="Reset this conversation.",type="primary"):
    st.session_state.clear()
    st.rerun()


uploaded_files = st.file_uploader(label="Upload your (PDF, TXT, DOCX, CSV)",
                                 type=["pdf","txt","docx","csv"],
                                 accept_multiple_files=True)
working_dir = os.getcwd()
if uploaded_files:
    os.makedirs("uploaded", exist_ok=True)
    vector_store_path = f"{working_dir}/vector_store"
    all_chunks = load_chunks(uploaded_files)

    if "vectorstores" not in st.session_state:
        st.session_state.vectorstores = vector_store(all_chunks,vector_store_path)

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = create_chain(st.session_state.vectorstores)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



user_input = st.chat_input("Ask any questions relevant to uploaded pdf")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.conversation_chain({"question": user_input})
        assistant_response = response["answer"]
        st.write_stream(stream_data(assistant_response))
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

if __name__ =="__main__":
    load_dotenv()
