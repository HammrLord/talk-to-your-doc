import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_utils import load_chunks, vector_store, create_chain
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage


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

if st.button("Reset Session",help="Reset this conversation.",type="primary"):
    st.session_state.clear()
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    st.rerun()

if "memory" not  in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )



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
        st.session_state.conversation_chain = create_chain(st.session_state.vectorstores, memory=st.session_state.memory)

if "memory" in st.session_state:
    for msg in st.session_state.memory.chat_memory.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)



user_input = st.chat_input("Ask any questions relevant to uploaded pdf")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        response = st.session_state.conversation_chain({"question": user_input})
        assistant_response = response["answer"]
        st.write_stream(stream_data(assistant_response))

if __name__ =="__main__":
    load_dotenv()
