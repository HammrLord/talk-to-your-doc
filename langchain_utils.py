import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader,TextLoader, Docx2txtLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv()

def load_chunks(files):
    working_dir = os.getcwd()
    all_chunks = []
    for file in files:
        file_path = f"{working_dir}/uploaded/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    
        file_type =os.path.splitext(file.name)[-1].lower() #extracting the extension
        if file_type == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_type == ".txt":
            loader = TextLoader(file_path)
        elif file_type == ".docx":
            loader = Docx2txtLoader(file_path)
        elif file_type ==".csv":
            loader = CSVLoader(file_path)
        else:
            continue
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        all_chunks.extend(chunks)
    return all_chunks


def vector_store(chunks,store_path):
    model = "sentence-transformers/all-MiniLM-L12-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    hf = HuggingFaceEndpointEmbeddings(
        model= model,
        task="feature-extraction"
    )
    vectorstore = FAISS.from_documents(chunks, hf)
    vectorstore.save_local(store_path)

    return vectorstore

def create_chain(vectorstore):
    llm = ChatGroq(model="llama-3.1-8b-instant",
                   temperature=0.5)

    retriever = vectorstore.as_retriever(  )
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    custom_prompt = PromptTemplate(
        input_variables=["context","chat_history","question"],
        template=(
            '''You are a helpful assistant whose role is to read the given information and analyze the needs of the user:
            Example: If the answer needs comparing things-->respond with tables, if question needs some visualizations-->respond
            with necessary diagrams.
            Before responding look at three things:
            1. Context: {context}
            2. Chat History: {chat_history}
            3. Question: {question}
            **NOTE:
            1. If the Context is empty, ask user to ask questions related to the uploaded documents.
            2. Help the users by asking them one single question they might want to know.
            3. Format your responses using correct markdowns.**
            '''
        )
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        verbose=True
    )
    return chain

