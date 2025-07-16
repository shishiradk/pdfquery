import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
import os

load_dotenv()

# --------- PDF Text Extraction ---------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# --------- Text Chunking ---------
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


# --------- Vector Store Creation ---------
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# --------- Conversational QA Chain ---------
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the context,
    just say "answer is not available in the context". Don't make up an answer.

    Context:\n{context}
    Question:\n{question}

    Answer:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base", 
        model_kwargs={"temperature": 0.3, "max_length": 512}
    )

    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain


# --------- User Input Processing ---------
def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response["output_text"])


# --------- Streamlit App Main ---------
def main():
    st.set_page_config("Chat with PDF")
    st.header("Chat with PDF using Hugging Face 🤗")

    user_question = st.text_input("Ask a question about the PDF(s):")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📄 Upload PDF Files")
        pdf_docs = st.file_uploader("Upload PDF(s) and click Process", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing Complete ")


if __name__ == "__main__":
    main()
