import streamlit as st
import os
import tempfile

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate

OPENAI_API_KEY = os.getenv("API_KEY")

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="RAG App", layout="wide")
st.title("📄 Retrieval-Augmented Generation (RAG) with LangChain + Streamlit")

# API Key input (secure)
openai_api_key = OPENAI_API_KEY

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# ---------------------------
# Process PDF and Create Vector Store
# ---------------------------
if uploaded_file and openai_api_key:
    try:
        # Save uploaded file temporarily
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load PDF
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # Store in FAISS vector DB
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Create retriever
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # Create QA chain
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        st.success("✅ PDF processed successfully! You can now ask questions.")

        # ---------------------------
        # Question Answering
        # ---------------------------
        query = st.text_input("Ask a question about the PDF:")
        if query:
            with st.spinner("Thinking..."):
                result = qa_chain.invoke({"query": query})
                st.markdown(f"**Answer:** {result['result']}")

                # Show sources
                with st.expander("📚 Sources"):
                    for i, doc in enumerate(result["source_documents"], start=1):
                        st.markdown(f"**Source {i}:** {doc.metadata.get('source', 'Unknown')}")
                        st.write(doc.page_content[:300] + "...")

    except Exception as e:
        st.error(f"Error: {str(e)}")



"""
class RAGPDFParser:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY
        )

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=OPENAI_API_KEY
        )

        self.vector_store = None

    def process_pdf(self, pdf_file):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.getvalue())
                temp_path = tmp.name

            loader = PyPDFLoader(temp_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=100
            )
            chunks = splitter.split_documents(documents)

            self.vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )

            os.remove(temp_path)
            return len(chunks)

        except Exception as e:
            st.error(f"PDF processing failed: {e}")
            return 0

    def get_answer(self, query):
        if not self.vector_store:
            return "Please upload and process a PDF first."

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant.
            Answer the question using only the context below.

            Context:
            {context}

            Question:
            {question}
            """
        )

        rag_chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain.invoke(query)


def main():
    st.set_page_config(page_title="PDF RAG App (FAISS)", layout="wide")
    st.title("📄 PDF Question Answering — FAISS RAG")

    if "rag_app" not in st.session_state:
        st.session_state.rag_app = RAGPDFParser()

    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if pdf_file and st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            chunks = st.session_state.rag_app.process_pdf(pdf_file)
        if chunks:
            st.success(f"PDF processed successfully — {chunks} chunks created.")

    query = st.text_input("Ask a question about the document")

    if query:
        with st.spinner("Generating answer..."):
            answer = st.session_state.rag_app.get_answer(query)
        st.markdown("### Answer")
        st.write(answer)


if __name__ == "__main__":
    main()
"""
