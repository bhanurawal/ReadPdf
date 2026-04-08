import streamlit as st
import os
import uuid
import tempfile

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate

OPENAI_API_KEY = os.getenv("API_KEY")


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
        self.persist_directory = "vector_store"

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

            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
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
    st.set_page_config(page_title="PDF RAG App", layout="wide")
    st.title("📄 PDF Question Answering (RAG)")

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
