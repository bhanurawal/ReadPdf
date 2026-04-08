import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import PyPDF2
import uuid
import torch
import os
from sentence_transformers import SentenceTransformer, util

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI


OPENAI_API_KEY = os.getenv("API_KEY")

def read_pdf_pypdf2(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

class RAGPDFParser:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model= "text-embedding-3-small",api_key=OPENAI_API_KEY)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0,api_key=OPENAI_API_KEY)
        self.vector_store = None
        self.persist_directory = "vector_store"

    def process_pdf(self, pdf_file):
        """Process uploaded PDF file and create vector store"""
        try:
            # Create a unique temporary file name
            temp_pdf_path = f"temp_{uuid.uuid4()}.pdf"
            # Save uploaded file temporarily
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_file.getvalue())
            
            # Load and split the PDF
            loader = PyPDFLoader(temp_pdf_path)
            documents = loader.load()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=100,
                length_function=len
            )
            texts = text_splitter.split_documents(documents)
            
            # Create vector store
            self.vector_store = FAISS.from_documents(
                documents=texts,
                embedding=self.embeddings
            )
            
            # Clean up temporary file
            os.remove(temp_pdf_path)
            return len(texts)
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return 0

    def get_answer(self, query):
        """Get answer for the query using RAG"""
        try:
            if not self.vector_store:
                return "Please upload a PDF document first."
            
            # Create retrieval chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": 3}
                )
            )
            
            # Get answer
            response = qa_chain.invoke({"query": query})
            return response["result"]
        except Exception as e:
            return f"Error generating answer: {str(e)}"
            
def split_text(textPdf):
  """
  Split the text content of the given list of Document objects into smaller chunks.
  Args:
    documents (list[Document]): List of Document objects containing text content to split.
  Returns:
    list[Document]: List of Document objects representing the split text chunks.
  """
  # Initialize text splitter with specified parameters
  text_splitter = RecursiveCharacterTextSplitter(
     chunk_size=1000, # Size of each chunk in characters
     chunk_overlap=100, # Overlap between consecutive chunks
     length_function=len, # Function to compute the length of the text
     add_start_index=True, # Flag to add start index to each chunk
   )

  #  text_splitter = SemanticChunker(lc_embed_model)

  # Split documents into smaller chunks using text splitter
  chunks = text_splitter.split_text(textPdf)
  print(f"Split documents into {len(chunks)} chunks.")

  return chunks # Return the list of split text chunks
    

def main():

    ####os.environ["api_key"] == st.secrets["API_KEY"]
    
    st.title("File Upload and Q&A App")
    
    # uploaded_file = st.file_uploader("Choose a file")
    # if uploaded_file is not None:
    #     # Process the uploaded file
    #     st.write("File uploaded successfully!")
    #     # Add your RAG model processing code here
    #     binary_data = uploaded_file.getvalue()
    #     #### pdf_viewer(input=binary_data, width=700)
    #     chunks = split_text(read_pdf_pypdf2(uploaded_file))
            
    # question = st.text_input("Ask a question about the file")
    
    # Initialize RAG application in session state
    if 'rag_app' not in st.session_state:
        st.session_state.rag_app = RAGPDFParser()
        
     # # File upload
    pdf_file = st.file_uploader("Upload your PDF", type=['pdf'])
    if pdf_file:
        st.write("File uploaded successfully")
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                num_chunks = st.session_state.rag_app.process_pdf(pdf_file)
            if num_chunks > 0:
                st.success(f"PDF processed successfully! Created {num_chunks} text chunks.")
    
    # Query input
    query = st.text_input("Ask a question about your PDF:")
    if query:
        with st.spinner("Getting answer..."):
            answer = st.session_state.rag_app.get_answer(query)
            st.write("Answer:", answer)


if __name__ == "__main__":
    main()
