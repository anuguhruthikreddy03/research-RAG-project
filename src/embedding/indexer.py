import os
import tempfile
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

load_dotenv()

def create_vectorstore_from_uploaded_file(uploaded_file):
    """
    Ingests file streams from the Streamlit view layer, structures layout text blocks
    via memory-safe Docling options, and indexes those records inside an in-memory 
    Qdrant Hybrid dense + sparse matrix.
    """
    processed_documents = []
    
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.read())
        temp_pdf_path = temp_pdf.name

    try:
        
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False  
        pipeline_options.do_table_structure = True  #
        
        native_converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        loader = DoclingLoader(
            file_path=temp_pdf_path,
            export_type=ExportType.DOC_CHUNKS,
            converter=native_converter
        )
        
        print(f"Docling decomposing layout profiles for: {uploaded_file.name}")
        raw_chunks = loader.load()
        
        
        for doc in raw_chunks:
            cleaned_content = " ".join(doc.page_content.split())
            if cleaned_content.strip():
                processed_documents.append(
                    Document(
                        page_content=cleaned_content,
                        metadata={"filename": uploaded_file.name}  
                    )
                )
    finally:
        
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            
    if not processed_documents:
        return None, []

    
    dense_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    
    qdrant_hybrid = QdrantVectorStore.from_documents(
        documents=processed_documents,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        location=":memory:",
        collection_name="harness_hybrid",
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse"
    )
    
    return qdrant_hybrid, processed_documents
