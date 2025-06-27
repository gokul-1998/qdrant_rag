"""
Centralized module for lazy-loading LangChain components.
This helps reduce startup time and avoid deprecation warnings.
"""

from typing import List, Dict, Any, Optional, Type, TypeVar, Union

# Type variable for Document
T = TypeVar('T')

def get_document_class():
    """Lazy load the Document class"""
    from langchain.schema import Document
    return Document

def get_pdf_loader():
    """Lazy load PyPDFLoader from the correct location"""
    from langchain_community.document_loaders import PyPDFLoader
    return PyPDFLoader

def get_text_splitter():
    """Lazy load RecursiveCharacterTextSplitter"""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter

def get_chroma_db():
    """Lazy load Chroma vectorstore"""
    from langchain_community.vectorstores import Chroma
    return Chroma

def get_prompt_template():
    """Lazy load PromptTemplate"""
    from langchain_core.prompts import PromptTemplate
    return PromptTemplate

def get_sentence_transformer():
    """Lazy load SentenceTransformer"""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer
