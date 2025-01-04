# docitup/__init__.py

from .base_loaders import BaseLoader
from .docling_loaders import DoclingLoader
from .llamaparse_loaders import LlamaparseLoader
from .markitdown_loaders import MarkitdownLoader
from .pymupdf4llm_loaders import PyMUPdf4LLMLoader
from .fitz_loaders import FitzPyMUPDFLoader
from .pypdf_loaders import PyPdfLoader
from .pypdf2_loaders import PyPdf2Loader

__all__ = [
    "BaseLoader",
    "DoclingLoader",
    "LlamaparseLoader",
    "MarkitdownLoader",
    "PyMUPdf4LLMLoader",
    "FitzPyMUPDFLoader",
    "PyPdfLoader",
    "PyPdf2Loader"
]
