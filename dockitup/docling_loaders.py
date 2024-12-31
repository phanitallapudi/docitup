from docling.document_converter import DocumentConverter
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document as LCDocument
from typing import Iterator, List

from .base_loaders import BaseLoader as LCBaseLoader

class DoclingPDFLoader(BaseLoader, LCBaseLoader):
    def __init__(
            self, 
            file_path: str | list[str], 
            chunk_size: int = 1000, 
            chunk_overlap: int = 100
        ) -> None:
        super().__init__()
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self._converter = DocumentConverter()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def lazy_load(self) -> Iterator[LCDocument]:
        # List to store split documents
        all_documents: List[LCDocument] = []

        # Process each file path
        for source in self._file_paths:
            dl_doc = self._converter.convert(source).document
            text = dl_doc.export_to_markdown()
            
            # Use _text_splitter to break the document into chunks
            chunks = self._text_splitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                documents=text,
                metadata={"source": source}
            )
            
            # Add the chunks to the list
            all_documents.extend(chunks)
        
        # Return the documents as an iterator
        return iter(all_documents)
