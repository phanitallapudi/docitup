from docling.document_converter import DocumentConverter
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from typing import Iterator

class DoclingPDFLoader(BaseLoader):
    def __init__(self, file_path: str | list[str]) -> None:
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self._converter = DocumentConverter()

    def lazy_load(self) -> Iterator[LCDocument]:
        for source in self._file_paths:
            dl_doc = self._converter.convert(source).document
            text = dl_doc.export_to_markdown()
            yield LCDocument(page_content=text, metadata={"source": source})