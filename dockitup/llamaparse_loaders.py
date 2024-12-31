from llama_index.core import SimpleDirectoryReader
from llama_parse import LlamaParse  # type: ignore
from llama_parse.utils import ResultType  # type: ignore
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document as LCDocument
from typing import Iterator, List

from .base_loaders import BaseLoader as LCBaseLoader

class LlamaparseLoader(BaseLoader, LCBaseLoader):
    def __init__(
            self, 
            file_path: str | list[str], 
            result_type: ResultType, 
            api_key: str,
            chunk_size: int = 1000,
            chunk_overlap: int = 100
        ) -> None:
        super().__init__()
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self.parser = LlamaParse(
            api_key=api_key,
            result_type=result_type.MD
        )
        self.file_extractor = {".pdf": self.parser}
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def lazy_load(self) -> Iterator[LCDocument]:
        documents = SimpleDirectoryReader(input_files=self._file_paths, file_extractor=self.file_extractor).load_data() # type: ignore
        texts: List[LCDocument] = []
        
        for doc in documents:
            if 'file_path' in doc.metadata:
                doc.metadata['source'] = doc.metadata.pop('file_path')

            chunks = self._text_splitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                documents=doc.text,
                metadata=doc.metadata
            )
            texts.extend(chunks)

        return iter(texts)
