from llama_index.core import SimpleDirectoryReader
from llama_parse import LlamaParse  # type: ignore
from llama_parse.utils import ResultType  # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document as LCDocument
from typing import Iterator, Dict, List, Optional

class LlamaparseLoader(BaseLoader):
    def __init__(
            self, 
            file_path: str | list[str], 
            result_type: ResultType, 
            api_key: str,
            chunk_size: int = 1000,
            chunk_overlap: int = 100
        ) -> None:
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self.parser = LlamaParse(
            api_key=api_key,
            result_type=result_type.MD
        )
        self.file_extractor = {".pdf": self.parser}
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _text_splitter(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        documents: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        is_separator_regex: bool = False,
    ):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=is_separator_regex,
        )
        return text_splitter.create_documents( # type: ignore
            texts=[documents], # type: ignore
            metadatas=[metadata] if metadata else [{}],
        ) 

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
