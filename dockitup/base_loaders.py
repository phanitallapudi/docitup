from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Dict, Optional

class BaseLoader:
    def __init__(self) -> None:
        pass

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