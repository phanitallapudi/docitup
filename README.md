# Docitup
  
This package provides various document loaders that utilize different methods for processing and chunking documents. It is designed to facilitate the loading of documents in various formats into a structured format suitable for using them with langchain vector databases
  
## Overview  
  
The package includes the following loaders:  
- **PyMUPdf4LLMLoader**: Loads and splits documents from files using the `pymupdf4llm` library.  
- **MarkitdownLoader**: Loads documents using the `MarkItDown` library.  
- **LlamaparseLoader**: Loads documents using the `LlamaParse` library and processes different file types.  
- **DoclingPDFLoader**: Converts documents to text and splits them accordingly.  
  
## Installation  
  
To install this package, simply run:  
  
```bash  
pip install docitup 
```

## Usage
 

### PyMUPdf4LLMLoader
 
```python
from docitup import PyMUPdf4LLMLoader 
  
loader = PyMUPdf4LLMLoader(file_path='path/to/your/file.pdf')  
documents = loader.load()   
```

### MarkitdownLoader

```python
from docitup import MarkitDownLoader
  
loader = MarkitdownLoader(file_path='path/to/your/file.md')  
documents = loader.load()  
```

### LlamaparseLoader

```python
from docitup import LlamaparseLoader
from llama_parse.utils import ResultType
  
loader = LlamaparseLoader(file_path='path/to/your/directory', result_type=ResultType.MD, api_key='your_api_key')  
documents = loader.load()  
```

### DoclingPDFLoader
```python
from docitup import DoclingLoader
  
loader = DoclingLoader(file_path='path/to/your/file.pdf')  
documents = loader.load()
```

### FitzPyMUPDFLoader
```python
from docitup import FitzPyMUPDFLoader
  
loader = FitzPyMUPDFLoader(file_path='path/to/your/file.pdf')  
documents = loader.load()
```

### PyPdfLoader
```python
from docitup import PyPdfLoader
  
loader = PyPdfLoader(file_path='path/to/your/file.pdf')  
documents = loader.load()
```

### PyPdfLoader2
```python
from docitup import PyPdfLoader2
  
loader = PyPdf2Loader(file_path='path/to/your/file.pdf')  
documents = loader.load()
```

## Configuration Options

Each loader can be configured with the following optional parameters:

`splitter_type`: The type of text splitter to use ("recursive" or other).

`chunk_size`: The size of each chunk (default is 1000).

`chunk_overlap`: The number of overlapping characters between chunks (default is 100).

### Example Usage with all parameters
```python
from docitup import LlamaparseLoader

# Initialize the loader
loader = LlamaparseLoader(
    file_path="example.pdf",
    api_key="your_api_key",
    splitter_type="recursive",
    chunk_size=500,
    chunk_overlap=50,
    extra_metadata={"category": "example"}
)

# Load documents lazily
for document in loader.load():
    print("Text Chunk:", document.text)
    print("Metadata:", document.metadata)

```

## Contributing
 
Contributions are welcome! Please feel free to submit issues or pull requests for improvements or bug fixes.

## License
 
This project is licensed under the MIT License. See the LICENSE file for more information.

## Acknowledgements

This package is made possible by the following libraries:

* [pymupdf4llm](https://pypi.org/project/pymupdf4llm/)
* [MarkItDown](https://pypi.org/project/markitdown/)
* [LlamaParse](https://pypi.org/project/llama-parse/)
* [Docling](https://pypi.org/project/docling/)
