from setuptools import setup, find_packages

setup(
    name="docitup",  # same as in the toml file
    version="0.1.2",  # same as in the toml file
    author="Panindhra",  # same as in the toml file
    author_email="phanitallapudi@gmail.com",  # same as in the toml file
    description="docitup is a Python package designed to simplify document processing for LangChain. It provides various loaders to extract content from different file types and convert them into LangChain-compatible document classes, ready for storage in LangChain-supported vector stores.",  # you can update this with a proper description
    long_description=open("README.md").read(),  # same as in the toml file
    long_description_content_type="text/markdown",  # typically used for markdown files
    url="https://github.com/phanitallapudi/docitup",  # replace with your actual repo URL if available
    packages=find_packages(),  # automatically find all packages in the project
    install_requires=[  # translating poetry dependencies
        "docling>=2.14.0",
        "langchain-community>=0.3.13",
        "llama-parse>=0.5.19",
        "langchain>=0.3.13",
        "llama-index-readers-file>=0.4.1",
        "markitdown>=0.0.1a3",
        "pymupdf4llm>=0.0.17",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.12",  # as per your Python version requirement in the toml file
    classifiers=[  # typical classifiers for PyPI
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",  # update if different license
        "Operating System :: OS Independent",
    ],
    include_package_data=True,  # this will include additional files like README.md
    zip_safe=False,  # you can set this to True if your package doesn't rely on compiled code
)
