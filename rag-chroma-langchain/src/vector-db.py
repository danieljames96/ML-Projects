from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import OpenAI
import openai
from pathlib import Path
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

pdf_path = r"d:/ML Projects/rag-chroma-langchain/data/pdfs/annex-a-about-synapxe.pdf"
loaders = [PyPDFLoader(pdf_path)]

docs = []
for file in loaders:
    docs.extend(file.load())
#split text to chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(docs)
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
#print(len(docs))

vectorstore = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db_nccn")

# print(vectorstore._collection.count())