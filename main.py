import os as os.environ["HUGGINGFACEHUB_API_TOKEN"] = "HUGGINGFACEHUB_API_TOKEN"
import PyPDF2
import random
import itertools
import textwrap
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from io import StringIO
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator

st.set_page_config(page_title="Buscar Metodologias", page_icon=':shark:')

def getDocument():
loader = TextLoader('./file.pdf')
documents = loader.load()
return documents



def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')
    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text
  #Implementar esta parte del codigo mas adelante

def main():
embeddings = HuggingFaceEmbeddings()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
db = FAISS.from_documents(docs, embeddings)
llm=HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":512})
chain = load_qa_chain(llm, chain_type="stuff")
user_question = st.text_input("Ingresa tu pregunta:")
if user_question:
  answer = db.similarity_search(user_question)
  st.write("Respuesta:", answer)
  chain.run(input_documents=docs, question=user_question)
#query = "What did the president say about the Supreme Court"
#docs = db.similarity_search(query)
if __name__ == "__main__":
    main()
