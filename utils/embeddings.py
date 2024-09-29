# embeddings.py

import os
import pickle
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def create_embeddings(text_file):
    # Import necessary modules from indic-nlp-library
    from indicnlp.tokenize import sentence_tokenize
    from indicnlp import common
    from indicnlp import loader

    # Set the path to the Indic NLP Resources folder
    INDIC_RESOURCES_PATH = "C:/Users/sibia/Downloads/indic_nlp_resources-master/indic_nlp_resources-master"  
    common.set_resources_path(INDIC_RESOURCES_PATH)
    loader.load()

 
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split text into sentences using the Indic NLP Library
    language = 'ta'  # 'ta' is the ISO code for Tamil
    documents = sentence_tokenize.sentence_split(text, lang=language)

    # Initialize OpenAI Embeddings
    embeddings = OpenAIEmbeddings()

    # Create FAISS vector store
    vectorstore = FAISS.from_texts(documents, embeddings)

    # Save the vector store to disk
    vectorstore.save_local('data/faiss_index')

if __name__ == "__main__":
    create_embeddings('data/unit10.txt')