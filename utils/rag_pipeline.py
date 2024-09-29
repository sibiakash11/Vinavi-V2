# utils/rag_pipeline.py

import os
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.openai import OpenAI as LangchainOpenAI
from langchain.embeddings.base import Embeddings

# Initialize the OpenAI client using the API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define a custom Embeddings class compatible with FAISS
class OpenAIEmbedding(Embeddings):
    def __init__(self, client, model="text-embedding-ada-002"):
        self.client = client
        self.model = model

    def embed_documents(self, texts):
        # Batch processing of texts to get embeddings
        cleaned_texts = [text.replace("\n", " ") for text in texts]
        embeddings = [self.client.embeddings.create(input=[text], model=self.model).data[0].embedding for text in cleaned_texts]
        return embeddings

    def embed_query(self, text):
        # Single query embedding
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(input=[text], model=self.model)
        return response.data[0].embedding

# Function to load the RAG (Retrieval-Augmented Generation) chain
def load_rag_chain():
    # Initialize the OpenAIEmbedding class
    embedding_model = OpenAIEmbedding(client)

    # Create or load the FAISS vector store using the embedding model
    vectorstore = FAISS.from_texts(["sample text to create initial index"], embedding_model)

    # Correctly configure the OpenAI model with the latest setup
    llm = LangchainOpenAI(model="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # Define the conversational retrieval chain with the OpenAI LLM model
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )
    return qa_chain
