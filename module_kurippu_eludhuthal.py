# module_kurippu_eludhuthal.py

import os
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA  # Using RetrievalQA for RAG pipeline
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

def setup_rag_pipeline_kurippu_eludhuthal() -> RetrievalQA:
    """Sets up the RAG pipeline for 'Kurippu Eludhuthal' to assist with note-writing exercises."""
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.3,
    )

    prompt_template = """
You are a helpful Tamil companion for 9-year-old kids in Singapore. Your task is to assist kids with 'குறிப்பு எழுத்து' which focuses on note-writing exercises. Provide guidance on how to structure notes and key points to include.

1. Offer suggestions on organizing their notes effectively.
2. Provide examples of how to summarize information.
3. Encourage the child to use their own words and creativity.
4. Keep the language simple and easy to understand.
5. Do not write the notes for them; guide them on how to do it themselves.

Context: {context}

Question: {question}

Answer:
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    # Load the vector store for the workbook
    vectorstore_path = "data/vectorstore_med"
    if os.path.exists(vectorstore_path):
        embeddings_model = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings_model,
            allow_dangerous_deserialization=True
        )
    else:
        raise FileNotFoundError("Vector store 'vectorstore_kurippu' not found in the data folder.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 'stuff' chain type concatenates retrieved docs
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )
    return qa_chain
