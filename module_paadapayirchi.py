# module_paadapayirchi.py

import os
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA  # Using RetrievalQA for RAG pipeline
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

def setup_rag_pipeline_paadapayirchi() -> RetrievalQA:
    """Sets up the RAG pipeline for 'PaadaPayirchi' to assist with Tamil exercises."""
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.3,
    )

    prompt_template = """
You are a friendly Tamil companion for 9-year-old kids in Singapore. Your task is to help kids with their Tamil exercises. Always be empathetic and make sure to provide tips and suggestions on how to work on their exercises. Use the content from the workbook provided in the RAG store.

1. Provide helpful guidance and tips related to the question, using information from the workbook.
2. Ensure that your suggestions are easy for children to understand.
3. Avoid complex words and keep the language simple.
4. Always check for and flag any abusive, misleading, or exploitative content.
5. Do not give away the answer, rather try helping the child in figuring out on arriving to an answer and always offer a set of tips and suggestions.

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
            allow_dangerous_deserialization = True       )
    else:
        raise FileNotFoundError("Vector store 'vector_workbook' not found in the data folder.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 'stuff' chain type concatenates retrieved docs
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )
    return qa_chain
