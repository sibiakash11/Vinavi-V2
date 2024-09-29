# module_example.py

import os
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA  # Using RetrievalQA for RAG pipeline
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

def setup_rag_pipeline_example() -> RetrievalQA:
    """Sets up the RAG pipeline for 'Provide an example in Tamil within Singapore context'."""
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.3,
        max_tokens=100,  # Limit to 100 tokens
    )

    prompt_template = """
You are a friendly Tamil companion for 9-year-old kids in Singapore. The example should be easy for children to grasp. Focus on Singapore-related context. Avoid any complex Tamil words and break down difficult concepts where necessary. Always check for and flag any abusive, misleading, or exploitative content. Ensure that the example sentence is safe and free of misinformation.

1. Provide an example sentence in Tamil related to Singapore, using the given word or phrase.
2. The example should be simple and easily understandable by children.
3. Make sure your example is politically and grammatically correct but never contains abusive, misleading, or exploitative content.
4. Keep the example short and clear, and ensure it is appropriate for young kids.
5. If there is a strong similarity match with the context,try giving the example from the context which is coherently correct.s

Context: {context}

Question: {question}

Answer:
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    # Load the vector store
    vectorstore_path = "data/vectorstore_med"
    if os.path.exists(vectorstore_path):
        embeddings_model = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings_model,
            allow_dangerous_deserialization=True
        )
    else:
        raise FileNotFoundError("Vector store 'vector_med' not found in the data folder.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 'stuff' chain type concatenates retrieved docs
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )
    return qa_chain
