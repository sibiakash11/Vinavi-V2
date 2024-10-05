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
        max_tokens=150,  # Increased token limit for a more detailed response
    )

    prompt_template = """
You are a friendly Tamil-speaking companion for 9-year-old children in Singapore. Your task is to provide a few example sentences related to the given word/topic in simple Tamil. These examples should be easy for children to understand.

Important instructions:
1. Provide **two example sentences** in Tamil related to the given question.
2. **Use content from the provided context** if there is a strong similarity, but make sure it is explained simply. Do **not hallucinate** content from the book. Only use information explicitly provided in the context.
3. **Avoid complex Tamil words**. Instead, use simple language that is suitable for young children.
4. **Explain each example** clearly using the words used in the sentence in Tamil to help children understand what the example means.
5. Each example must be grammatically and politically correct, culturally relevant to Singapore if there is a feasibility, and easy to relate to for children.
6. **Do not include the question itself in your response**. Start directly with the examples.

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

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 'stuff' chain type concatenates retrieved docs
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )
    return qa_chain
