# models/llm_integration.py
import openai

# Set your OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

def generate_response(query, context):
    """
    Generate a response using GPT-3.5/4 with the provided context.

    Args:
        query (str): The user's query.
        context (str): The retrieved context relevant to the query.

    Returns:
        str: Generated response from the LLM.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a friendly learning companion that communicates in simple Tamil."},
            {"role": "user", "content": f"{query}. Context: {context}"}
        ]
    )
    return response['choices'][0]['message']['content']