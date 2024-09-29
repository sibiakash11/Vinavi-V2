import os
import streamlit as st
from streamlit_mic_recorder import speech_to_text
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Import the modules
from module_meaning import setup_rag_pipeline as setup_rag_pipeline_meaning
from module_example import setup_rag_pipeline as setup_rag_pipeline_example

# Set page configuration
st.set_page_config(page_title="Tamil Kids Companion", page_icon="📝")

# Add a title and description to the Streamlit app
st.title("Tamil Companion for Kids")
st.write("""
You are a Tamil companion for 9-year-old kids in Singapore. Ask any question related to the Tamil book provided.
""")


# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Initialize selected option
if 'selected_option' not in st.session_state:
    st.session_state['selected_option'] = None

# Function for content moderation (same as before)
def moderate_content(user_input: str) -> bool:
    """
    Uses GPT-4 to check if the user input contains inappropriate, abusive, or exploitative content.
    Returns True if inappropriate content is found, False otherwise.
    """
    moderation_prompt = """
You are an assistant that checks if a user's input is appropriate for a 9-year-old child in Singapore.
Your task is to analyze the input and determine if it contains any inappropriate, abusive, or exploitative content.
If the input is inappropriate for a child, respond with "Yes".
If the input is appropriate, respond with "No".

User Input: {user_input}

Is the user input inappropriate for a 9-year-old child? (Yes/No):
"""

    # Initialize the content moderation LLM
    moderation_llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.0,
        max_tokens=5,
    )

    prompt = PromptTemplate(
        input_variables=["user_input"],
        template=moderation_prompt,
    )

    # Format the prompt
    formatted_prompt = prompt.format(user_input=user_input)

    # Get the moderation response
    response = moderation_llm.predict(formatted_prompt).strip().lower()

    # Check if the response starts with 'yes'
    if response.startswith('yes'):
        return True  # Content is inappropriate
    else:
        return False  # Content is appropriate

# User input at the bottom
st.write("## Your Input")

# Add a column layout for the input bar and the microphone button
col1, col2 = st.columns([4, 1])

# Speech-to-text with the mic button
with col2:
    st.write("🎤 Record:")
    # Use speech-to-text for Tamil (ta-IN for Tamil)
    tamil_text = speech_to_text(language='ta-IN', start_prompt="🎤 Start", stop_prompt="🛑 Stop", key='STT')
    
# Text input
with col1:
    if tamil_text:
        user_input = st.text_input("Type your question here or use the mic:", value=tamil_text)
    else:
        user_input = st.text_input("Type your question here or use the mic:")

# Option buttons (Added new option "மேலும் கூறு")
st.write("## Choose an Option")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("பொருள் கூறுக"):
        st.session_state['selected_option'] = 'meaning'
with col2:
    if st.button("உதாரணம் கொடுக்கவும்"):
        st.session_state['selected_option'] = 'example'
with col3:
    if st.button("சுருக்கமாக சொல்வது"):
        st.session_state['selected_option'] = 'summarize'
with col4:
    if st.button("விளக்கம் தருக"):
        st.session_state['selected_option'] = 'explain'
with col5:
    if st.button("மேலும் கூறு"):
        st.session_state['selected_option'] = 'breakdown'

# Process input and generate response
if user_input and st.session_state['selected_option']:
    # First, add user message to chat history
    st.session_state['messages'].append({"role": "user", "content": user_input})

    # Check for inappropriate content
    with st.spinner("Checking content..."):
        if moderate_content(user_input):
            # If content is inappropriate, respond with a predefined statement
            predefined_response = "மன்னிக்கவும், அந்த கேள்விக்கு பதில் வழங்க முடியவில்லை."
            st.session_state['messages'].append({"role": "assistant", "content": predefined_response})
        else:
            # Proceed with the selected RAG pipeline
            if st.session_state['selected_option'] == 'meaning':
                qa_chain = setup_rag_pipeline_meaning()
            elif st.session_state['selected_option'] == 'example':
                qa_chain = setup_rag_pipeline_example()
            elif st.session_state['selected_option'] == 'summarize':
                qa_chain = setup_rag_pipeline_summarize()
            elif st.session_state['selected_option'] == 'explain':
                qa_chain = setup_rag_pipeline_explain()
            elif st.session_state['selected_option'] == 'breakdown':
                qa_chain = setup_rag_pipeline_breakdown()
            else:
                st.error("Invalid option selected.")
                st.stop()

            # Process the user input through the RAG pipeline
            with st.spinner("Thinking..."):
                result = qa_chain({"query": user_input})
                answer = result['result']

            # Add assistant response to chat history
            st.session_state['messages'].append({"role": "assistant", "content": answer})

    # Reset selected option after processing
    st.session_state['selected_option'] = None

# Create a container for the chat history and display it above the input
st.write("## Chat History")
with st.container():
    # Display chat history (latest message closest to input)
    messages = list(reversed(st.session_state['messages']))
    
    for idx in range(0, len(messages), 2):  # Iterate in pairs of user and assistant
        if idx + 1 < len(messages):
            # Display user message first
            st.markdown(f"**You:** {messages[idx]['content']}")
            # Then display the assistant response
            st.markdown(f"**Assistant:** {messages[idx + 1]['content']}")
            # Add separator after the assistant's message
            st.markdown("<hr>", unsafe_allow_html=True)
