import os
import streamlit as st
from typing import List
import base64
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS  # Import gTTS for text-to-speech
import io
import streamlit.components.v1 as components

# Import the modules
from module_meaning import setup_rag_pipeline_meaning
from module_example import setup_rag_pipeline_example
from module_translation import setup_translation_chain
from module_paadapayirchi import setup_rag_pipeline_paadapayirchi
from module_melum_kooru import setup_melum_kooru_chain

# Set page configuration with wide layout
st.set_page_config(page_title="Tamil Kids Companion", page_icon="📝", layout='wide')

# Load and encode the background image
image_path = "data/img.jpg"
if os.path.exists(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

# Add custom CSS for the background and styling chat history
st.markdown(f"""
    <style>
        body {{
            background-image: url('data:image/jpg;base64,{encoded_image}');
            background-size: cover;
            background-attachment: fixed;
        }}
        .main-content {{
            background-color: rgba(255, 255, 255, 0.8);
            padding: 2rem;
            border-radius: 10px;
        }}
        .chat-history {{
            background-color: #2c2f33;
            color: white;
            border-radius: 10px;
            padding: 10px;
            font-size: 1.1rem;
        }}
        .chat-message {{
            padding: 10px;
            border-bottom: 1px solid #444;
        }}
        .user-message {{
            color: #00c3ff;
        }}
        .assistant-message {{
            color: #fcba03;
        }}
        .input-row {{
            display: flex;
            align-items: center;
        }}
        .input-text {{
            flex: 1;
            padding: 10px;
            margin-right: 10px;
        }}
        .stButton>button {{
            width: 100%;
            height: 50px;
            font-size: 18px;
            border-radius: 8px;
        }}
        .center-text {{
            text-align: center;
        }}
        /* Make the buttons bigger */
        .btn-group {{
            display: flex;
            justify-content: space-evenly;
            padding: 10px;
        }}
        .btn-group button {{
            width: 10px;
        }}
    </style>
""", unsafe_allow_html=True)
# Center Title and Description
st.markdown("<h1 class='center-text'>தமிழ் குழந்தைகளின் துணைவர்</h1>", unsafe_allow_html=True)
st.markdown("<p class='center-text'>9 வயது குழந்தைகளுக்கான தமிழ் துணைவர். தமிழ் புத்தகத்துடன் தொடர்புடைய எந்த கேள்வியையும் கேளுங்கள்.</p>", unsafe_allow_html=True)

# The API key should be provided as an environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'selected_option' not in st.session_state:
    st.session_state['selected_option'] = None

if 'last_answer' not in st.session_state:
    st.session_state['last_answer'] = ''

if 'mic_input' not in st.session_state:
    st.session_state['mic_input'] = ''

# Function for content moderation
def moderate_content(user_input: str) -> bool:
    moderation_prompt = """
You are an assistant that checks if a user's input is appropriate for a 9-year-old child in Singapore in Tamil and English languages.
Your task is to analyze the input and determine if it contains any inappropriate, abusive, or exploitative content.
If the input is inappropriate for a child, respond with "Yes". If the input is appropriate, respond with "No".

User Input: {user_input}

Is the user input inappropriate for a 9-year-old child? (Yes/No):
"""
    moderation_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0, max_tokens=5)
    prompt = PromptTemplate(input_variables=["user_input"], template=moderation_prompt)
    formatted_prompt = prompt.format(user_input=user_input)
    response = moderation_llm.predict(formatted_prompt).strip().lower()
    return response.startswith('yes')

# Function to play audio using text-to-speech (gTTS)
def autoplay_audio(text):
    tts = gTTS(text, lang='ta')
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    audio_bytes = audio_fp.read()
    b64 = base64.b64encode(audio_bytes).decode()
    html_string = f"""
        <audio autoplay>
        <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
        </audio>
    """
    components.html(html_string, height=0)

# Create a two-column layout: Left for chat history, Right for input and responses
col_chat, col_main = st.columns([1, 3])

# Left Column: Chat History
with col_chat:
    st.write("## உரையாடல் வரலாறு")
    # Display chat history from latest to oldest
    for idx, message in reversed(list(enumerate(st.session_state['messages']))):
        if message["role"] == "user":
            st.markdown(f"**நீங்கள்:** {message['content']}")
            if idx - 1 >= 0 and st.session_state['messages'][idx - 1]["role"] == "assistant":
                st.markdown(f"**உதவியாளர்:** {st.session_state['messages'][idx - 1]['content']}")
                st.markdown("---")
        if message["role"] == "assistant":
            continue

# Right Column: Input and Assistant's Response
with col_main:
    st.write("## உங்கள் கேள்வி")
    input_col, mic_col = st.columns([9, 1])

        # Speech-to-text with the mic button
    with mic_col:
        tamil_text = speech_to_text(language='ta-IN', start_prompt="🎤", stop_prompt="🛑", key='STT')
        if tamil_text:
            st.session_state['mic_input'] = tamil_text  # Store mic input to session state
            user_input = tamil_text  # Immediately update the input field value

    # Text input
    with input_col:
        user_input = st.text_input("உங்கள் கேள்வியை இங்கே தட்டச்சு செய்யவும் அல்லது மைக் பயன்படுத்தவும்:", value=st.session_state.get('mic_input', ''))


    st.markdown("<p style='font-size:14px;'>தேர்வு செய்ய ஒரு விருப்பம்</p>", unsafe_allow_html=True)
    button_cols = st.columns([1, 1, 1, 1, 1])
    button_labels = ["பொருள்", "உதாரணம்", "மொழிபெயர்ப்பு", "பாடப் பயிற்சி", "விரிவாக"]
    button_keys = ['meaning', 'example', 'translation', 'paadapayirchi', 'melum_kooru']

    for i, col in enumerate(button_cols):
        with col:
            if st.button(button_labels[i], key=f"{button_keys[i]}_btn"):
                st.session_state['selected_option'] = button_keys[i]

    if user_input and st.session_state['selected_option']:
        st.session_state['messages'].append({"role": "user", "content": user_input})

        # Check for inappropriate content
        with st.spinner("சரிபார்க்கிறது..."):
            if moderate_content(user_input):
                predefined_response = "மன்னிக்கவும், நான் அந்த கேள்விக்கு பதில் அளிக்க முடியாது."
                st.session_state['messages'].append({"role": "assistant", "content": predefined_response})
                st.session_state['last_answer'] = predefined_response
                st.error(predefined_response)
            else:
                with st.spinner("சிந்திக்கிறது..."):
                    if st.session_state['selected_option'] == 'meaning':
                        qa_chain = setup_rag_pipeline_meaning()
                        result = qa_chain({"query": user_input})
                        answer = result['result']
                    elif st.session_state['selected_option'] == 'example':
                        qa_chain = setup_rag_pipeline_example()
                        result = qa_chain({"query": user_input})
                        answer = result['result']
                    elif st.session_state['selected_option'] == 'translation':
                        translation_chain = setup_translation_chain()
                        answer = translation_chain.run(question=user_input)
                    elif st.session_state['selected_option'] == 'paadapayirchi':
                        qa_chain = setup_rag_pipeline_paadapayirchi()
                        result = qa_chain({"query": user_input})
                        answer = result['result']
                    elif st.session_state['selected_option'] == 'melum_kooru':
                        if 'melum_kooru_chain' not in st.session_state:
                            st.session_state['melum_kooru_chain'] = setup_melum_kooru_chain()
                        conversation_chain = st.session_state['melum_kooru_chain']
                        answer = conversation_chain.run(input=user_input)
                    else:
                        st.error("தவறான விருப்பம் தேர்ந்தெடுக்கப்பட்டது.")
                        st.stop()

                    st.session_state['messages'].append({"role": "assistant", "content": answer})
                    st.session_state['last_answer'] = answer

        st.session_state['selected_option'] = None

    # Display assistant's response
    if st.session_state.get('last_answer'):
        st.write("## உதவியாளர் பதில்")
        st.markdown(f"**உதவியாளர்:** {st.session_state['last_answer']}")

        # Read Aloud Button
        if st.button("வாசிக்க", key='read_aloud_btn'):
            autoplay_audio(st.session_state['last_answer'])