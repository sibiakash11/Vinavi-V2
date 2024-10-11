from langchain.chat_models import ChatOpenAI
import base64
import io
from gtts import gTTS

def reset_essay_session(st):
    """Reset the essay writing session state."""
    keys_to_reset = [
        'essay_step', 'essay_title', 'brainstorming_qna',
        'essay_structure', 'essay_content', 'essay_feedback',
        'essay_mode_started', 'is_processing'
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            st.session_state[key] = '' if isinstance(st.session_state.get(key), str) else False if isinstance(st.session_state.get(key), bool) else 0 if isinstance(st.session_state.get(key), int) else {}

def generate_brainstorming_qna(essay_title, api_key):
    """Generate 5 brainstorming questions with answers/facts in Tamil."""
    prompt = f"""
You are an essay assistant specialized in Tamil for 9-year-old children. Help the child brainstorm ideas for an essay titled '{essay_title}'.

Generate 5 well-formatted, simple open-ended questions in Tamil. For each question, provide an answer or fact appropriate for a child learning Tamil.

At the end, add a prompting question encouraging the child to come up with more ideas.

Format:

கேள்வி 1: [Question in Tamil]
பதில்: [Answer in Tamil]

[Repeat for 5 questions]

"""
    brainstorming_llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.7,
        max_tokens=600,
        openai_api_key=api_key
    )
    response = brainstorming_llm.predict(prompt).strip()
    return response

def generate_essay_structure(essay_title, brainstorming_qna, api_key):
    """Provide the essay structure with all the ideas but not the essay itself."""
    prompt = f"""
You are an essay assistant specialized in Tamil for 9-year-old children. Help the child structure their essay titled '{essay_title}'.

Based on the following brainstorming questions and answers:

{brainstorming_qna}

Provide a simple essay structure in Tamil in 200 words overall, outlining how to organize the introduction (முன்னுரை), body (உட்பகுதி), and conclusion (முடிவு). Include the ideas discussed but do not write the essay itself. Explain how the essay should be structured.

Format:

முன்னுரை:
- [Guidance on what to include]

உட்பகுதி:
- [Guidance on what to include]

முடிவு:
- [Guidance on what to include]
"""
    structure_llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.7,
        openai_api_key=api_key
    )
    response = structure_llm.predict(prompt).strip()
    return response

def get_essay_feedback(essay_content, api_key, brainstorming_qna, essay_title):
    """Provide constructive feedback on the child's essay."""
    prompt = f"""
You are an essay assistant specialized in Tamil for 9-year-old children. Provide constructive feedback of overall 160 words on the following essay:

{essay_content}

Check for grammatical and spelling errors. Explain if the essay is aligned with the brainstorming questions and the topic '{essay_title}'. Highlight strengths and suggest simple improvements appropriate for a child learning Tamil. Keep the language encouraging and easy to understand.
"""
    feedback_llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.7,
        openai_api_key=api_key
    )
    feedback = feedback_llm.predict(prompt).strip()
    return feedback
