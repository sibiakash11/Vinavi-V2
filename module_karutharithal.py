# module_karutharithal.py

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

def generate_karutharithal_exercise(api_key):
    """
    Generate a child-friendly 150-word passage and 3 related questions.
    """
    # Use OpenAI API to generate the passage and questions
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.7,
        openai_api_key=api_key
    )
    
    prompt = """
Generate a child-friendly passage in Tamil suitable for a 9-year-old child. The passage should be approximately 150 words, meaningful, and easy to understand.
The passage should be completetely grammatically and politically correct.
The stories should never have direct speech.
After the passage, create three questions based directly on the passage content. The answers to these questions should be available directly in the passage.

Provide the output in the following format:

Passage:
[Your passage]

Questions:
1. Question one?
2. Question two?
3. Question three?
"""
    response = llm.predict(prompt)
    # Parse the response to extract the passage and questions
    lines = response.strip().split('\n')
    passage_lines = []
    questions = []
    is_passage = False
    is_questions = False
    for line in lines:
        if line.strip().lower().startswith('passage:'):
            is_passage = True
            is_questions = False
            continue
        elif line.strip().lower().startswith('questions:'):
            is_passage = False
            is_questions = True
            continue
        if is_passage:
            passage_lines.append(line)
        elif is_questions:
            if line.strip():
                questions.append(line.strip()[3:].strip())  # Remove numbering
    passage = '\n'.join(passage_lines).strip()
    return {'passage': passage, 'questions': questions}

def validate_karutharithal_answers(passage, questions, user_answers, api_key):
    """
    Validate the user's answers and provide detailed feedback.
    """
    # Use OpenAI API to validate the answers
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.7,
        openai_api_key=api_key
    )
    feedback = ''
    for idx, (question, user_answer) in enumerate(zip(questions, user_answers)):
        prompt = f"""
You are a helpful assistant proficient in Tamil.

A child has read the following passage:

Passage:
{passage}

They have answered the following question:

Question:
{question}

Child's Answer:
{user_answer}

First, determine if the child's answer is correct based on the passage. Find similarity to the given answer and mark an answer correct if similarity is more than 30 percent. If it is correct, respond: "சரி! உங்கள் பதில் சரியானது." Then, provide a brief explanation reinforcing why the answer is correct.
If there is no answer at all then respond in tamil stating that no answer was entered.
If the answer is incorrect, respond: "தவறு. உங்கள் பதில் சரியானதல்ல." Then, explain what the correct answer is and why it is correct, referencing the passage.

Provide your response in Tamil.
"""
        response = llm.predict(prompt)
        feedback += f"பதில் {idx+1}:\n{response.strip()}\n\n"
    return feedback.strip()
