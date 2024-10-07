from langchain.chat_models import ChatOpenAI

def generate_nirappugaa_exercise(api_key):
    """
    Generate a child-friendly 75-word passage with 2-3 blanks along with strong clues for each blank.
    """
    # Use OpenAI API to generate the passage with blanks
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.7,
        openai_api_key=api_key
    )
    
    prompt = """
Generate a child-friendly passage in Tamil suitable for a 9-year-old child. The passage should be approximately 75 words, meaningful, and easy to understand for a 9-year-old kid.

The passage must use only pure Tamil words, avoiding any English-based words.

Select 3 important words from the passage that are appropriate to be turned into blanks for a fill-in-the-blanks exercise. Replace these words in the passage with "_________________________".
The words should be **nouns, adjectives, or verbs** (but not proper nouns). The selected words should be **crucial to the overall meaning of the passage**.

For each blank, provide a **clue** that will help a 9-year-old child identify the correct word. The clue should be **simple, contain 2 sentences**, and directly relate to the context of the blank word.

Provide 6 different words as **options**, including the correct answers for the blanks. Ensure that the **correct answers are randomized** among the options.

**Format**:
Passage:
[Your passage with blanks]

Blanks:
1. [Actual word for blank 1]
2. [Actual word for blank 2]
...

Options:
Option1, Option2, Option3, ...

Clues:
1. [Clue for blank 1]
2. [Clue for blank 2]
...
"""
    response = llm.predict(prompt)

    # Debug: Print the raw LLM response to inspect
    print("LLM Response:", response)

    # Parse the response to extract the passage, blanks, clues, and options
    lines = response.strip().split('\n')
    passage_lines = []
    blanks = []
    clues = []
    options = []

    is_passage = False
    is_blanks = False
    is_clues = False
    is_options = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith('passage:'):
            is_passage = True
            is_blanks = False
            is_clues = False
            is_options = False
            continue
        elif line.lower().startswith('blanks:'):
            is_passage = False
            is_blanks = True
            is_clues = False
            is_options = False
            continue
        elif line.lower().startswith('options:'):
            is_passage = False
            is_blanks = False
            is_clues = False
            is_options = True
            continue
        elif line.lower().startswith('clues:'):
            is_passage = False
            is_blanks = False
            is_clues = True
            is_options = False
            continue

        if is_passage:
            passage_lines.append(line)
        elif is_blanks and '.' in line:
            # Split by first occurrence of '.' and extract the word
            try:
                blanks.append(line.split('.', 1)[1].strip())
            except IndexError:
                print(f"Error parsing blank: {line}")  # Debug information for parsing issue
        elif is_options:
            # Split options by commas
            options = [opt.strip() for opt in line.split(',')]
        elif is_clues and '.' in line:
            try:
                clues.append(line.split('.', 1)[1].strip())
            except IndexError:
                print(f"Error parsing clue: {line}")  # Debug information for parsing issue

    passage = '\n'.join(passage_lines).strip()

    # Ensure that all components (passage, blanks, clues, options) are generated properly
    if not passage or len(blanks) < 2 or len(clues) < 2 or len(options) < 6:
        # Debug: Print the components for inspection if the ValueError is triggered
        print("Passage Generated:", passage)
        print("Blanks Generated:", blanks)
        print("Clues Generated:", clues)
        print("Options Generated:", options)
        raise ValueError("The passage, blanks, clues, or options were not generated correctly. Please check the prompt or parsing logic.")

    # Construct the full exercise with passage, clues, and options for display to the child
    full_exercise = f"{passage}\n\nகுறிப்புகள்:\n" + "\n".join([f"{idx+1}. {clue}" for idx, clue in enumerate(clues)]) + "\n\nவிருப்பங்கள்:\n" + ", ".join(options)

    return {'passage': full_exercise, 'blanks': blanks, 'options': options}

def validate_nirappugaa_answers(passage, blanks, user_answers, options):
    """
    Validate the user's answers for the blanks and provide detailed feedback.
    """
    feedback = ''
    for idx, (correct_word, user_answer) in enumerate(zip(blanks, user_answers)):
        if user_answer == correct_word:
            feedback += f"பகுதி {idx+1}: சரி! உங்கள் பதில் சரியானது. '{correct_word}' என்பது சரியான பதில்.\n\n"
        else:
            feedback += f"பகுதி {idx+1}: தவறு. உங்கள் பதில் சரியானதல்ல. சரியான பதில் '{correct_word}' ஆகும். தயவுசெய்து குறிப்பு பார்க்கவும்: '{options[idx]}'.\n\n"
    
    return feedback.strip()
