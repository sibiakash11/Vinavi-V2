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

Select 2 to 3 important words from the passage that are appropriate to be turned into blanks for a fill-in-the-blanks exercise. Replace these words in the passage with blanks like "_________________________".
The words can be nouns, adjectives, or verbs but not proper nouns. 

After the passage, provide a list of strong clues that will help the child figure out each blank. The clues should be simple, clear, and suitable for a 9-year-old to understand. Each clue should be meaningful enough to make it easy for the child to identify the correct word.
The clues can have multiple sentences for each blank giving more context to the blank. 

Finally, give 6 to 8 different words which include the answers for the blanks and other words from which the kid can choose the right options. The correct answers should be randomized within the options.

Provide the output in the following format:

Passage:
[Your passage with blanks]

Blanks:
1. The actual word for blank 1
2. The actual word for blank 2
...

Options:
Option1, Option2, Option3, ...

Clues:
1. Clue for blank 1
2. Clue for blank 2
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

def validate_nirappugaa_answers(passage, blanks, user_answers, api_key):
    """
    Validate the user's answers for the blanks and provide detailed feedback.
    """
    feedback = ''
    for idx, (correct_word, user_answer) in enumerate(zip(blanks, user_answers)):
        prompt = f"""
You are a helpful assistant proficient in Tamil.

A child has attempted to fill in the blank number {idx+1} in the following passage:

Passage:
{passage}

Child's Answer for Blank {idx+1}:
{user_answer}

First, determine if the child's answer is correct based on the context of the passage. The answer is correct even if it is a synonym of the respective blank word. If it is correct, respond: "சரி! உங்கள் பதில் சரியானது." Then, provide a brief explanation reinforcing why the answer is correct.

If the answer is incorrect, respond: "தவறு. உங்கள் பதில் சரியானதல்ல." Then, explain what the correct word is, why it is the correct choice in this context, and provide a simplified version of the clue to help the child understand.

Provide your response in Tamil.
"""
        llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.7,
            openai_api_key=api_key
        )
        response = llm.predict(prompt)
        feedback += f"பகுதி {idx+1}:\n{response.strip()}\n\n"
    return feedback.strip()
