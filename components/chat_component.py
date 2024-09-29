# app.py
from chainlit import chainlit
from chainlit import Message, Input

# Initialize the Chainlit application
@chainlit.run()
async def main():
    # Display the main chat interface
    await chatbot()

async def chatbot():
    """
    Basic chatbot function that interacts with the user through text input and generates responses.
    """
    # Ask for user input
    user_input = await Input("Ask me anything...")

    # If the user provides input, process it and generate a response
    if user_input:
        # Here, you can use a function to generate a response (e.g., from an LLM)
        # For simplicity, we'll just echo back the input as the response
        response = f"You said: {user_input}"

        # Display the user's input and the bot's response
        await Message(f"**User:** {user_input}").send()
        await Message(f"**Bot:** {response}").send()
