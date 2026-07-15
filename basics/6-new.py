#conversation memory
# Import the os module to access environment variables
import os

# Import load_dotenv to load variables from the .env file
from dotenv import load_dotenv

# Import LLMChain to connect the prompt, LLM, and memory together
from langchain.chains import LLMChain

# Import ChatGroq to use Groq-hosted language models
from langchain_groq import ChatGroq

# Import PromptTemplate to create a reusable prompt with dynamic variables
from langchain.prompts import PromptTemplate

# Import ConversationBufferMemory to store the conversation history
from langchain.memory import ConversationBufferMemory


# ---------------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# ---------------------------------------------------------

# Load environment variables from the .env file
# override=True means values from the .env file will override
# existing environment variables with the same name
load_dotenv()


# ---------------------------------------------------------
# 2. GET GROQ API KEY
# ---------------------------------------------------------

# Read the GROQ_API_KEY value from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

# Check whether the API key exists
# If it doesn't exist, stop the program and raise an error
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")


# ---------------------------------------------------------
# 3. MODEL CONFIGURATION
# ---------------------------------------------------------

# Controls the randomness of the model's responses
# Higher value = more creative/random
# Lower value = more focused/deterministic
temperature = 0.7

# Maximum number of tokens the model can generate in its response
max_tokens = 1500

# Groq model that will be used
model_name = "llama-3.1-8b-instant"


# ---------------------------------------------------------
# 4. INITIALIZE THE GROQ LANGUAGE MODEL
# ---------------------------------------------------------

# Create the ChatGroq model object
# This object is responsible for communicating with the Groq API
llm = ChatGroq(
    model=model_name,
    temperature=temperature,
    max_tokens=max_tokens,
    groq_api_key=groq_api_key
)


# ---------------------------------------------------------
# 5. CREATE THE PROMPT TEMPLATE
# ---------------------------------------------------------

# Define the instructions and dynamic variables for the chatbot
#
# {conversation_history}
#     Contains previous messages stored by memory
#
# {additional_information}
#     Provides extra context about the current user question
#
# {human_input}
#     Contains the user's current question
template = """
    You are a chatbot that is helpful.
    Your goal is to help the user and also make jokes.
    Take what the user is saying, make a joke out of it,
    and also answer the question.

    Previous conversation:
    {conversation_history}

    Additional information:
    {additional_information}

    Human: {human_input}
    Chatbot:
"""


# ---------------------------------------------------------
# 6. CREATE THE PROMPT TEMPLATE OBJECT
# ---------------------------------------------------------

# Create a PromptTemplate object
#
# input_variables tells LangChain which dynamic values
# must be supplied when the chain is executed
prompt = PromptTemplate(
    input_variables=[
        "conversation_history",
        "human_input",
        "additional_information"
    ],
    template=template
)


# ---------------------------------------------------------
# 7. CREATE CONVERSATION MEMORY
# ---------------------------------------------------------

# ConversationBufferMemory stores the entire conversation history
# without summarizing or removing previous messages
memory = ConversationBufferMemory(

    # This must match the {conversation_history}
    # variable used inside the prompt template
    memory_key="conversation_history",

    # Specifies which input field represents the user's message
    input_key="human_input",

    # Return previous conversation history as message objects
    # rather than as a plain string
    return_messages=True
)


# ---------------------------------------------------------
# 8. CREATE THE LLM CHAIN
# ---------------------------------------------------------

# LLMChain combines:
# 1. The Groq language model
# 2. The prompt template
# 3. The conversation memory
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,

    # Shows internal chain execution details in the terminal
    verbose=True
)


# ---------------------------------------------------------
# 9. FIRST QUESTION
# ---------------------------------------------------------

# Ask the chatbot whether a pear is a fruit or vegetable
# This conversation will be automatically stored in memory
response1 = chain.predict(
    human_input="Is pear a fruit or vegetable?",
    additional_information=(
        "The user is curious about the classification of pears."
    )
)

# Print the chatbot's response
print(response1)


# ---------------------------------------------------------
# 10. SECOND QUESTION — TESTING MEMORY
# ---------------------------------------------------------

# Ask about the previous conversation
# The model should remember that the user previously asked about a pear
response2 = chain.predict(
    human_input="What was one of the fruits I first asked you about?",
    additional_information=(
        "The user is asking about a previous fruit they mentioned."
    )
)

# Print the chatbot's response
print(response2)


# ---------------------------------------------------------
# 11. THIRD QUESTION
# ---------------------------------------------------------

# Ask a general knowledge question about India's 2014 General Election
response3 = chain.predict(
    human_input=(
        "Which party won the General Elections in India in the year 2014?"
    ),
    additional_information=(
        "The user is asking about the winner of the "
        "2014 General Elections in India."
    )
)

# Print the chatbot's response
print(response3)