# ============================================================
# LangChain Structured Output Parser Example using Groq
# ============================================================

# Import load_dotenv to load environment variables from a .env file
from dotenv import load_dotenv

# Import os to access environment variables
import os

# Import ChatGroq to communicate with Groq-hosted LLM models
from langchain_groq import ChatGroq

# Import PromptTemplate to create a reusable prompt
from langchain_core.prompts import PromptTemplate

# Import StructuredOutputParser and ResponseSchema
# These are used to force the LLM response into a predefined structure
from langchain_classic.output_parsers import (
    StructuredOutputParser,
    ResponseSchema
)


# ============================================================
# 1. Load environment variables
# ============================================================

# Load variables from the .env file
# override=True means .env values override existing environment variables
load_dotenv(override=True)


# ============================================================
# 2. Get the Groq API key
# ============================================================

# Read GROQ_API_KEY from the environment
groq_api_key = os.getenv("GROQ_API_KEY")

# Check whether the API key exists
if not groq_api_key:
    raise ValueError("Groq API key environment variable is not set")


# ============================================================
# 3. Configure the LLM
# ============================================================

# Controls randomness of the model's response
# Higher value = more creative
# Lower value = more deterministic
temperature = 0.7

# Maximum number of tokens the model can generate
max_tokens = 1500

# Groq model name
model_name = "llama-3.1-8b-instant"


# Create the Groq LLM object
llm = ChatGroq(
    model=model_name,
    temperature=temperature,
    max_tokens=max_tokens,
    groq_api_key=groq_api_key
)


# ============================================================
# 4. Define the expected response structure
# ============================================================

# ResponseSchema defines the fields that we expect
# the LLM to return in its response
response_schemas = [

    # First field: original poorly formatted text
    ResponseSchema(
        name="bad_string",
        description="The original poorly formatted user response"
    ),

    # Second field: corrected and properly formatted text
    ResponseSchema(
        name="good_string",
        description="The corrected, well-formatted, and properly spelled user response"
    ),
]


# ============================================================
# 5. Create the Structured Output Parser
# ============================================================

# Create a parser based on the response schemas defined above
output_parser = StructuredOutputParser.from_response_schemas(
    response_schemas
)


# ============================================================
# 6. Get formatting instructions
# ============================================================

# LangChain automatically generates instructions telling the LLM
# exactly how its output should be structured
format_instructions = output_parser.get_format_instructions()

print("Format Instructions:")
print(format_instructions)


# ============================================================
# 7. Create the prompt template
# ============================================================

template = """
You shall be given a poorly formatted string from a user.

Your tasks:
1. Keep the original poorly formatted text.
2. Correct all spelling mistakes.
3. Improve abbreviations such as "u", "hv", etc.
4. Return both the original and corrected versions.

{format_instructions}

User Input:
{user_input}

YOUR RESPONSE:
"""


# ============================================================
# 8. Create the PromptTemplate object
# ============================================================

prompt = PromptTemplate(
    
    # user_input must be provided when formatting the prompt
    input_variables=["user_input"],

    # The actual prompt template
    template=template,

    # format_instructions is already known,
    # so we pass it as a partial variable
    partial_variables={
        "format_instructions": format_instructions
    }
)


# ============================================================
# 9. Format the prompt with user input
# ============================================================

prompt_value = prompt.format(
    user_input="""
Welcome to Londan! I hope u hv a great time her.
The wether is nice and the foody is delicious.
Don't forgat to visit the Big Ban and the Towar of Londan.
Enjoy your stey!
"""
)


# ============================================================
# 10. Send the formatted prompt to the LLM
# ============================================================

llm_output = llm.invoke(prompt_value)


# Print the complete AIMessage object
print("\nLLM Output:")
print(llm_output)


# ============================================================
# 11. Parse the LLM's text response
# ============================================================

# llm_output is an AIMessage object
# llm_output.content contains the actual text response
#
# output_parser.parse() converts the structured text response
# into a Python dictionary
result = output_parser.parse(llm_output.content)


# ============================================================
# 12. Access the parsed result
# ============================================================

print("\nParsed Result:")
print(result)

# Access individual dictionary values using their keys
print("\nBad String:")
print(result["bad_string"])

print("\nGood String:")
print(result["good_string"])