import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"   # Updated model
)

# Prompt template
prompt_template = PromptTemplate(
    input_variables=["topic1", "topic2"],
    template="""
Suggest a detailed comparison between

{topic1}

and

{topic2}
"""
)

# Format prompt
prompt = prompt_template.format(
    topic1="Sony",
    topic2="Bravia XR"
)

# Invoke model
response = llm.invoke(prompt)

# Print response
print(response.content)