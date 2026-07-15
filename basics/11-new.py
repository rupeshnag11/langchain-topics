#langchain -QA simple

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv(override=True)

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

temperature = 0.7
max_tokens = 1500
model_name = "llama-3.3-70b-versatile"

llm = ChatGroq(
    model=model_name,
    temperature=temperature,
    max_tokens=max_tokens,
    groq_api_key=groq_api_key
)

context = """
    Ramkumar is 50 Years Old.
    Bob is 29 Years Old.
    Ramkumar is a software engineer.
    Bob is a data scientist.
    Ramkumar lives in Chennai.
    Bob lives in Bangalore.
"""

question = "Who is older, Ramkumar or Bob?"
prompt = context + "\n\n" + question

output = llm.invoke(prompt)
print(f"Question: {question}")
print(f"Answer: {output.content.strip()}")


question  = "where does Ramkumar live?"
prompt = context + "\n\n" + question
output = llm.invoke(prompt)
print(f"Question: {question}")
print(f"Answer: {output.content.strip()}")

question = "What is Bob's profession?"
prompt = context + "\n\n" + question
output = llm.invoke(prompt)
print(f"Question: {question}")
print(f"Answer: {output.content.strip()}")