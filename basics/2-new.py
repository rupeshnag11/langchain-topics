import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model_name = "llama-3.1-8b-instant"

llm = ChatGroq(
    model=model_name,
    temperature=0.0,
    max_tokens=1000,
    groq_api_key=groq_api_key
)

prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Give me a professional and good tweet about {topic1} and {topic2}."
)

chain = prompt | llm

response = chain.invoke({
    "topic1": "Hyderabad",
    "topic2": "Charminar"
})

print(response.content)
