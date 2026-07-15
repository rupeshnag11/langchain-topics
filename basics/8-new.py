import os
from dotenv import load_dotenv
from typing import Optional, Sequence

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# Load environment variables from .env file
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")


# Define structured output schema
class Person(BaseModel):
    name: str = Field(..., description="The name of the person")
    age: Optional[int] = Field(None, description="The age of the person")
    occupation: Optional[str] = Field(
        None,
        description="The occupation of the person"
    )
    years_of_experience: Optional[int] = Field(
        None,
        description="Years of experience in their field"
    )
    skills: Optional[list[str]] = Field(
        None,
        description="List of skills of the person"
    )


# Initialize LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=1500,
    groq_api_key=groq_api_key
)


# Add structured output schema to the model
structured_llm = llm.with_structured_output(Person)


# Create prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a world-class algorithm for extracting "
            "structured data from text."
        ),
        (
            "human",
            "Use the given schema to extract information "
            "from the following input:\n\n{input}"
        )
    ]
)


# Create chain
chain = prompt | structured_llm


# Input text
request = """
Ramkumar is 50 years old, and he works as a Software Engineer at Google.
He has 20 years of experience in the field and is skilled in Python, Java, and C++.
"""


# Invoke chain
person = chain.invoke({"input": request})


# Print complete response
print("Response:", person)


# Access structured fields directly
print("\nExtracted Data:")
print(f"Name: {person.name}")
print(f"Age: {person.age}")
print(f"Occupation: {person.occupation}")
print(f"Years of Experience: {person.years_of_experience}")
print(f"Skills: {person.skills}")



class People(BaseModel):
    people: Sequence[Person] = Field(
        ...,
        description="List of all people extracted from the text."
    )


# Use a more capable model for nested structured output (People contains list of Person)
# llama-3.1-8b-instant struggles with nested schemas; llama-3.3-70b-versatile handles it reliably
llm_large = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=1500,
    groq_api_key=groq_api_key
)

# Create a separate structured LLM and chain for multiple-person extraction
structured_llm_people = llm_large.with_structured_output(People)

chain_people = prompt | structured_llm_people


request = """
    Ramkumar is 50 Years Old, and he works as a Software Engineer at Google.
    He has 20 years of experience in the field and is skilled in Python, Java, and C++.
    
    Lakshmi is 41 Years Old, and he works as a Software Engineer at Microsoft.
    He has 20 years of experience in the field and is skilled in Python, Java, and C++.
    
    Both of them are skilled in LangChain and OpenAI.
"""


response = chain_people.invoke({"input": request})

print(response)


if isinstance(response, People):
    print("Extracted Data:")

    for person in response.people:
        print(f"Name: {person.name}")
        print(f"Age: {person.age}")
        print(f"Occupation: {person.occupation}")
        print(f"Years of Experience: {person.years_of_experience}")
        print(f"Skills: {person.skills}")
        print("-" * 20)