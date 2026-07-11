import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model_name = "llama-3.1-8b-instant"

llm = ChatGroq(
    model=model_name,
    temperature=0.0,
    max_tokens=1000,
    groq_api_key=groq_api_key
)

template1 = """
    You're a helpful assistant, who has good and proficient english writing skills.
    Write a Blog Outline for the following topic:
    Topic: {topic}
"""

template2 = """
    You're a helpful assistant, who has good and proficient english writing skills.
    Write a Blog Post based on the following outline:
    Outline: {outline}
"""

template3 = """
    You're a helpful assistant, who has good and proficient language skills including Translating.
    
    Translate the following text to Telugu:
    {blog_post}
"""

prompt1 = PromptTemplate(
    input_variables=["topic"],
    template=template1
)

prompt2 = PromptTemplate(
    input_variables=["outline"],
    template=template2
)

prompt3 = PromptTemplate(
    input_variables=["blog_post"],
    template=template3
)

chain1 = LLMChain(
    llm=llm,
    prompt=prompt1,
    output_key="outline"
)

chain2 = LLMChain(
    llm=llm,
    prompt=prompt2,
    output_key="blog_post"
)

chain3 = LLMChain(
    llm=llm,
    prompt=prompt3,
    output_key="translated_text"
)

sequential_chain = SequentialChain(
    chains=[chain1, chain2, chain3],
    input_variables=["topic"],
    output_variables=["outline", "blog_post", "translated_text"],
    verbose=True
)

response = sequential_chain.invoke({
    "topic": "The Future of Artificial Intelligence"
})

print("Outline:")
print(response["outline"])

print("\nBlog Post:")
print(response["blog_post"])

print("\nTranslated Text:")
print(response["translated_text"])