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
    groq_api_key = groq_api_key
)


template = """
%INSTRUCTIONS:

You are a helpful assistant who understands the context of a conversation and can provide summarized 
relevant information based on the user's input.

Explain me as if i'm illiterate

The summary should be concise, clear, and directly related to the user's request, make it as simple as possible.

The response should not cross more than 5 sentences.

%TEXT:

{text}
"""

prompt = PromptTemplate(
    input_variables=["text"],
    template=template
)



confusing_text = """
In the examples of complex sentences below, the dependent clause comes first. 
Notice that the dependent clause begins with a subordinating conjunction (words like since, because, while) 
and that the clauses are separated by a comma:
Because he was late again, he would be docked a day’s pay.
While I am a passionate basketball fan, I prefer football.
Although she was considered smart, she failed all her exams.
Whenever it rains, I like to wear my blue coat.
"""


prompt_value = prompt.format(text=confusing_text)


response = llm.invoke(prompt_value)

print("Response:")
print(response.content)
