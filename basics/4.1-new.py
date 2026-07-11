import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]
model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
temperature = 0.1
max_tokens = 1000

llm = ChatGroq(
    model = model_name,
    temperature=temperature,
    max_tokens=max_tokens,
    groq_api_key = groq_api_key
)


with open(r"C:\Users\Rupesh\Desktop\AIML\AIML\Advanced\sony-blr-agentic-ai-practices\sony-blr-agentic-ai-practices\lc-training-data\good.txt", "r") as file:
    text = file.read()
    
print(text[:100])

number_of_tokens = llm.get_num_tokens(text)

print(f"There are {number_of_tokens} Token(s) in the file!")


text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n"],
    chunk_size = 3000,
    chunk_overlap = 300
)




documents = text_splitter.create_documents([text])

print(f"Now, you have {len(documents)} documents instead of 1 document!")



chain = load_summarize_chain(
    llm = llm,
    chain_type = "map_reduce"
)


output = chain.invoke(documents)

print(output)
