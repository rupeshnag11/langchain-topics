#### LangChain Conversational Memory with Agent

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.agents import load_tools, initialize_agent


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

tools = load_tools(
    [
        "llm-math",
    ],
    llm=llm
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="conversational-react-description",
    verbose=True,
    memory=memory
)


output1 = agent.run(
    "What is the square root of 144? "
    "Then, what is the result of multiplying that by 10?"
)

print(output1)



output2 = agent.run(
    """
    add 5 to the previous result,
    and square root the final result.
    """)

print(output2)


