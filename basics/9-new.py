#### LangChain - Agents with Tools (Minimal)

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

load_dotenv(override=True)

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("groq_API_KEY environment variable is not set.")

temperature = 0.7
max_tokens = 1500
model_name = "llama-3.3-70b-versatile"

llm = ChatGroq(
    model=model_name,
    temperature=temperature,
    max_tokens=max_tokens,
    groq_api_key=groq_api_key
)

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiplies two integers."""
    return first_int * second_int

@tool
def add(first_int: int, second_int: int) -> int:
    """Adds two integers."""
    return first_int + second_int

@tool
def exponentize(base: int, exponent: int) -> int:
    """Raises base to the power of exponent."""
    return base ** exponent

@tool
def get_weather(city: str) -> str:
    """Fetches the current weather for a given city."""
    # This is a placeholder function. In a real application, you would implement
    # an API call to a weather service.

    weather_data = {
        "New York": "Sunny, 25°C",
        "Los Angeles": "Cloudy, 22°C",
        "Chicago": "Rainy, 18°C",
        "Houston": "Sunny, 30°C",
        "Phoenix": "Hot, 35°C"
    }
    
    return weather_data.get(city, "Weather data not available for this city.")  


tools = [multiply, add, exponentize, get_weather]

# Equivalent to hub.pull("hwchase17/openai-tools-agent") — defined locally
# to avoid the deprecated langchainhub network call.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)


response = agent_executor.invoke({
    "input": 
        """
        What is 2 raised to the power of 3, then multiplied by 4, and finally added to 5?
        And also, what is the weather in New York?
        """
})

print(response)