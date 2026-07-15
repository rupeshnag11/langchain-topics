#### LangChain Conversational Memory with Agent (Modern - LangGraph)

import os
import math
import numexpr
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv(override=True)

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")


# --------------------------------------------------
# 2. Initialize LLM
# --------------------------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_tokens=1500,
    groq_api_key=groq_api_key
)


# --------------------------------------------------
# 3. Define tools
#    Custom math tool using numexpr — replaces deprecated
#    langchain_community load_tools(["llm-math"])
# --------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Supports: +, -, *, /, **, sqrt(), sin(), cos(), log(), etc.
    Example inputs: 'sqrt(144)', '12 * 10', '125 ** 0.5'
    """
    try:
        # numexpr evaluates math expressions safely and fast
        result = numexpr.evaluate(expression.strip())
        return str(float(result))
    except Exception:
        # Fall back to Python's math module for functions like sqrt()
        try:
            result = eval(expression, {"__builtins__": {}}, vars(math))
            return str(float(result))
        except Exception as e:
            return f"Error evaluating expression: {e}"


tools = [calculator]


# --------------------------------------------------
# 4. Set up memory
#    MemorySaver  = modern replacement for ConversationBufferMemory
# --------------------------------------------------

memory = MemorySaver()


# --------------------------------------------------
# 5. Create agent
#    create_agent (langchain.agents) = modern replacement
#    for the deprecated langgraph.prebuilt.create_react_agent
# --------------------------------------------------

agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=memory
)


# --------------------------------------------------
# 6. Thread config ties all queries into one session
#    (replaces memory_key="chat_history")
# --------------------------------------------------

config = {"configurable": {"thread_id": "session-1"}}


# --------------------------------------------------
# 7. Query 1
# --------------------------------------------------

output1 = agent.invoke(
    {
        "messages": [
            ("human", "What is the square root of 144? Then, what is the result of multiplying that by 10?")
        ]
    },
    config=config
)

print("\n========== OUTPUT 1 ==========\n")
print(output1["messages"][-1].content)


# --------------------------------------------------
# 8. Query 2 (agent remembers Query 1's result via MemorySaver)
# --------------------------------------------------

output2 = agent.invoke(
    {
        "messages": [
            ("human", "Add 5 to the previous result, and then square root the final result.")
        ]
    },
    config=config
)

print("\n========== OUTPUT 2 ==========\n")
print(output2["messages"][-1].content)
