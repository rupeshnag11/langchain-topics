import os
import re
import time

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase

load_dotenv(override=True)

groq_api_key = os.getenv('GROQ_API_KEY')

if not groq_api_key:
    raise ValueError('GROQ_API_KEY environment variable is not set.')

temperature = 0.7
max_tokens = 1500
model_name = 'llama-3.1-8b-instant'

llm = ChatGroq(
    model=model_name,
    temperature=temperature,
    max_tokens=max_tokens,
    groq_api_key=groq_api_key,
    request_timeout=60,
)

import pathlib
db_path = f"sqlite:///{pathlib.Path(__file__).parent / 'chinook.db'}"
db = SQLDatabase.from_uri(db_path)

print(db.dialect)
print(db.get_usable_table_names())




from langchain_classic.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.tools import tool


@tool
def parse(query_string: str) -> str:
    """Parse the SQL query string and return just the SQL query."""

    # Handle full LLM output like: "Question: ...\nSQLQuery: SELECT ..."
    if "SQLQuery:" in query_string:
        sql = query_string.split("SQLQuery:")[-1].strip()
        return sql.rstrip(";").strip()

    # Fallback: strip a "QUERY: " prefix if present
    splitted_string = query_string.split(":")
    if len(splitted_string) >= 2:
        query = splitted_string[1].strip()
    else:
        query = query_string

    return query.rstrip(";").strip()


def invoke_with_retry(chain, inputs, max_retries=5):
    """Invoke a LangChain chain, retrying on Groq rate limits and transient network errors."""
    from groq import RateLimitError, APIConnectionError, APITimeoutError
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except RateLimitError as e:
            match = re.search(r'try again in (\d+\.?\d*)s', str(e))
            wait = float(match.group(1)) + 5 if match else 30
            print(f"[Rate limited] Waiting {wait:.0f}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(wait)
        except (APIConnectionError, APITimeoutError) as e:
            wait = 10 * (attempt + 1)  # 10s, 20s, 30s... progressive backoff
            print(f"[Network error] {type(e).__name__}. Waiting {wait}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {max_retries} retries.")


chain = create_sql_query_chain(
    llm=llm,
    db=db,
)
question = "How many employees are there in the database?"
response = invoke_with_retry(chain, {"question": question})

print("Question:", question)
print("Response:", response)


question = "which country's customers have spent the most?"

chain = create_sql_query_chain(
    llm=llm,
    db=db,
)

time.sleep(5)
response = invoke_with_retry(chain, {"question": question})

print("Question:", question)
print("Response:", response)



try:
    result = db.run(parse.invoke(response))
    print("DB result:", result)
except Exception as e:
    print(f"[db.run] SQL execution failed (LLM may have generated invalid SQL): {e}")

chain.get_prompts()[0].pretty_print()



print("parse (no prefix):", parse.invoke("""
      SELECT COUNT("EmployeeId") AS EmployeeCount
        FROM Employee
      """))


print("parse (QUERY: prefix):", parse.invoke("""
      QUERY: SELECT COUNT("EmployeeId") AS EmployeeCount
        FROM Employee
      """))

execute_query = QuerySQLDatabaseTool(db=db)
write_query = create_sql_query_chain(
    llm=llm,
    db=db,
)
chain = write_query | parse | execute_query
question = "How many employees are there in the database?"
response = invoke_with_retry(chain, {"question": question})

print("Question:", question)
print("Response:", response)



question = "which country's customers have spent the most?"

execute_query = QuerySQLDatabaseTool(db=db)
write_query = create_sql_query_chain(
    llm=llm,
    db=db,
)
chain = write_query | parse | execute_query

time.sleep(5)
response = invoke_with_retry(chain, {"question": question})

print("Question:", question)
print("Response:", response)


from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

answer_prompt = PromptTemplate.from_template(
    """
        Given the following user question, corresponding SQL Query, and SQL Result, answer the user question in a concise manner.
        
        User Question: {question}
        SQL Query: {query}
        SQL Result: {result}
        Answer: 
    """
)

chain = RunnablePassthrough \
    .assign(query=write_query) \
    .assign(result=itemgetter("query") | parse | execute_query) | \
    answer_prompt | llm | StrOutputParser()


question = "How many employees are there in the database?"
response = invoke_with_retry(chain, {"question": question})

print("Question:", question)
print("Response:", response)


question = "which country's customers have spent the most?"
time.sleep(5)
response = invoke_with_retry(chain, {"question": question})

print("Question:", question)
print("Response:", response)


import ast


def query_as_list(database, query):
    """
    Execute a SQL query and return the results as a list of dictionaries.
    """
    result = database.run(query)
    result = [el for sub in ast.literal_eval(result) for el in sub if el]
    result = [re.sub(r"\b\d+\b", "", string).strip() for string in result]

    return list(set(result))

artists = query_as_list(db, "SELECT Name FROM Artist")
albums = query_as_list(db, "SELECT Title FROM Album")

print("artists[5]:", artists[5])

print("albums[:5]:", albums[:5])

from langchain_classic.agents.agent_toolkits import create_retriever_tool
from langchain_community.vectorstores import FAISS
# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEmbeddings

vector_database = FAISS.from_texts(
    texts=artists + albums,
    embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'),
)

retriever = vector_database.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 2}
)

description = '''
    use to lookup values to filter on.
    input is an approximate spelling of the valid and proper nouns.
    Use the noun most similar to the input.
    If the input is not a valid noun, return an empty string.
'''

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name='retriever',
    description=description,
)

response = retriever_tool.invoke("Alis Chains")

print("Input: Alis Chains")
print("Response:", response)

response = retriever_tool.invoke("Do we have any artists by named Alis Chains")

print("Input: Alis Chains")
print("Response:", response)