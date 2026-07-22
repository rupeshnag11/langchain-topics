# The Professor's Guide to SQL Agents
*An Academic and Practical Lecture on Bridging Natural Language and Structured Databases using LLMs*

Welcome back, class! Today, we are continuing our deep dive into **Natural Language Interfaces to Databases (NLIDB)**. I have saved this guide directly in your project workspace so you can keep it handy. 

In this session, we will expand our understanding of the notebook `1-sql-agents.ipynb` line by line, explore real-world analogies, inspect the backend mechanics, and discuss a critical architectural distinction: **SQL Chains vs. SQL Agents**.

---

## 1. High-Level Backend Flow (System Architecture)

Here is how the data flows behind the scenes:

```mermaid
sequenceDiagram
    autonumber
    actor User as User (Human)
    participant LC as LangChain (Orchestrator)
    participant LLM as Llama-3 (Brain)
    participant DB as SQLite (Data Store)
    participant VecDB as Vector DB (Fuzzy Spell Checker)

    User->>LC: "How many employees do we have?"
    Note over LC,VecDB: Step A: Spell Checking (Proper Noun Lookup)
    LC->>VecDB: Fuzzy check names/titles (retriever_tool)
    VecDB-->>LC: Correct terms (if any)
    
    Note over LC,LLM: Step B: SQL Generation Prompt
    LC->>LLM: Prompt = Database Schema (Tables, Columns) + User Question
    LLM-->>LC: Generated SQL: "SELECT COUNT(*) FROM Employee;"
    
    Note over LC,DB: Step C: Execution
    LC->>DB: Execute: "SELECT COUNT(*) FROM Employee;"
    DB-->>LC: Raw Output: [(8,)]
    
    Note over LC,LLM: Step D: Conversational Synthesis
    LC->>LLM: Prompt = Original Question + SQL + Raw Output (8)
    LLM-->>LC: Natural response: "There are 8 employees in the database."
    LC->>User: "There are 8 employees in the database."
```

---

## 2. Detailed Code-by-Code Explanation

Let's dissect each cell in the notebook.

### Part A: Gathering the Tools & Establishing the Connection

#### Code Block 1: Imports
```python
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_huggingface import HuggingFaceEmbeddings
```
*   **Layman's Explanation**: We are importing Python libraries. We need tools to read configuration settings, communicate with a fast language model (Groq/Llama), hook up our database, and parse sentence meanings (embeddings).
*   **Real-World Analogy**: Think of this as a manager gathering their team before starting a project: a translator (Groq), a filing clerk (SQLDatabase), and a dictionary (Embeddings).
*   **Backend Mechanics**: 
    *   `load_dotenv` reads a `.env` file to populate environment variables.
    *   `ChatGroq` configures an HTTP client to send REST requests to the Groq API.
    *   `SQLDatabase` leverages **SQLAlchemy** to inspect table structures, column names, and data types automatically.

---

#### Code Block 2: Initializing the Language Model (LLM)
```python
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
    groq_api_key=groq_api_key
)
```
*   **Layman's Explanation**: We fetch our API key. We then build our AI engine, choosing Llama 3.1 8B. We set the `temperature` to `0.7` to balance factual correctness with conversational flow.
*   **Real-World Analogy**: Tuning a radio station. We set the dial (temperature and model parameters) so the broadcast sounds clear and professional.
*   **Backend Mechanics**: This constructs a configuration object. When invoked, it sends payload data containing your text prompt, temperature parameters, and key headers directly to Groq's hosting servers.

---

#### Code Block 3: Connecting to the SQLite Database
```python
db_path = "sqlite:///./chinook.db"
db = SQLDatabase.from_uri(db_path)

print(db.dialect)
print(db.get_usable_table_names())
```
*   **Output**:
    ```text
    sqlite
    ['Album', 'Artist', 'Customer', 'Employee', 'Genre', 'Invoice', 'InvoiceLine', 'MediaType', 'Playlist', 'PlaylistTrack', 'Track']
    ```
*   **Layman's Explanation**: We point to a database file (`chinook.db`) containing sales and music tracks. We print the database type (`sqlite`) and the list of available folders (tables) in the database.
*   **Real-World Analogy**: Unlocking a digital filing cabinet and reading the index cards on the front of each drawer.
*   **Backend Mechanics**: The database driver issues metadata queries to the database engine. In SQLite, this involves inspecting the `sqlite_master` table to list user tables and schema structures.

---

### Part B: The Translation Phase (SQL Query Generation)

#### Code Blocks 4 & 5: Creating the SQL Query Chain
```python
from langchain_classic.chains import create_sql_query_chain

chain = create_sql_query_chain(
    llm=llm,
    db=db,
)

question = "How many employees are there in the database?"
response = chain.invoke({"question": question})
```
*   **Output**:
    ```text
    Question: How many employees are there in the database?
    Response: Question: How many employees are there in the database?
    SQLQuery: SELECT COUNT(*) FROM "Employee"
    ```
*   **Layman's Explanation**: We construct a basic translation pipeline. We give it a human question, and it returns a SQL query: `SELECT COUNT(*) FROM "Employee"`.
*   **Backend Mechanics**: Under the hood, `create_sql_query_chain` generates a system prompt. It extracts the schema for the tables (e.g. `CREATE TABLE Employee (EmployeeId INTEGER, FirstName TEXT...)`) and injects it into the prompt. The LLM processes this schema alongside the question to output the correct SQL query block.

---

#### Code Block 8: Executing Queries Manually
```python
try:
    sql = response.split("SQLQuery:")[-1].strip().rstrip(";") if "SQLQuery:" in response else response
    result = db.run(sql)
    print("DB result:", result)
except Exception as e:
    print(f"[db.run] SQL execution failed: {e}")
```
*   **Layman's Explanation**: The model outputs a conversational response containing `SQLQuery: ...`. We write code to strip away the conversational fluff and isolate the SQL statement, then run it directly on the database.
*   **Backend Mechanics**: `db.run()` compiles and runs the SQL statement on the database engine. It returns a string representing the row result (e.g. `[(8,)]`).

---

### Part C: Automating the Pipeline with LCEL

#### Code Block 10: The Parser Function
```python
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.tools import tool

@tool
def parse(query_string: str) -> str:
    """Parse the SQL query string and return a human-readable explanation."""
    if "SQLQuery:" in query_string:
        sql = query_string.split("SQLQuery:")[-1].strip()
        return sql.rstrip(";").strip()
    
    splitted_string = query_string.split(":")
    if len(splitted_string) >= 2:
        query = splitted_string[1].strip()
    else:
        query = query_string
    return query.rstrip(";").strip()
```
*   **Layman's Explanation**: This is an automated cleaner. If the model outputs metadata or notes, this code strips it away, leaving a clean SQL statement.
*   **Backend Mechanics**: The `@tool` decorator wraps the Python function inside a LangChain `Runnable` interface, allowing it to easily chain with other LangChain tools using the pipe (`|`) syntax.

---

#### Code Block 13: The Automated Run Pipeline
```python
execute_query = QuerySQLDatabaseTool(db=db)
write_query = create_sql_query_chain(llm=llm, db=db)

chain = write_query | parse | execute_query

response = chain.invoke({"question": question})
```
*   **Layman's Explanation**: We combine our components. The question goes to the query writer, the query goes to the cleaner, and the clean query goes to the database executor.
*   **Backend Mechanics**: The `|` operator implements **functional pipelining**. The output of the left component is automatically passed as the input to the right component.

---

### Part D: Synthesizing the Final Conversational Answer

#### Code Block 15 & 16: The Complete Query-Answer Loop
```python
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
```
*   **Layman's Explanation**: This builds the full cycle. Instead of returning raw database tuples like `[(8,)]`, we feed the question, the generated SQL, and the database result back into the model to construct a natural sentence like *"There are 8 employees in the database."*
*   **Backend Mechanics**: 
    1.  `RunnablePassthrough.assign` computes variables in parallel.
    2.  The dictionary containing the question, query, and result is formatted into the `answer_prompt`.
    3.  The formatted prompt is sent to the LLM.
    4.  `StrOutputParser` extracts the final string response.

---

### Part E: Handling Typos & Synonyms (Vector Embeddings)

#### Code Blocks 22 & 23: The Semantic Search Spell Checker
```python
from langchain_classic.agents.agent_toolkits import create_retriever_tool
from langchain_community.vectorstores import FAISS

vector_database = FAISS.from_texts(
    texts=artists + albums,
    embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'),
)

retriever = vector_database.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 2}
)
```
*   **Layman's Explanation**: A user might search for *"Alis Chains"* (a typo for the band *Alice In Chains*). A traditional database lookup will find nothing. Here, we extract all artist and album names, convert them into mathematical vectors (numerical footprints), and index them.
*   **Real-World Analogy**: An alphabetical phone book compared to a concept map. The concept map groups things that sound alike or mean the same thing, letting you find the right entry even if you misspell it.
*   **Backend Mechanics**: 
    *   `HuggingFaceEmbeddings` maps words into high-dimensional vector representations.
    *   `FAISS` computes similarity distances (such as Cosine or L2 Euclidean distance) to find the closest match.
    *   When the user types *"Alis Chains"*, the similarity search finds `"Alice In Chains"` as the nearest matching vector.

---

## 3. Deep Dive: SQL Chains vs. SQL Agents (New Concept!)

While the notebook relies primarily on **SQL Chains**, it is titled "Understanding SQL Agents". Let's explain the difference between the two concepts:

| Feature | SQL Chain (Used in this Notebook) | SQL Agent (Advanced Pattern) |
| :--- | :--- | :--- |
| **Control Flow** | **Linear & Pre-determined**: Step A $\rightarrow$ Step B $\rightarrow$ Step C. | **Dynamic & Loop-based**: The LLM decides what step to take next in a loop. |
| **Error Handling** | **Fragile**: If the LLM generates bad SQL, the pipeline crashes. | **Resilient**: If a query fails, the agent reads the error message and rewrites the SQL. |
| **Tool Usage** | **Fixed**: Can only run the predefined database execution tool. | **Flexible**: Can search vector DBs, read tables, or lookup columns dynamically. |

### How an Agent Solves SQL Errors Under the Hood

When using a true SQL Agent (e.g. `create_sql_agent` in LangChain), the execution flow is conversational:

1. **User asks**: "Who is our top spending customer from Canada?"
2. **Agent thinks**: I need to query the database. First, let me list the tables.
3. **Agent action**: Calls `list_tables_tool`.
4. **Tool response**: `['Customer', 'Invoice', ...]`
5. **Agent thinks**: I need columns from `Customer` and `Invoice`. Let me check schema of these tables.
6. **Agent action**: Calls `schema_lookup_tool` for `Customer` and `Invoice`.
7. **Tool response**: `Customer` has `CustomerId, Country`. `Invoice` has `CustomerId, Total`.
8. **Agent thinks**: I will write a join query.
9. **Agent action**: Calls `execute_query_tool` with `SELECT Name FROM Customer JOIN Invoice...`
10. **Database response**: `Error: no such column: Customer.Name` (because the column is actually `FirstName` and `LastName`).
11. **Agent thinks**: Ah, I made a mistake. The column `Name` does not exist. Let me use `FirstName` and `LastName` instead.
12. **Agent action**: Calls `execute_query_tool` with the corrected query.
13. **Database response**: `[('John', 'Smith')]`
14. **Agent final answer**: "The top spending customer from Canada is John Smith."

This self-correcting loop is what makes **Agents** much more powerful than simple linear **Chains**.
