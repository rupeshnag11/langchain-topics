# Import os to access environment variables
import os

# Import load_dotenv to load the Groq API key from the .env file
from dotenv import load_dotenv

# Import PyPDFLoader to read and extract text from a PDF file
from langchain_community.document_loaders import PyPDFLoader

# Import ChatGroq to use Groq-hosted language models
from langchain_groq import ChatGroq

# Import PromptTemplate to create a structured prompt
from langchain_core.prompts import PromptTemplate


# ---------------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# ---------------------------------------------------------

# Load environment variables from the .env file
# override=True means values in .env override existing
# environment variables with the same name
load_dotenv(override=True)


# ---------------------------------------------------------
# 2. GET GROQ API KEY
# ---------------------------------------------------------

# Read the GROQ_API_KEY from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

# Stop the program if the API key is not found
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")


# ---------------------------------------------------------
# 3. DEFINE THE PDF FILE PATH
# ---------------------------------------------------------

# Use a raw string (r"...") for Windows file paths
# This prevents backslashes like \U or \A from being treated
# as special escape characters
pdf_path = r"C:\Users\Rupesh\Desktop\AIML\AIML\Advanced\sony-blr-agentic-ai-practices\sony-blr-agentic-ai-practices\lc-training-data\rag-docs\embedded-software-engineer-resume-example.pdf"


# ---------------------------------------------------------
# 4. LOAD THE PDF
# ---------------------------------------------------------

# Create a PyPDFLoader object using the PDF file path
loader = PyPDFLoader(pdf_path)

# Load the PDF
# Each page is returned as a separate Document object
documents = loader.load()

# Print the number of pages loaded
print(f"Number of pages loaded: {len(documents)}")


# ---------------------------------------------------------
# 5. COMBINE TEXT FROM ALL PDF PAGES
# ---------------------------------------------------------

# Extract page_content from every Document object
# and combine everything into a single string
resume_text = "\n\n".join(
    document.page_content for document in documents
)


# ---------------------------------------------------------
# 6. INITIALIZE THE GROQ LANGUAGE MODEL
# ---------------------------------------------------------

# Create the ChatGroq model object
llm = ChatGroq(
    model="llama-3.1-8b-instant",

    # Lower temperature gives more focused and factual responses
    temperature=0.2,

    # Maximum number of tokens the model can generate
    max_tokens=1000,

    # Groq API key
    groq_api_key=groq_api_key
)


# ---------------------------------------------------------
# 7. CREATE THE PROMPT TEMPLATE
# ---------------------------------------------------------

# Tell the model exactly which information to extract
template = """
You are an AI assistant that analyzes resumes.

Read the resume text provided below and summarize only the following:

1. Name of the person
2. Qualifications / Education
3. Professional Experience

Instructions:
- Extract information only from the provided resume.
- Do not invent or assume any missing information.
- Keep the summary concise and clear.
- For experience, mention job titles, companies, duration,
  and key responsibilities when available.

Resume Text:
{resume_text}

Resume Summary:
"""


# ---------------------------------------------------------
# 8. CREATE THE PROMPT
# ---------------------------------------------------------

# Create a reusable PromptTemplate object
prompt = PromptTemplate(
    input_variables=["resume_text"],
    template=template
)


# ---------------------------------------------------------
# 9. CREATE THE CHAIN
# ---------------------------------------------------------

# Modern LangChain uses the pipe operator (|)
# to connect the prompt with the language model
chain = prompt | llm


# ---------------------------------------------------------
# 10. RUN THE CHAIN
# ---------------------------------------------------------

# Pass the extracted resume text to the chain
response = chain.invoke({
    "resume_text": resume_text
})


# ---------------------------------------------------------
# 11. PRINT THE SUMMARY
# ---------------------------------------------------------

# ChatGroq returns an AIMessage object
# The actual generated text is stored in .content
print("\n========== RESUME SUMMARY ==========\n")
print(response.content)