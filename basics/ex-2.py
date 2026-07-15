# Import os to access environment variables
import os

# Import load_dotenv to load the GROQ_API_KEY from the .env file
from dotenv import load_dotenv

# Import PdfReader to read and extract text from the PDF
from pypdf import PdfReader

# Import ChatGroq to use the Groq language model
from langchain_groq import ChatGroq

# Import PromptTemplate to create a structured prompt
from langchain_core.prompts import PromptTemplate


# ---------------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# ---------------------------------------------------------

# Load variables from the .env file
load_dotenv(override=True)


# ---------------------------------------------------------
# 2. GET GROQ API KEY
# ---------------------------------------------------------

# Get the GROQ_API_KEY from the .env file
groq_api_key = os.getenv("GROQ_API_KEY")

# Check whether the API key is available
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")


# ---------------------------------------------------------
# 3. DEFINE THE PDF FILE PATH
# ---------------------------------------------------------

# Use r before the string because this is a Windows file path
pdf_path = r"C:\Users\Rupesh\Desktop\AIML\AIML\Advanced\sony-blr-agentic-ai-practices\sony-blr-agentic-ai-practices\lc-training-data\rag-docs\embedded-software-engineer-resume-example.pdf"


# ---------------------------------------------------------
# 4. READ THE PDF USING PdfReader
# ---------------------------------------------------------

# Create a PdfReader object for the given PDF
reader = PdfReader(pdf_path)

# Create an empty string to store the complete resume text
resume_text = ""

# Loop through every page in the PDF
for page in reader.pages:

    # Extract text from the current page
    page_text = page.extract_text()

    # Add the text only if extraction was successful
    if page_text:
        resume_text += page_text + "\n"


# ---------------------------------------------------------
# 5. INITIALIZE THE GROQ MODEL
# ---------------------------------------------------------

# Create the Groq LLM object
llm = ChatGroq(
    model="llama-3.1-8b-instant",

    # Lower temperature gives more factual and consistent answers
    temperature=0.2,

    # Maximum number of tokens in the generated response
    max_tokens=1000,

    # Groq API key
    groq_api_key=groq_api_key
)


# ---------------------------------------------------------
# 6. CREATE THE PROMPT TEMPLATE
# ---------------------------------------------------------

# Instruct the model to extract specific information
# from the resume
template = """
You are an AI assistant specialized in analyzing resumes.

Read the resume text provided below and extract and summarize
the following information:

1. Name:
   - Full name of the person.

2. Qualifications / Education:
   - Degree or qualification.
   - College or university name.
   - Graduation year, if available.

3. Work Experience:
   For each job, provide:
   - Job title or role.
   - Company or organization name.
   - Employment duration, if available.
   - Key responsibilities and work performed.

4. Overall Experience Summary:
   - Summarize the person's total professional experience.
   - Mention the main technologies, skills, and areas of expertise.

Important instructions:
- Use only information available in the resume.
- Do not invent or assume any missing information.
- If information is unavailable, clearly say "Not mentioned".
- Keep the answer clear, concise, and structured.

Resume Text:

{resume_text}

Resume Summary:
"""


# ---------------------------------------------------------
# 7. CREATE THE PROMPT
# ---------------------------------------------------------

# Create the PromptTemplate object
prompt = PromptTemplate(
    input_variables=["resume_text"],
    template=template
)


# ---------------------------------------------------------
# 8. CREATE THE CHAIN
# ---------------------------------------------------------

# Connect the prompt template to the Groq model
chain = prompt | llm


# ---------------------------------------------------------
# 9. RUN THE CHAIN
# ---------------------------------------------------------

# Send the extracted resume text to the model
response = chain.invoke({
    "resume_text": resume_text
})


# ---------------------------------------------------------
# 10. PRINT THE RESULT
# ---------------------------------------------------------

print("\n========== RESUME SUMMARY ==========\n")

# ChatGroq returns an AIMessage object
# .content contains the actual generated answer
print(response.content)