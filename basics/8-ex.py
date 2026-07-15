# credit card image report-> only transaction values.abs

import os
import base64
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# Path of the credit card transaction image
image_path = r"C:\Users\Rupesh\Downloads\credit.png"

# Convert image to Base64
with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

# Initialize a vision-capable Groq model
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    groq_api_key=groq_api_key
)

# Create message containing text instructions and image
message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": """
Analyze the credit card transaction image.

First, extract and print all transaction values visible in the image, such as:
- Customer name
- Card number or last four digits
- Transaction amount
- Currency
- Merchant name
- Transaction date
- Transaction time
- Location
- Transaction type
- Any other available details

Then analyze the transaction and provide:

1. Transaction Summary
2. Risk Level: Low, Medium, or High
3. Suspicious Factors
4. Detailed Analysis
5. Recommended Action

Do not invent missing information. If a value is not visible in the image,
write "Not available".
"""
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encoded_image}"
            }
        }
    ]
)

# Send image to Groq vision model
response = llm.invoke([message])

# Print result
print("=" * 60)
print("CREDIT CARD TRANSACTION EXTRACTION AND ANALYSIS")
print("=" * 60)

print(response.content)