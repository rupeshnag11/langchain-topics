import os
import base64

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import LLMChain, SequentialChain

# Load environment variables from .env
load_dotenv()

# Get Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Use a vision-capable Groq model
model_name = "meta-llama/llama-4-scout-17b-16e-instruct"

# Initialize Groq chat model
llm = ChatGroq(
    model=model_name,
    temperature=0.0,
    max_tokens=1000,
    groq_api_key=groq_api_key
)


def encode_base64_image(image_path: str) -> str:
    """
    Read an image file and convert it to a Base64 string.
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    return encoded_string


# Image path — use r before the string for Windows paths
image_path = r"C:\Users\Rupesh\Downloads\us-home-mortgage-market-size.png"

# Convert image to Base64
image_base64 = encode_base64_image(image_path)


# Create multimodal message
message = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant that can analyze images "
            "and provide insights."
        )
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Analyze the following image and provide insights "
                    "on the US mortgage rate trends."
                )
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                }
            }
        ]
    }
]


# Send message to Groq model
response = llm.invoke(message)


# Print response
print("\nSummary of US Mortgage Rate Trends:")
print(response.content)