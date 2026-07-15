import os
import base64
from typing import Optional, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


# --------------------------------------------------
# 2. Define Pydantic model
# --------------------------------------------------

class CreditCardTransaction(BaseModel):

    customer_name: Optional[str] = Field(
        description="Name of the customer visible in the image"
    )

    card_last_four_digits: Optional[str] = Field(
        description="Last four digits of the credit card"
    )

    transaction_amount: Optional[float] = Field(
        description="Transaction amount"
    )

    currency: Optional[str] = Field(
        description="Currency of the transaction, for example INR or USD"
    )

    merchant_name: Optional[str] = Field(
        description="Name of the merchant"
    )

    transaction_date: Optional[str] = Field(
        description="Date of the transaction"
    )

    transaction_time: Optional[str] = Field(
        description="Time of the transaction"
    )

    location: Optional[str] = Field(
        description="Location of the transaction"
    )

    transaction_type: Optional[str] = Field(
        description="Type of transaction, such as online purchase, POS, or ATM"
    )

    risk_level: Optional[str] = Field(
        description="Fraud risk level: Low, Medium, or High"
    )

    suspicious_factors: List[str] = Field(
        default_factory=list,
        description="List of suspicious factors identified in the transaction. Return an empty list [] if there are none."
    )

    analysis: str = Field(
        description="Detailed analysis of the credit card transaction"
    )

    recommended_action: str = Field(
        description="Recommended action such as Approve, Review, or Block"
    )


# --------------------------------------------------
# 3. Initialize Groq vision model
# --------------------------------------------------

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    groq_api_key=groq_api_key
)


# --------------------------------------------------
# 4. Add Pydantic structured output
# --------------------------------------------------

structured_llm = llm.with_structured_output(CreditCardTransaction)


# --------------------------------------------------
# 5. Image path
# --------------------------------------------------

image_path = r"C:\Users\Rupesh\Downloads\credit.png"


# --------------------------------------------------
# 6. Convert image into Base64
# --------------------------------------------------

with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(
        image_file.read()
    ).decode("utf-8")


# --------------------------------------------------
# 7. Create multimodal HumanMessage
# --------------------------------------------------

message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": """
You are a credit card transaction analyst.

Analyze the provided credit card transaction image.

Your tasks are:

1. Extract all visible transaction information from the image.
2. Do not invent any missing values.
3. If information is unavailable, return null.
4. Analyze whether the transaction appears normal or suspicious.
5. Assign a risk level: Low, Medium, or High.
6. Identify suspicious factors. If there are none, return an empty list [] for suspicious_factors.
7. Provide a detailed analysis.
8. Recommend an appropriate action.

Base your analysis only on the information visible in the image.
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


# --------------------------------------------------
# 8. Invoke LangChain
# --------------------------------------------------

result = structured_llm.invoke([message])


# --------------------------------------------------
# 9. Print complete Pydantic object
# --------------------------------------------------

print("\n========== COMPLETE RESULT ==========\n")

print(result)


# --------------------------------------------------
# 10. Print individual values
# --------------------------------------------------

print("\n========== TRANSACTION DETAILS ==========\n")

print("Customer Name:", result.customer_name)
print("Card Last Four Digits:", result.card_last_four_digits)
print("Transaction Amount:", result.transaction_amount)
print("Currency:", result.currency)
print("Merchant Name:", result.merchant_name)
print("Transaction Date:", result.transaction_date)
print("Transaction Time:", result.transaction_time)
print("Location:", result.location)
print("Transaction Type:", result.transaction_type)


print("\n========== TRANSACTION ANALYSIS ==========\n")

print("Risk Level:", result.risk_level)

print("\nSuspicious Factors:")

for factor in result.suspicious_factors:
    print("-", factor)

print("\nDetailed Analysis:")
print(result.analysis)

print("\nRecommended Action:")
print(result.recommended_action)