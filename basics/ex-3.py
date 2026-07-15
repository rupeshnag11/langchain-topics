

import os
from dotenv import load_dotenv
from typing import List

from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


# --------------------------------------------------
# 2. Create Groq LLM
# --------------------------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=groq_api_key
)


# --------------------------------------------------
# 3. Sample inventory data
# Replace this with API/database data if required
# --------------------------------------------------

inventory = [
    {
        "product_name": "Laptop",
        "units_available": 25,
        "is_recent": True
    },
    {
        "product_name": "Wireless Mouse",
        "units_available": 50,
        "is_recent": True
    },
    {
        "product_name": "Mechanical Keyboard",
        "units_available": 30,
        "is_recent": True
    },
    {
        "product_name": "Old Monitor",
        "units_available": 10,
        "is_recent": False
    }
]


# --------------------------------------------------
# 4. Create tools
# --------------------------------------------------

@tool
def get_recent_products() -> List[dict]:
    """
    Get all recently added products from the inventory,
    including the number of units available for each product.
    """
    return [
        product
        for product in inventory
        if product["is_recent"]
    ]


@tool
def calculate_total_units(products: List[dict]) -> int:
    """
    Calculate the total number of available units
    across a list of products.
    """
    return sum(
        product["units_available"]
        for product in products
    )


# --------------------------------------------------
# 5. Bind tools to LLM
# --------------------------------------------------

tools = [
    get_recent_products,
    calculate_total_units
]

llm_with_tools = llm.bind_tools(tools)

tools_by_name = {
    tool.name: tool
    for tool in tools
}


# --------------------------------------------------
# 6. Ask LLM to select appropriate tools
# --------------------------------------------------

user_query = """
Get all recent products from inventory.
Calculate the total number of units available.
Then prepare an email subject and body to send to
rupeshnag2233@gmail.com.
"""

response = llm_with_tools.invoke(user_query)


# --------------------------------------------------
# 7. Execute tool calls
# --------------------------------------------------

tool_results = {}

for tool_call in response.tool_calls:

    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    selected_tool = tools_by_name[tool_name]

    result = selected_tool.invoke(tool_args)

    tool_results[tool_name] = result


# --------------------------------------------------
# 8. Get recent products
# --------------------------------------------------

recent_products = tool_results.get(
    "get_recent_products",
    get_recent_products.invoke({})
)


# Calculate total deterministically
total_units = sum(
    product["units_available"]
    for product in recent_products
)


# --------------------------------------------------
# 9. Define structured email output
# --------------------------------------------------

class EmailOutput(BaseModel):

    recipient: str = Field(
        description="Email recipient address"
    )

    subject: str = Field(
        description="Email subject"
    )

    body: str = Field(
        description="Email body containing products and total units"
    )


parser = PydanticOutputParser(
    pydantic_object=EmailOutput
)


# --------------------------------------------------
# 10. Create email-generation prompt
# --------------------------------------------------

email_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an inventory reporting assistant.

Create a professional email for the inventory team.

{format_instructions}
"""
        ),
        (
            "human",
            """
Recipient:
rupeshnag2233@gmail.com

Recent products:
{products}

Total units available:
{total_units}
"""
        )
    ]
).partial(
    format_instructions=parser.get_format_instructions()
)


# --------------------------------------------------
# 11. Create LCEL chain
# --------------------------------------------------

email_chain = email_prompt | llm | parser


# --------------------------------------------------
# 12. Generate final email
# --------------------------------------------------

email = email_chain.invoke(
    {
        "products": recent_products,
        "total_units": total_units
    }
)


# --------------------------------------------------
# 13. Display output
# --------------------------------------------------

print("\nRECENT PRODUCTS")
print("-" * 50)

for product in recent_products:
    print(
        f"{product['product_name']}: "
        f"{product['units_available']} units"
    )

print(f"\nTOTAL UNITS AVAILABLE: {total_units}")


print("\nEMAIL")
print("-" * 50)

print(f"To      : {email.recipient}")
print(f"Subject : {email.subject}")
print(f"\n{email.body}")