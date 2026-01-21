"""Basic example: Create a shopping agent with UCPToolkit.

Prerequisites:
- pip install langchain-ucp langchain-openai langgraph
- Set OPENAI_API_KEY environment variable
- UCP merchant server running at http://localhost:8000

Usage:
    python basic_agent.py              # Normal mode
    python basic_agent.py --verbose    # Verbose mode with debug logs
"""

import asyncio
import os
import sys

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

from langchain_ucp import UCPToolkit, Product


# Define your product catalog (for agent-side discovery)
# Full product details come from the merchant via UCP
PRODUCTS = [
    Product(id="bouquet_roses", title="Bouquet of Red Roses"),
    Product(id="bouquet_sunflowers", title="Sunflower Bundle"),
    Product(id="bouquet_tulips", title="Spring Tulips"),
    Product(id="orchid_white", title="White Orchid"),
    Product(id="pot_ceramic", title="Ceramic Pot"),
    Product(id="gardenias", title="Gardenias"),
]


async def main():
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    # Check for verbose flag
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    # Create toolkit with product catalog
    toolkit = UCPToolkit(
        merchant_url="http://localhost:8000",
        products=PRODUCTS,
        verbose=verbose,
    )

    # Create agent
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    agent = create_react_agent(llm, toolkit.get_tools())

    # Run agent
    result = await agent.ainvoke({
        "messages": [HumanMessage(content="Search for roses and add them to my cart")]
    })

    # Print response
    print(result["messages"][-1].content)

    # Cleanup
    await toolkit.close()


if __name__ == "__main__":
    asyncio.run(main())
