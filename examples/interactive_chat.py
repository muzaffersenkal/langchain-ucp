"""Interactive chat example with UCPToolkit.

Prerequisites:
- pip install langchain-ucp langchain-openai langgraph
- Set OPENAI_API_KEY environment variable
- UCP merchant server running at http://localhost:8000

Usage:
    python interactive_chat.py              # Normal mode
    python interactive_chat.py --verbose    # Verbose mode with debug logs
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
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    # Check for verbose flag
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    # Initialize with product catalog
    toolkit = UCPToolkit(
        merchant_url="http://localhost:8000",
        products=PRODUCTS,
        verbose=verbose,
    )
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    agent = create_react_agent(llm, toolkit.get_tools())

    messages = []

    print("\n" + "=" * 50)
    print("UCP Shopping Assistant")
    print("=" * 50)
    print("Type 'quit' to exit\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            messages.append(HumanMessage(content=user_input))

            result = await agent.ainvoke({"messages": messages})

            response = result["messages"][-1].content
            print(f"\nAssistant: {response}\n")

            messages = result["messages"]

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")

    await toolkit.close()


if __name__ == "__main__":
    asyncio.run(main())
