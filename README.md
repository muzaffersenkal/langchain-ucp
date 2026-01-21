# LangChain UCP

LangChain toolkit for Universal Commerce Protocol (UCP).

## Overview

`langchain-ucp` provides LangChain tools and toolkit for building AI agents that can interact with [UCP](https://ucp.dev)-compliant merchants.

## Installation

```bash
pip install langchain-ucp
```

## Quick Start

```python
from langchain_ucp import UCPToolkit, Product
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Define your product catalog (for agent-side discovery)
# Full product details come from the merchant via UCP
products = [
    Product(id="roses", title="Red Roses"),
    Product(id="tulips", title="Spring Tulips"),
    Product(id="orchid", title="White Orchid"),
]

# Create toolkit with product catalog
toolkit = UCPToolkit(
    merchant_url="http://localhost:8000",
    products=products,
)

# Create agent
llm = ChatOpenAI(model="gpt-4o")
agent = create_react_agent(llm, toolkit.get_tools())

# Run agent
result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "I want to buy some red roses"}]
})
```

## Product Catalog

The `Product` model is a temporary solution for agent-side discovery. It only requires the minimum fields needed for search.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | str | Yes | Unique product identifier (must match merchant's product ID) |
| `title` | str | Yes | Product display name (used for search) |

> **Note:** When UCP product discovery is implemented, the full product catalog (pricing, categories, descriptions, images) will be fetched directly from the merchant.

## Available Tools

| Tool | Description |
|------|-------------|
| `search_shopping_catalog` | Search the product catalog |
| `add_to_checkout` | Add products to cart |
| `remove_from_checkout` | Remove products from cart |
| `update_checkout` | Update product quantities |
| `get_checkout` | View current cart |
| `update_customer_details` | Add buyer info and address |
| `start_payment` | Prepare checkout for payment |
| `complete_checkout` | Complete purchase |
| `cancel_checkout` | Cancel checkout |
| `get_order` | Get order details |

## Examples

See the [examples](./examples) directory for complete working examples:

- `basic_agent.py` - Simple agent that adds items to cart
- `interactive_chat.py` - Interactive chat with the shopping agent

## License

Apache License 2.0
