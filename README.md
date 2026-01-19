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
from langchain_ucp import UCPToolkit
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Create toolkit
toolkit = UCPToolkit(merchant_url="http://localhost:8000")

# Create agent
llm = ChatOpenAI(model="gpt-4o")
agent = create_react_agent(llm, toolkit.get_tools())

# Run agent
result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "I want to buy some red roses"}]
})
```

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

## License

Apache License 2.0
