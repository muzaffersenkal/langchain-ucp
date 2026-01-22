"""LangChain toolkit for Universal Commerce Protocol (UCP).

This package provides LangChain tools and toolkit for building AI agents
that can interact with UCP-compliant merchants.

Example:
    >>> from langchain_ucp import UCPToolkit, Product
    >>> from langchain_openai import ChatOpenAI
    >>> from langgraph.prebuilt import create_react_agent
    >>>
    >>> products = [Product(id="roses", title="Red Roses")]
    >>> toolkit = UCPToolkit(merchant_url="http://localhost:8000", products=products)
    >>> llm = ChatOpenAI(model="gpt-4o")
    >>> agent = create_react_agent(llm, toolkit.get_tools())
"""

from langchain_ucp.client import UCPClient
from langchain_ucp.exceptions import (
    UCPError,
    UCPNotFoundError,
    UCPRequestError,
    UCPValidationError,
    UCPVersionError,
)
from langchain_ucp.store import Product, UCPStore
from langchain_ucp.toolkit import UCPToolkit
from langchain_ucp.tools import (
    AddToCheckoutTool,
    CancelCheckoutTool,
    CompleteCheckoutTool,
    GetCheckoutTool,
    GetOrderTool,
    RemoveFromCheckoutTool,
    SearchCatalogTool,
    StartPaymentTool,
    UpdateCheckoutTool,
    UpdateCustomerDetailsTool,
)

__version__ = "0.1.0"

__all__ = [
    # Main toolkit
    "UCPToolkit",
    # Client and store
    "UCPClient",
    "UCPStore",
    "Product",
    # Errors
    "UCPError",
    "UCPValidationError",
    "UCPNotFoundError",
    "UCPRequestError",
    "UCPVersionError",
    # Individual tools
    "SearchCatalogTool",
    "AddToCheckoutTool",
    "RemoveFromCheckoutTool",
    "UpdateCheckoutTool",
    "GetCheckoutTool",
    "UpdateCustomerDetailsTool",
    "StartPaymentTool",
    "CompleteCheckoutTool",
    "CancelCheckoutTool",
    "GetOrderTool",
]
