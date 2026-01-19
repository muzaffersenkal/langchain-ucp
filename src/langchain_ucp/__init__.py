"""LangChain toolkit for Universal Commerce Protocol (UCP).

This package provides LangChain tools and toolkit for building AI agents
that can interact with UCP-compliant merchants.

Example:
    >>> from langchain_ucp import UCPToolkit
    >>> from langchain_openai import ChatOpenAI
    >>> from langgraph.prebuilt import create_react_agent
    >>>
    >>> toolkit = UCPToolkit(merchant_url="http://localhost:8000")
    >>> llm = ChatOpenAI(model="gpt-4o")
    >>> agent = create_react_agent(llm, toolkit.get_tools())
"""

from langchain_ucp.toolkit import UCPToolkit
from langchain_ucp.client import UCPClient
from langchain_ucp.store import UCPStore
from langchain_ucp.tools import (
    SearchCatalogTool,
    AddToCheckoutTool,
    RemoveFromCheckoutTool,
    UpdateCheckoutTool,
    GetCheckoutTool,
    UpdateCustomerDetailsTool,
    StartPaymentTool,
    CompleteCheckoutTool,
    CancelCheckoutTool,
    GetOrderTool,
)

__version__ = "0.1.0"

__all__ = [
    # Main toolkit
    "UCPToolkit",
    # Client and store
    "UCPClient",
    "UCPStore",
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
