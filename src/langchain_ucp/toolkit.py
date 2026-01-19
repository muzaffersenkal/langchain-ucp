"""UCP Toolkit for LangChain agents.

This module provides the main UCPToolkit class that bundles all UCP tools
for easy integration with LangChain agents.
"""

from typing import List

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict

from langchain_ucp.client import UCPClient
from langchain_ucp.store import UCPStore, Product
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


class UCPToolkit(BaseModel):
    """Toolkit for UCP (Universal Commerce Protocol) operations.

    This toolkit provides all the tools needed to build a shopping agent
    that can interact with UCP-compliant merchants.

    Example:
        >>> from langchain_ucp import UCPToolkit
        >>> from langchain_openai import ChatOpenAI
        >>> from langgraph.prebuilt import create_react_agent
        >>>
        >>> # Create toolkit
        >>> toolkit = UCPToolkit(merchant_url="http://localhost:8000")
        >>>
        >>> # Create agent
        >>> llm = ChatOpenAI(model="gpt-4o")
        >>> agent = create_react_agent(llm, toolkit.get_tools())
        >>>
        >>> # Run agent
        >>> result = await agent.ainvoke({
        ...     "messages": [{"role": "user", "content": "Add roses to my cart"}]
        ... })

    Attributes:
        merchant_url: URL of the UCP merchant server
        agent_name: Name of this agent for UCP-Agent header
        products: Optional custom product catalog
        client: UCP HTTP client (auto-created)
        store: UCP store for state management (auto-created)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    merchant_url: str = Field(description="URL of the UCP merchant server")
    agent_name: str = Field(
        default="langchain-ucp-agent",
        description="Name of this agent for UCP-Agent header",
    )
    products: List[Product] | None = Field(
        default=None,
        description="Optional custom product catalog",
    )

    # Internal components (created on initialization)
    _client: UCPClient | None = None
    _store: UCPStore | None = None

    def model_post_init(self, __context) -> None:
        """Initialize client and store after model creation."""
        self._client = UCPClient(
            merchant_url=self.merchant_url,
            agent_name=self.agent_name,
        )
        self._store = UCPStore(
            client=self._client,
            products=self.products,
        )

    @property
    def client(self) -> UCPClient:
        """Get the UCP client."""
        if self._client is None:
            self._client = UCPClient(
                merchant_url=self.merchant_url,
                agent_name=self.agent_name,
            )
        return self._client

    @property
    def store(self) -> UCPStore:
        """Get the UCP store."""
        if self._store is None:
            self._store = UCPStore(
                client=self.client,
                products=self.products,
            )
        return self._store

    def get_tools(self) -> List[BaseTool]:
        """Get all UCP tools.

        Returns:
            List of LangChain tools for UCP operations
        """
        return [
            SearchCatalogTool(store=self.store),
            AddToCheckoutTool(store=self.store),
            RemoveFromCheckoutTool(store=self.store),
            UpdateCheckoutTool(store=self.store),
            GetCheckoutTool(store=self.store),
            UpdateCustomerDetailsTool(store=self.store),
            StartPaymentTool(store=self.store),
            CompleteCheckoutTool(store=self.store),
            CancelCheckoutTool(store=self.store),
            GetOrderTool(store=self.store),
        ]

    async def close(self) -> None:
        """Close the toolkit and release resources."""
        if self._client is not None:
            await self._client.close()

    def clear_session(self) -> None:
        """Clear the current checkout session."""
        if self._store is not None:
            self._store.clear_session()
