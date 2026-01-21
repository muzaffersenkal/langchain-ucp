"""UCP Toolkit for LangChain agents.

This module provides the main UCPToolkit class that bundles all UCP tools
for easy integration with LangChain agents.
"""

import logging
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

logger = logging.getLogger(__name__)


class UCPToolkit(BaseModel):
    """Toolkit for UCP (Universal Commerce Protocol) operations.

    This toolkit provides all the tools needed to build a shopping agent
    that can interact with UCP-compliant merchants.

    Example:
        >>> from langchain_ucp import UCPToolkit, Product
        >>> from langchain_openai import ChatOpenAI
        >>> from langgraph.prebuilt import create_react_agent
        >>>
        >>> # Define product catalog (for agent-side discovery)
        >>> products = [
        ...     Product(id="roses", title="Red Roses"),
        ...     Product(id="tulips", title="Tulips"),
        ... ]
        >>>
        >>> # Create toolkit
        >>> toolkit = UCPToolkit(
        ...     merchant_url="http://localhost:8000",
        ...     products=products,
        ... )
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
        products: Product catalog (required)
        verbose: Enable verbose logging
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    merchant_url: str = Field(description="URL of the UCP merchant server")
    agent_name: str = Field(
        default="langchain-ucp-agent",
        description="Name of this agent for UCP-Agent header",
    )
    products: List[Product] = Field(
        description="Product catalog for the store",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging of API calls and tool operations",
    )

    # Internal components (created on initialization)
    _client: UCPClient | None = None
    _store: UCPStore | None = None

    def model_post_init(self, __context) -> None:
        """Initialize client and store after model creation."""
        if self.verbose:
            # Configure logging for verbose output
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            )
            logger.debug(f"[UCPToolkit] Initializing with merchant_url={self.merchant_url}")

        self._client = UCPClient(
            merchant_url=self.merchant_url,
            agent_name=self.agent_name,
            verbose=self.verbose,
        )
        self._store = UCPStore(
            client=self._client,
            products=self.products,
            verbose=self.verbose,
        )

        if self.verbose:
            logger.debug(f"[UCPToolkit] Loaded {len(self._store.products)} products")

    @property
    def client(self) -> UCPClient:
        """Get the UCP client."""
        if self._client is None:
            self._client = UCPClient(
                merchant_url=self.merchant_url,
                agent_name=self.agent_name,
                verbose=self.verbose,
            )
        return self._client

    @property
    def store(self) -> UCPStore:
        """Get the UCP store."""
        if self._store is None:
            self._store = UCPStore(
                client=self.client,
                products=self.products,
                verbose=self.verbose,
            )
        return self._store

    def get_tools(self) -> List[BaseTool]:
        """Get all UCP tools.

        Returns:
            List of LangChain tools for UCP operations
        """
        tools = [
            SearchCatalogTool(store=self.store, verbose=self.verbose),
            AddToCheckoutTool(store=self.store, verbose=self.verbose),
            RemoveFromCheckoutTool(store=self.store, verbose=self.verbose),
            UpdateCheckoutTool(store=self.store, verbose=self.verbose),
            GetCheckoutTool(store=self.store, verbose=self.verbose),
            UpdateCustomerDetailsTool(store=self.store, verbose=self.verbose),
            StartPaymentTool(store=self.store, verbose=self.verbose),
            CompleteCheckoutTool(store=self.store, verbose=self.verbose),
            CancelCheckoutTool(store=self.store, verbose=self.verbose),
            GetOrderTool(store=self.store, verbose=self.verbose),
        ]

        if self.verbose:
            logger.debug(f"[UCPToolkit] Created {len(tools)} tools")

        return tools

    async def close(self) -> None:
        """Close the toolkit and release resources."""
        if self._client is not None:
            await self._client.close()

    def clear_session(self) -> None:
        """Clear the current checkout session."""
        if self._store is not None:
            self._store.clear_session()
            if self.verbose:
                logger.debug("[UCPToolkit] Session cleared")
