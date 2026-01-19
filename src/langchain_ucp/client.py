"""UCP HTTP Client using the official UCP SDK models."""

import uuid
import logging
from typing import Any

import httpx
from pydantic import BaseModel

# UCP SDK imports for type safety
from ucp_sdk.models.schemas.shopping.checkout_resp import CheckoutResponse
from ucp_sdk.models.schemas.shopping.checkout_create_req import CheckoutCreateRequest
from ucp_sdk.models.schemas.shopping.checkout_update_req import CheckoutUpdateRequest
from ucp_sdk.models.discovery.profile_schema import UcpDiscoveryProfile

logger = logging.getLogger(__name__)

# UCP Protocol Version
UCP_VERSION = "2026-01-11"


class UCPClientConfig(BaseModel):
    """Configuration for UCP Client."""

    merchant_url: str
    agent_name: str = "langchain-ucp-agent"
    timeout: float = 30.0


class UCPClient:
    """Async HTTP client for UCP-compliant merchants.

    This client handles all HTTP communication with UCP merchants,
    including proper header management and request/response handling.

    Attributes:
        config: Client configuration
        http_client: Underlying httpx async client
    """

    def __init__(
        self,
        merchant_url: str,
        agent_name: str = "langchain-ucp-agent",
        timeout: float = 30.0,
    ):
        """Initialize UCP client.

        Args:
            merchant_url: Base URL of the UCP merchant server
            agent_name: Name of this agent for UCP-Agent header
            timeout: Request timeout in seconds
        """
        self.config = UCPClientConfig(
            merchant_url=merchant_url.rstrip("/"),
            agent_name=agent_name,
            timeout=timeout,
        )
        self._http_client: httpx.AsyncClient | None = None

    @property
    def http_client(self) -> httpx.AsyncClient:
        """Lazy initialization of HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.config.timeout)
        return self._http_client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    def _get_headers(self, idempotency_key: str | None = None) -> dict[str, str]:
        """Get standard UCP headers required by UCP protocol.

        Args:
            idempotency_key: Optional idempotency key for the request

        Returns:
            Dictionary of HTTP headers
        """
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "UCP-Agent": f'{self.config.agent_name}; version="{UCP_VERSION}"',
            "Request-Signature": "dummy-signature",  # In production, compute HMAC
            "Request-Id": str(uuid.uuid4()),
            "Idempotency-Key": idempotency_key or str(uuid.uuid4()),
        }

    async def discover(self) -> UcpDiscoveryProfile:
        """Discover merchant UCP profile.

        Returns:
            UCP discovery profile with capabilities and payment handlers

        Raises:
            httpx.HTTPError: If the request fails
        """
        response = await self.http_client.get(
            f"{self.config.merchant_url}/.well-known/ucp"
        )
        response.raise_for_status()
        return UcpDiscoveryProfile.model_validate(response.json())

    async def create_checkout(
        self,
        request: CheckoutCreateRequest,
        idempotency_key: str | None = None,
    ) -> CheckoutResponse:
        """Create a new checkout session.

        Args:
            request: Checkout creation request
            idempotency_key: Optional idempotency key

        Returns:
            Created checkout session

        Raises:
            httpx.HTTPError: If the request fails
        """
        response = await self.http_client.post(
            f"{self.config.merchant_url}/checkout-sessions",
            json=request.model_dump(mode="json", by_alias=True, exclude_none=True),
            headers=self._get_headers(idempotency_key),
        )
        response.raise_for_status()
        return CheckoutResponse.model_validate(response.json())

    async def get_checkout(self, checkout_id: str) -> CheckoutResponse:
        """Get checkout session by ID.

        Args:
            checkout_id: Checkout session ID

        Returns:
            Checkout session data

        Raises:
            httpx.HTTPError: If the request fails
        """
        response = await self.http_client.get(
            f"{self.config.merchant_url}/checkout-sessions/{checkout_id}",
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return CheckoutResponse.model_validate(response.json())

    async def update_checkout(
        self,
        checkout_id: str,
        request: CheckoutUpdateRequest,
        idempotency_key: str | None = None,
    ) -> CheckoutResponse:
        """Update an existing checkout session.

        Args:
            checkout_id: Checkout session ID
            request: Checkout update request
            idempotency_key: Optional idempotency key

        Returns:
            Updated checkout session

        Raises:
            httpx.HTTPError: If the request fails
        """
        response = await self.http_client.put(
            f"{self.config.merchant_url}/checkout-sessions/{checkout_id}",
            json=request.model_dump(mode="json", by_alias=True, exclude_none=True),
            headers=self._get_headers(idempotency_key),
        )
        response.raise_for_status()
        return CheckoutResponse.model_validate(response.json())

    async def complete_checkout(
        self,
        checkout_id: str,
        payment_data: dict[str, Any],
        risk_signals: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> CheckoutResponse:
        """Complete a checkout session with payment.

        Args:
            checkout_id: Checkout session ID
            payment_data: Payment information
            risk_signals: Optional risk assessment signals
            idempotency_key: Optional idempotency key

        Returns:
            Completed checkout with order confirmation

        Raises:
            httpx.HTTPError: If the request fails
        """
        payload = {
            "payment": payment_data,
            "risk_signals": risk_signals or {},
        }

        response = await self.http_client.post(
            f"{self.config.merchant_url}/checkout-sessions/{checkout_id}/complete",
            json=payload,
            headers=self._get_headers(idempotency_key),
        )
        response.raise_for_status()
        return CheckoutResponse.model_validate(response.json())

    async def cancel_checkout(
        self,
        checkout_id: str,
        idempotency_key: str | None = None,
    ) -> CheckoutResponse:
        """Cancel a checkout session.

        Args:
            checkout_id: Checkout session ID
            idempotency_key: Optional idempotency key

        Returns:
            Cancelled checkout session

        Raises:
            httpx.HTTPError: If the request fails
        """
        response = await self.http_client.post(
            f"{self.config.merchant_url}/checkout-sessions/{checkout_id}/cancel",
            headers=self._get_headers(idempotency_key),
        )
        response.raise_for_status()
        return CheckoutResponse.model_validate(response.json())

    async def get_order(self, order_id: str) -> dict[str, Any]:
        """Get order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order data

        Raises:
            httpx.HTTPError: If the request fails
        """
        response = await self.http_client.get(
            f"{self.config.merchant_url}/orders/{order_id}",
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return response.json()
