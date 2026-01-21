"""UCP HTTP Client using the official UCP SDK models."""

import uuid
import logging
import json
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel

# UCP SDK imports for type safety
from ucp_sdk.models.schemas.shopping.checkout_resp import CheckoutResponse
from ucp_sdk.models.schemas.shopping.checkout_create_req import CheckoutCreateRequest
from ucp_sdk.models.schemas.shopping.checkout_update_req import CheckoutUpdateRequest
from ucp_sdk.models.discovery.profile_schema import UcpDiscoveryProfile
from ucp_sdk.models.schemas.capability import Response as UcpCapability
from ucp_sdk.models.schemas.ucp import ResponseCheckout as UcpMetadata

logger = logging.getLogger(__name__)

# UCP Protocol Version
UCP_VERSION = "2026-01-11"


class UCPVersionError(Exception):
    """Raised when UCP version is incompatible."""

    def __init__(self, client_version: str, merchant_version: str):
        self.client_version = client_version
        self.merchant_version = merchant_version
        super().__init__(
            f"UCP version {client_version} is not supported. "
            f"Merchant implements version {merchant_version}."
        )


class UCPClientConfig(BaseModel):
    """Configuration for UCP Client."""

    merchant_url: str
    agent_name: str = "langchain-ucp-agent"
    timeout: float = 30.0
    verbose: bool = False


class UCPClient:
    """Async HTTP client for UCP-compliant merchants.

    This client handles all HTTP communication with UCP merchants,
    including proper header management, request/response handling,
    profile caching, and capability negotiation.

    Attributes:
        config: Client configuration
        http_client: Underlying httpx async client
        verbose: Enable verbose logging
        agent_capabilities: Capabilities this agent supports
    """

    def __init__(
        self,
        merchant_url: str,
        agent_name: str = "langchain-ucp-agent",
        timeout: float = 30.0,
        verbose: bool = False,
        agent_capabilities: list[dict[str, Any]] | None = None,
    ):
        """Initialize UCP client.

        Args:
            merchant_url: Base URL of the UCP merchant server
            agent_name: Name of this agent for UCP-Agent header
            timeout: Request timeout in seconds
            verbose: Enable verbose logging of requests/responses
            agent_capabilities: List of capabilities this agent supports
        """
        self.config = UCPClientConfig(
            merchant_url=merchant_url.rstrip("/"),
            agent_name=agent_name,
            timeout=timeout,
            verbose=verbose,
        )
        self._http_client: httpx.AsyncClient | None = None

        # Profile caching and capabilities
        self._cached_profile: UcpDiscoveryProfile | None = None
        self._agent_capabilities = agent_capabilities or []

        if verbose:
            logging.getLogger(__name__).setLevel(logging.DEBUG)

    @property
    def verbose(self) -> bool:
        """Check if verbose mode is enabled."""
        return self.config.verbose

    def _log(self, message: str, data: Any = None) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            if data:
                logger.debug(f"[UCP] {message}: {json.dumps(data, indent=2, default=str)}")
            else:
                logger.debug(f"[UCP] {message}")

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
        """Get standard UCP headers required by UCP protocol."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "UCP-Agent": f'{self.config.agent_name}; version="{UCP_VERSION}"',
            "Request-Signature": "dummy-signature",
            "Request-Id": str(uuid.uuid4()),
            "Idempotency-Key": idempotency_key or str(uuid.uuid4()),
        }
        return headers

    def clear_profile_cache(self) -> None:
        """Clear the cached merchant profile."""
        self._cached_profile = None

    def _validate_version(self, merchant_profile: UcpDiscoveryProfile) -> None:
        """Validate version compatibility between agent and merchant.

        Args:
            merchant_profile: The merchant's UCP profile

        Raises:
            UCPVersionError: If versions are incompatible
        """
        merchant_version_str = merchant_profile.ucp.version

        try:
            merchant_version = datetime.strptime(merchant_version_str, "%Y-%m-%d").date()
            agent_version = datetime.strptime(UCP_VERSION, "%Y-%m-%d").date()
        except ValueError as e:
            logger.warning(f"Could not parse UCP version: {e}")
            return

        # Agent should not request features from a newer version than merchant supports
        if agent_version > merchant_version:
            raise UCPVersionError(UCP_VERSION, merchant_version_str)

    def _get_common_capabilities(
        self, merchant_profile: UcpDiscoveryProfile
    ) -> list[UcpCapability]:
        """Find common capabilities between agent and merchant.

        Args:
            merchant_profile: The merchant's UCP profile

        Returns:
            List of capabilities supported by both agent and merchant
        """
        # Build set of agent capabilities for fast lookup
        agent_capability_set = {
            (cap.get("name"), cap.get("version"))
            for cap in self._agent_capabilities
        }

        # Filter merchant capabilities to only those agent supports
        merchant_capabilities = merchant_profile.ucp.capabilities or []
        common = [
            cap
            for cap in merchant_capabilities
            if (cap.name, cap.version.root if hasattr(cap.version, 'root') else cap.version)
            in agent_capability_set
        ]

        return common

    async def discover(
        self,
        use_cache: bool = True,
        validate_version: bool = True,
    ) -> UcpDiscoveryProfile:
        """Discover merchant UCP profile.

        Args:
            use_cache: Use cached profile if available
            validate_version: Validate version compatibility

        Returns:
            The merchant's UCP discovery profile

        Raises:
            UCPVersionError: If version validation fails
        """
        # Check cache first
        if use_cache and self._cached_profile:
            self._log("Using cached profile")
            return self._cached_profile

        url = f"{self.config.merchant_url}/.well-known/ucp"
        self._log(f"GET {url}")

        response = await self.http_client.get(url)
        response.raise_for_status()

        data = response.json()
        self._log("Discovery response", data)
        profile = UcpDiscoveryProfile.model_validate(data)

        # Validate version compatibility
        if validate_version:
            self._validate_version(profile)

        # Cache the profile
        self._cached_profile = profile

        return profile

    async def get_common_capabilities(self) -> list[UcpCapability]:
        """Get capabilities supported by both agent and merchant.

        Returns:
            List of common capabilities
        """
        profile = await self.discover()
        return self._get_common_capabilities(profile)

    async def get_negotiated_metadata(self) -> UcpMetadata:
        """Get UCP metadata with negotiated capabilities.

        Returns:
            UcpMetadata with common capabilities between agent and merchant
        """
        profile = await self.discover()
        common_capabilities = self._get_common_capabilities(profile)

        return UcpMetadata(
            version=profile.ucp.version,
            capabilities=common_capabilities,
        )

    async def create_checkout(
        self,
        request: CheckoutCreateRequest,
        idempotency_key: str | None = None,
    ) -> CheckoutResponse:
        """Create a new checkout session."""
        url = f"{self.config.merchant_url}/checkout-sessions"
        payload = request.model_dump(mode="json", by_alias=True, exclude_none=True)
        headers = self._get_headers(idempotency_key)

        self._log(f"POST {url}")
        self._log("Request payload", payload)

        response = await self.http_client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()
        self._log("Response", data)
        return CheckoutResponse.model_validate(data)

    async def get_checkout(self, checkout_id: str) -> CheckoutResponse:
        """Get checkout session by ID."""
        url = f"{self.config.merchant_url}/checkout-sessions/{checkout_id}"
        self._log(f"GET {url}")

        response = await self.http_client.get(url, headers=self._get_headers())
        response.raise_for_status()

        data = response.json()
        self._log("Response", data)
        return CheckoutResponse.model_validate(data)

    async def update_checkout(
        self,
        checkout_id: str,
        request: CheckoutUpdateRequest,
        idempotency_key: str | None = None,
    ) -> CheckoutResponse:
        """Update an existing checkout session."""
        url = f"{self.config.merchant_url}/checkout-sessions/{checkout_id}"
        payload = request.model_dump(mode="json", by_alias=True, exclude_none=True)
        headers = self._get_headers(idempotency_key)

        self._log(f"PUT {url}")
        self._log("Request payload", payload)

        response = await self.http_client.put(url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()
        self._log("Response", data)
        return CheckoutResponse.model_validate(data)

    async def complete_checkout(
        self,
        checkout_id: str,
        payment_data: dict[str, Any],
        risk_signals: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> CheckoutResponse:
        """Complete a checkout session with payment."""
        url = f"{self.config.merchant_url}/checkout-sessions/{checkout_id}/complete"
        payload = {
            "payment": payment_data,
            "risk_signals": risk_signals or {},
        }
        headers = self._get_headers(idempotency_key)

        self._log(f"POST {url}")
        self._log("Request payload", payload)

        response = await self.http_client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()
        self._log("Response", data)
        return CheckoutResponse.model_validate(data)

    async def cancel_checkout(
        self,
        checkout_id: str,
        idempotency_key: str | None = None,
    ) -> CheckoutResponse:
        """Cancel a checkout session."""
        url = f"{self.config.merchant_url}/checkout-sessions/{checkout_id}/cancel"
        self._log(f"POST {url}")

        response = await self.http_client.post(url, headers=self._get_headers(idempotency_key))
        response.raise_for_status()

        data = response.json()
        self._log("Response", data)
        return CheckoutResponse.model_validate(data)

    async def get_order(self, order_id: str) -> dict[str, Any]:
        """Get order by ID."""
        url = f"{self.config.merchant_url}/orders/{order_id}"
        self._log(f"GET {url}")

        response = await self.http_client.get(url, headers=self._get_headers())
        response.raise_for_status()

        data = response.json()
        self._log("Response", data)
        return data
