"""UCP HTTP Client for communicating with UCP-compliant merchants."""

import json
import logging
import uuid
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel

from ucp_sdk.models.discovery.profile_schema import UcpDiscoveryProfile
from ucp_sdk.models.schemas.capability import Response as UcpCapability
from ucp_sdk.models.schemas.shopping.checkout_create_req import CheckoutCreateRequest
from ucp_sdk.models.schemas.shopping.checkout_resp import CheckoutResponse
from ucp_sdk.models.schemas.shopping.checkout_update_req import CheckoutUpdateRequest
from ucp_sdk.models.schemas.ucp import ResponseCheckout as UcpMetadata

from langchain_ucp.exceptions import (
    UCPError,
    UCPNotFoundError,
    UCPRequestError,
    UCPValidationError,
    UCPVersionError,
)

logger = logging.getLogger(__name__)

# Constants
UCP_VERSION = "2026-01-11"
DEFAULT_AGENT_NAME = "langchain-ucp-agent"
DEFAULT_TIMEOUT = 30.0


class UCPClientConfig(BaseModel):
    """Configuration for UCP Client."""

    merchant_url: str
    agent_name: str = DEFAULT_AGENT_NAME
    timeout: float = DEFAULT_TIMEOUT
    verbose: bool = False


class UCPClient:
    """Async HTTP client for UCP-compliant merchants.

    Handles HTTP communication, header management, profile caching,
    and capability negotiation.
    """

    def __init__(
        self,
        merchant_url: str,
        agent_name: str = DEFAULT_AGENT_NAME,
        timeout: float = DEFAULT_TIMEOUT,
        verbose: bool = False,
        agent_capabilities: list[dict[str, Any]] | None = None,
    ):
        """Initialize UCP client.

        Args:
            merchant_url: Base URL of the UCP merchant server
            agent_name: Name of this agent for UCP-Agent header
            timeout: Request timeout in seconds
            verbose: Enable verbose logging
            agent_capabilities: Capabilities this agent supports
        """
        self.config = UCPClientConfig(
            merchant_url=merchant_url.rstrip("/"),
            agent_name=agent_name,
            timeout=timeout,
            verbose=verbose,
        )
        self._http_client: httpx.AsyncClient | None = None
        self._cached_profile: UcpDiscoveryProfile | None = None
        self._agent_capabilities = agent_capabilities or []

        if verbose:
            logging.getLogger(__name__).setLevel(logging.DEBUG)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def verbose(self) -> bool:
        """Check if verbose mode is enabled."""
        return self.config.verbose

    @property
    def http_client(self) -> httpx.AsyncClient:
        """Lazy initialization of HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.config.timeout)
        return self._http_client

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    def clear_profile_cache(self) -> None:
        """Clear the cached merchant profile."""
        self._cached_profile = None

    # -------------------------------------------------------------------------
    # Discovery
    # -------------------------------------------------------------------------

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
        if use_cache and self._cached_profile:
            self._log("Using cached profile")
            return self._cached_profile

        url = f"{self.config.merchant_url}/.well-known/ucp"
        data = await self._get(url)
        profile = UcpDiscoveryProfile.model_validate(data)

        if validate_version:
            self._validate_version(profile)

        self._cached_profile = profile
        return profile

    async def get_common_capabilities(self) -> list[UcpCapability]:
        """Get capabilities supported by both agent and merchant."""
        profile = await self.discover()
        return self._get_common_capabilities(profile)

    async def get_negotiated_metadata(self) -> UcpMetadata:
        """Get UCP metadata with negotiated capabilities."""
        profile = await self.discover()
        common_capabilities = self._get_common_capabilities(profile)
        return UcpMetadata(
            version=profile.ucp.version,
            capabilities=common_capabilities,
        )

    # -------------------------------------------------------------------------
    # Checkout Operations
    # -------------------------------------------------------------------------

    async def create_checkout(
        self,
        request: CheckoutCreateRequest,
        idempotency_key: str | None = None,
    ) -> CheckoutResponse:
        """Create a new checkout session."""
        url = f"{self.config.merchant_url}/checkout-sessions"
        payload = request.model_dump(mode="json", by_alias=True, exclude_none=True)
        data = await self._post(url, payload, idempotency_key)
        return CheckoutResponse.model_validate(data)

    async def get_checkout(self, checkout_id: str) -> CheckoutResponse:
        """Get checkout session by ID."""
        url = f"{self.config.merchant_url}/checkout-sessions/{checkout_id}"
        data = await self._get(url)
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
        data = await self._put(url, payload, idempotency_key)
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
            "payment_data": payment_data,
            "risk_signals": risk_signals or {},
        }
        data = await self._post(url, payload, idempotency_key)
        return CheckoutResponse.model_validate(data)

    async def cancel_checkout(
        self,
        checkout_id: str,
        idempotency_key: str | None = None,
    ) -> CheckoutResponse:
        """Cancel a checkout session."""
        url = f"{self.config.merchant_url}/checkout-sessions/{checkout_id}/cancel"
        data = await self._post(url, None, idempotency_key)
        return CheckoutResponse.model_validate(data)

    # -------------------------------------------------------------------------
    # Order Operations
    # -------------------------------------------------------------------------

    async def get_order(self, order_id: str) -> dict[str, Any]:
        """Get order by ID."""
        url = f"{self.config.merchant_url}/orders/{order_id}"
        return await self._get(url)

    # -------------------------------------------------------------------------
    # Private: HTTP Methods
    # -------------------------------------------------------------------------

    async def _get(self, url: str) -> dict[str, Any]:
        """Execute GET request."""
        self._log(f"GET {url}")
        response = await self.http_client.get(url, headers=self._get_headers())
        data = self._handle_response(response)
        self._log("Response", data)
        return data

    async def _post(
        self,
        url: str,
        payload: dict[str, Any] | None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Execute POST request."""
        self._log(f"POST {url}")
        if payload:
            self._log("Request payload", payload)
        headers = self._get_headers(idempotency_key)
        response = await self.http_client.post(url, json=payload, headers=headers)
        data = self._handle_response(response)
        self._log("Response", data)
        return data

    async def _put(
        self,
        url: str,
        payload: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Execute PUT request."""
        self._log(f"PUT {url}")
        self._log("Request payload", payload)
        headers = self._get_headers(idempotency_key)
        response = await self.http_client.put(url, json=payload, headers=headers)
        data = self._handle_response(response)
        self._log("Response", data)
        return data

    # -------------------------------------------------------------------------
    # Private: Helpers
    # -------------------------------------------------------------------------

    def _get_headers(self, idempotency_key: str | None = None) -> dict[str, str]:
        """Get standard UCP headers."""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "UCP-Agent": f'{self.config.agent_name}; version="{UCP_VERSION}"',
            "Request-Signature": "dummy-signature",
            "Request-Id": str(uuid.uuid4()),
            "Idempotency-Key": idempotency_key or str(uuid.uuid4()),
        }

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """Handle HTTP response and raise appropriate errors."""
        if response.is_success:
            return response.json()

        error = self._parse_error(response)
        self._log(f"Error: {error}")
        raise error

    def _parse_error(self, response: httpx.Response) -> UCPError:
        """Parse error response and return appropriate exception."""
        status_code = response.status_code

        try:
            error_data = response.json()
        except Exception:
            error_data = {"message": response.text or "Unknown error"}

        message = (
            error_data.get("message")
            or error_data.get("detail")
            or str(error_data)
        )

        if status_code == 422:
            return self._parse_validation_error(error_data, message)
        if status_code == 404:
            return UCPNotFoundError(message, status_code=status_code)
        if status_code == 400:
            return UCPRequestError(f"Bad request: {message}", status_code=status_code)

        return UCPError(message, status_code=status_code, details=error_data)

    def _parse_validation_error(
        self, error_data: dict, message: str
    ) -> UCPValidationError:
        """Parse validation error details."""
        if "detail" in error_data and isinstance(error_data["detail"], list):
            field_errors = [
                {
                    "field": ".".join(str(loc) for loc in err.get("loc", [])) or "unknown",
                    "message": err.get("msg", "invalid"),
                }
                for err in error_data["detail"]
            ]
            return UCPValidationError(
                f"Invalid request: {len(field_errors)} field(s) have errors",
                field_errors=field_errors,
            )
        return UCPValidationError(message)

    def _validate_version(self, merchant_profile: UcpDiscoveryProfile) -> None:
        """Validate version compatibility."""
        merchant_version_str = merchant_profile.ucp.version

        try:
            merchant_version = datetime.strptime(merchant_version_str, "%Y-%m-%d").date()
            agent_version = datetime.strptime(UCP_VERSION, "%Y-%m-%d").date()
        except ValueError as e:
            logger.warning(f"Could not parse UCP version: {e}")
            return

        if agent_version > merchant_version:
            raise UCPVersionError(UCP_VERSION, merchant_version_str)

    def _get_common_capabilities(
        self, merchant_profile: UcpDiscoveryProfile
    ) -> list[UcpCapability]:
        """Find common capabilities between agent and merchant."""
        agent_capability_set = {
            (cap.get("name"), cap.get("version"))
            for cap in self._agent_capabilities
        }

        merchant_capabilities = merchant_profile.ucp.capabilities or []
        return [
            cap
            for cap in merchant_capabilities
            if (
                cap.name,
                cap.version.root if hasattr(cap.version, "root") else cap.version,
            )
            in agent_capability_set
        ]

    def _log(self, message: str, data: Any = None) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            if data:
                logger.debug(f"[UCP] {message}: {json.dumps(data, indent=2, default=str)}")
            else:
                logger.debug(f"[UCP] {message}")
