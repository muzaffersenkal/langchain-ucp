"""UCP Store for managing checkout sessions and product catalog."""

import logging
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from ucp_sdk.models.schemas.shopping.checkout_create_req import CheckoutCreateRequest
from ucp_sdk.models.schemas.shopping.checkout_resp import CheckoutResponse
from ucp_sdk.models.schemas.shopping.checkout_update_req import CheckoutUpdateRequest
from ucp_sdk.models.schemas.shopping.payment_create_req import PaymentCreateRequest
from ucp_sdk.models.schemas.shopping.payment_update_req import PaymentUpdateRequest
from ucp_sdk.models.schemas.shopping.types.buyer import Buyer
from ucp_sdk.models.schemas.shopping.types.item_create_req import ItemCreateRequest
from ucp_sdk.models.schemas.shopping.types.item_update_req import ItemUpdateRequest
from ucp_sdk.models.schemas.shopping.types.line_item_create_req import LineItemCreateRequest
from ucp_sdk.models.schemas.shopping.types.line_item_update_req import LineItemUpdateRequest

from langchain_ucp.client import UCPClient

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CURRENCY = "USD"
DEFAULT_PAYMENT_HANDLER = "mock_payment_handler"
DEFAULT_PAYMENT_TOKEN = "success_token"


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


class Product(BaseModel):
    """Product model for agent-side catalog discovery.

    Note: Temporary solution until UCP product discovery is implemented.
    """

    id: str
    title: str
    description: str | None = None
    image_url: str | None = None


class ProductSearchResult(BaseModel):
    """Product search result."""

    products: list[Product]
    query: str
    total: int


# -----------------------------------------------------------------------------
# Store
# -----------------------------------------------------------------------------


class UCPStore:
    """Store for managing UCP checkout sessions and product catalog."""

    def __init__(
        self,
        client: UCPClient,
        products: list[Product],
        verbose: bool = False,
    ):
        """Initialize UCP Store.

        Args:
            client: UCP HTTP client instance
            products: Product catalog
            verbose: Enable verbose logging
        """
        self.client = client
        self.checkout_id: str | None = None
        self._checkout_cache: CheckoutResponse | None = None
        self.verbose = verbose
        self.products: dict[str, Product] = {p.id: p for p in products}

        if verbose:
            logging.getLogger(__name__).setLevel(logging.DEBUG)

    # -------------------------------------------------------------------------
    # Product Operations
    # -------------------------------------------------------------------------

    def search_products(self, query: str) -> ProductSearchResult:
        """Search the product catalog."""
        keywords = query.lower().split()
        matching = [
            product
            for product in self.products.values()
            if self._matches_keywords(product, keywords)
        ]

        return ProductSearchResult(
            products=matching or list(self.products.values()),
            query=query,
            total=len(matching) if matching else len(self.products),
        )

    def get_product(self, product_id: str) -> Product | None:
        """Get product by ID."""
        return self.products.get(product_id)

    # -------------------------------------------------------------------------
    # Checkout Operations
    # -------------------------------------------------------------------------

    async def add_to_checkout(
        self,
        product_id: str,
        quantity: int = 1,
    ) -> CheckoutResponse:
        """Add a product to the checkout session."""
        product = self._get_product_or_raise(product_id)

        if self.checkout_id:
            return await self._add_to_existing_checkout(product_id, quantity)

        line_item = LineItemCreateRequest(
            item=ItemCreateRequest(id=product.id),
            quantity=quantity,
        )
        return await self._create_new_checkout([line_item])

    async def remove_from_checkout(self, product_id: str) -> CheckoutResponse:
        """Remove a product from the checkout."""
        self._ensure_active_checkout()

        existing = await self.client.get_checkout(self.checkout_id)
        updated_items = [
            self._to_line_item_update(item)
            for item in (existing.line_items or [])
            if item.item.id != product_id
        ]

        return await self._update_checkout_items(existing, updated_items)

    async def update_checkout_quantity(
        self,
        product_id: str,
        quantity: int,
    ) -> CheckoutResponse:
        """Update the quantity of a product in checkout."""
        if quantity == 0:
            return await self.remove_from_checkout(product_id)

        self._ensure_active_checkout()

        existing = await self.client.get_checkout(self.checkout_id)
        updated_items = [
            self._to_line_item_update(item, quantity if item.item.id == product_id else None)
            for item in (existing.line_items or [])
        ]

        return await self._update_checkout_items(existing, updated_items)

    async def get_checkout(self) -> CheckoutResponse | None:
        """Get the current checkout session."""
        if not self.checkout_id:
            return None

        checkout = await self.client.get_checkout(self.checkout_id)
        self._checkout_cache = checkout
        return checkout

    # -------------------------------------------------------------------------
    # Customer & Fulfillment
    # -------------------------------------------------------------------------

    async def update_customer_details(
        self,
        first_name: str,
        last_name: str,
        street_address: str,
        address_locality: str,
        address_region: str,
        postal_code: str,
        address_country: str = "US",
        extended_address: str | None = None,
        email: str | None = None,
    ) -> CheckoutResponse:
        """Update customer details and complete the fulfillment flow.

        Handles the full flow:
        1. Add shipping address
        2. Select destination
        3. Select shipping option
        """
        self._ensure_active_checkout()

        existing = await self.client.get_checkout(self.checkout_id)
        line_items = self._build_line_items_for_update(existing)
        buyer = self._build_buyer(email, first_name, last_name)
        address = self._build_address(
            first_name, last_name, street_address, address_locality,
            address_region, postal_code, address_country, extended_address
        )

        # Step 1: Add shipping address
        checkout = await self._add_shipping_address(existing, line_items, buyer, address)

        # Step 2: Select destination
        checkout = await self._select_destination(checkout, line_items, buyer)

        # Step 3: Select shipping option
        checkout = await self._select_shipping_option(checkout, line_items, buyer)

        self._checkout_cache = checkout
        return checkout

    # -------------------------------------------------------------------------
    # Payment & Completion
    # -------------------------------------------------------------------------

    async def start_payment(self) -> CheckoutResponse | str:
        """Prepare checkout for payment."""
        self._ensure_active_checkout()

        checkout = await self.client.get_checkout(self.checkout_id)
        missing = self._get_missing_for_payment(checkout)

        if missing:
            return f"Please provide: {', '.join(missing)}"

        self._checkout_cache = checkout
        return checkout

    async def complete_checkout(
        self,
        payment_handler_id: str = DEFAULT_PAYMENT_HANDLER,
        payment_token: str = DEFAULT_PAYMENT_TOKEN,
    ) -> CheckoutResponse:
        """Complete the checkout with payment."""
        self._ensure_active_checkout()

        checkout = await self.client.get_checkout(self.checkout_id)

        if checkout.status != "ready_for_complete":
            raise ValueError(
                f"Checkout not ready. Status: {checkout.status}. "
                "Please add buyer info and shipping address first."
            )

        payment_data = self._build_payment_data(payment_handler_id, payment_token)

        completed = await self.client.complete_checkout(
            self.checkout_id,
            payment_data=payment_data,
            risk_signals={"device_id": "langchain_agent"},
        )

        self.clear_session()
        return completed

    async def cancel_checkout(self) -> CheckoutResponse:
        """Cancel the current checkout."""
        self._ensure_active_checkout()

        checkout = await self.client.cancel_checkout(self.checkout_id)
        self.clear_session()
        return checkout

    # -------------------------------------------------------------------------
    # Order Operations
    # -------------------------------------------------------------------------

    async def get_order(self, order_id: str) -> dict[str, Any]:
        """Get order details."""
        return await self.client.get_order(order_id)

    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    def clear_session(self) -> None:
        """Clear the current checkout session."""
        self.checkout_id = None
        self._checkout_cache = None

    # -------------------------------------------------------------------------
    # Private: Checkout Helpers
    # -------------------------------------------------------------------------

    async def _create_new_checkout(
        self,
        line_items: list[LineItemCreateRequest],
    ) -> CheckoutResponse:
        """Create a new checkout session."""
        create_req = CheckoutCreateRequest(
            currency=DEFAULT_CURRENCY,
            line_items=line_items,
            payment=PaymentCreateRequest(instruments=[]),
        )
        checkout = await self.client.create_checkout(create_req)
        self.checkout_id = checkout.id
        self._checkout_cache = checkout
        return checkout

    async def _add_to_existing_checkout(
        self,
        product_id: str,
        quantity: int,
    ) -> CheckoutResponse:
        """Add item to existing checkout."""
        try:
            existing = await self.client.get_checkout(self.checkout_id)
            updated_items = self._merge_item_into_checkout(
                existing, product_id, quantity
            )
            return await self._update_checkout_items(existing, updated_items)
        except Exception:
            line_item = LineItemCreateRequest(
                item=ItemCreateRequest(id=product_id),
                quantity=quantity,
            )
            return await self._create_new_checkout([line_item])

    async def _update_checkout_items(
        self,
        existing: CheckoutResponse,
        line_items: list[LineItemUpdateRequest],
    ) -> CheckoutResponse:
        """Update checkout with new line items."""
        update_req = CheckoutUpdateRequest(
            id=self.checkout_id,
            currency=existing.currency,
            line_items=line_items,
            payment=PaymentUpdateRequest(instruments=[]),
        )
        checkout = await self.client.update_checkout(self.checkout_id, update_req)
        self._checkout_cache = checkout
        return checkout

    # -------------------------------------------------------------------------
    # Private: Fulfillment Flow
    # -------------------------------------------------------------------------

    async def _add_shipping_address(
        self,
        existing: CheckoutResponse,
        line_items: list[LineItemUpdateRequest],
        buyer: Buyer | None,
        address: dict[str, Any],
    ) -> CheckoutResponse:
        """Step 1: Add shipping address."""
        fulfillment = {
            "methods": [{"type": "shipping", "destinations": [address]}]
        }

        update_req = CheckoutUpdateRequest(
            id=self.checkout_id,
            currency=existing.currency,
            line_items=line_items,
            payment=PaymentUpdateRequest(instruments=[]),
            buyer=buyer,
            fulfillment=fulfillment,
        )

        checkout = await self.client.update_checkout(self.checkout_id, update_req)
        self._log(f"Step 1: Added shipping address, status={checkout.status}")
        return checkout

    async def _select_destination(
        self,
        checkout: CheckoutResponse,
        line_items: list[LineItemUpdateRequest],
        buyer: Buyer | None,
    ) -> CheckoutResponse:
        """Step 2: Select destination to trigger option generation."""
        dest_id = self._get_destination_id(checkout)
        if not dest_id:
            self._log("Step 2: No destinations found")
            return checkout

        self._log(f"Step 2: Selecting destination {dest_id}")
        fulfillment = {
            "methods": [{"type": "shipping", "selected_destination_id": dest_id}]
        }

        update_req = CheckoutUpdateRequest(
            id=self.checkout_id,
            currency=checkout.currency,
            line_items=line_items,
            payment=PaymentUpdateRequest(instruments=[]),
            buyer=buyer,
            fulfillment=fulfillment,
        )

        checkout = await self.client.update_checkout(self.checkout_id, update_req)
        self._log(f"Step 2: Selected destination, status={checkout.status}")
        return checkout

    async def _select_shipping_option(
        self,
        checkout: CheckoutResponse,
        line_items: list[LineItemUpdateRequest],
        buyer: Buyer | None,
    ) -> CheckoutResponse:
        """Step 3: Select first available shipping option."""
        dest_id = self._get_destination_id(checkout)
        option_id = self._get_first_option_id(checkout)

        if not option_id:
            self._log("Step 3: No shipping options available")
            return checkout

        self._log(f"Step 3: Selecting option {option_id}")
        fulfillment = {
            "methods": [
                {
                    "type": "shipping",
                    "selected_destination_id": dest_id,
                    "groups": [{"selected_option_id": option_id}],
                }
            ]
        }

        update_req = CheckoutUpdateRequest(
            id=self.checkout_id,
            currency=checkout.currency,
            line_items=line_items,
            payment=PaymentUpdateRequest(instruments=[]),
            buyer=buyer,
            fulfillment=fulfillment,
        )

        checkout = await self.client.update_checkout(self.checkout_id, update_req)
        self._log(f"Step 3: Selected option, status={checkout.status}")
        return checkout

    # -------------------------------------------------------------------------
    # Private: Data Extraction
    # -------------------------------------------------------------------------

    def _get_destination_id(self, checkout: CheckoutResponse) -> str | None:
        """Extract destination ID from checkout."""
        data = checkout.model_dump(mode="json")
        methods = data.get("fulfillment", {}).get("methods", [])
        if methods:
            destinations = methods[0].get("destinations", [])
            if destinations:
                return destinations[0].get("id")
        return None

    def _get_first_option_id(self, checkout: CheckoutResponse) -> str | None:
        """Extract first shipping option ID from checkout."""
        data = checkout.model_dump(mode="json")
        methods = data.get("fulfillment", {}).get("methods", [])
        if methods:
            groups = methods[0].get("groups", [])
            if groups:
                options = groups[0].get("options", [])
                if options:
                    return options[0].get("id")
        return None

    # -------------------------------------------------------------------------
    # Private: Builders
    # -------------------------------------------------------------------------

    def _build_line_items_for_update(
        self,
        checkout: CheckoutResponse,
    ) -> list[LineItemUpdateRequest]:
        """Build line items list for update request."""
        return [
            self._to_line_item_update(item)
            for item in (checkout.line_items or [])
        ]

    def _build_buyer(
        self,
        email: str | None,
        first_name: str,
        last_name: str,
    ) -> Buyer | None:
        """Build buyer object."""
        if email:
            return Buyer(email=email, first_name=first_name, last_name=last_name)
        return None

    def _build_address(
        self,
        first_name: str,
        last_name: str,
        street_address: str,
        address_locality: str,
        address_region: str,
        postal_code: str,
        address_country: str,
        extended_address: str | None,
    ) -> dict[str, Any]:
        """Build shipping address dict."""
        address = {
            "id": f"dest_{uuid4().hex[:8]}",
            "street_address": street_address,
            "address_locality": address_locality,
            "address_region": address_region,
            "postal_code": postal_code,
            "address_country": address_country,
            "first_name": first_name,
            "last_name": last_name,
        }
        if extended_address:
            address["extended_address"] = extended_address
        return address

    def _build_payment_data(
        self,
        handler_id: str,
        token: str,
    ) -> dict[str, Any]:
        """Build payment instrument data."""
        return {
            "id": f"inst_{uuid4().hex[:8]}",
            "handler_id": handler_id,
            "handler_name": handler_id,
            "type": "card",
            "brand": "Visa",
            "last_digits": "4242",
            "credential": {"type": "token", "token": token},
        }

    # -------------------------------------------------------------------------
    # Private: Converters
    # -------------------------------------------------------------------------

    def _to_line_item_update(
        self,
        item: Any,
        override_quantity: int | None = None,
    ) -> LineItemUpdateRequest:
        """Convert line item to update request."""
        return LineItemUpdateRequest(
            id=item.id,
            item=ItemUpdateRequest(id=item.item.id),
            quantity=override_quantity if override_quantity is not None else item.quantity,
        )

    def _merge_item_into_checkout(
        self,
        checkout: CheckoutResponse,
        product_id: str,
        quantity: int,
    ) -> list[LineItemUpdateRequest]:
        """Merge new item into existing checkout items."""
        existing_items = list(checkout.line_items) if checkout.line_items else []
        found = False
        updated_items = []

        for item in existing_items:
            if item.item.id == product_id:
                updated_items.append(
                    self._to_line_item_update(item, item.quantity + quantity)
                )
                found = True
            else:
                updated_items.append(self._to_line_item_update(item))

        if not found:
            updated_items.append(
                LineItemUpdateRequest(
                    item=ItemUpdateRequest(id=product_id),
                    quantity=quantity,
                )
            )

        return updated_items

    # -------------------------------------------------------------------------
    # Private: Validation
    # -------------------------------------------------------------------------

    def _ensure_active_checkout(self) -> None:
        """Ensure there is an active checkout session."""
        if not self.checkout_id:
            raise ValueError("No active checkout session")

    def _get_product_or_raise(self, product_id: str) -> Product:
        """Get product or raise ValueError."""
        product = self.get_product(product_id)
        if not product:
            raise ValueError(f"Product {product_id} not found in catalog")
        return product

    def _get_missing_for_payment(self, checkout: CheckoutResponse) -> list[str]:
        """Get list of missing items for payment."""
        missing = []
        if not checkout.buyer:
            missing.append("buyer email address")
        if not hasattr(checkout, "fulfillment") or not checkout.fulfillment:
            missing.append("shipping address")
        return missing

    def _matches_keywords(self, product: Product, keywords: list[str]) -> bool:
        """Check if product matches search keywords."""
        searchable = f"{product.title} {product.id}".lower()
        return any(keyword in searchable for keyword in keywords)

    # -------------------------------------------------------------------------
    # Private: Logging
    # -------------------------------------------------------------------------

    def _log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            logger.debug(f"[UCPStore] {message}")
