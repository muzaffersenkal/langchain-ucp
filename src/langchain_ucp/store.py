"""UCP Store for managing checkout sessions and product catalog.

This module provides state management for UCP checkout sessions,
similar to the RetailStore pattern in the business_agent.
"""

import logging
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

# UCP SDK imports
from ucp_sdk.models.schemas.shopping.checkout_resp import CheckoutResponse
from ucp_sdk.models.schemas.shopping.checkout_create_req import CheckoutCreateRequest
from ucp_sdk.models.schemas.shopping.checkout_update_req import CheckoutUpdateRequest
from ucp_sdk.models.schemas.shopping.types.line_item_create_req import LineItemCreateRequest
from ucp_sdk.models.schemas.shopping.types.line_item_update_req import LineItemUpdateRequest
from ucp_sdk.models.schemas.shopping.types.item_create_req import ItemCreateRequest
from ucp_sdk.models.schemas.shopping.types.item_update_req import ItemUpdateRequest
from ucp_sdk.models.schemas.shopping.types.buyer import Buyer
from ucp_sdk.models.schemas.shopping.payment_create_req import PaymentCreateRequest
from ucp_sdk.models.schemas.shopping.payment_update_req import PaymentUpdateRequest

from langchain_ucp.client import UCPClient

logger = logging.getLogger(__name__)


class Product(BaseModel):
    """Product model for the catalog."""

    id: str
    title: str
    price: int  # Price in cents
    category: str = ""
    description: str = ""
    image_url: str | None = None


class ProductSearchResult(BaseModel):
    """Product search result."""

    products: list[Product]
    query: str
    total: int


class UCPStore:
    """Store for managing UCP checkout sessions and product catalog.

    This class provides a high-level interface for managing shopping
    sessions, similar to the RetailStore in the A2A business_agent.

    Attributes:
        client: UCP HTTP client
        checkout_id: Current checkout session ID
        products: Product catalog
        verbose: Enable verbose logging
    """

    def __init__(
        self,
        client: UCPClient,
        products: list[Product] | None = None,
        verbose: bool = False,
    ):
        """Initialize UCP Store.

        Args:
            client: UCP HTTP client instance
            products: Optional product catalog. If not provided, uses default.
            verbose: Enable verbose logging
        """
        self.client = client
        self.checkout_id: str | None = None
        self._checkout_cache: CheckoutResponse | None = None
        self.verbose = verbose

        if verbose:
            logging.getLogger(__name__).setLevel(logging.DEBUG)

        # Initialize product catalog
        self.products: dict[str, Product] = {}
        if products:
            for product in products:
                self.products[product.id] = product
        else:
            self._initialize_default_products()

    def _log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            logger.debug(f"[UCPStore] {message}")

    def _initialize_default_products(self) -> None:
        """Initialize default product catalog (flower shop example)."""
        default_products = [
            Product(
                id="bouquet_roses",
                title="Bouquet of Red Roses",
                price=3500,
                category="flowers",
            ),
            Product(
                id="bouquet_sunflowers",
                title="Sunflower Bundle",
                price=2500,
                category="flowers",
            ),
            Product(
                id="bouquet_tulips",
                title="Spring Tulips",
                price=3000,
                category="flowers",
            ),
            Product(
                id="orchid_white",
                title="White Orchid",
                price=4500,
                category="flowers",
            ),
            Product(
                id="pot_ceramic",
                title="Ceramic Pot",
                price=1500,
                category="accessories",
            ),
            Product(
                id="gardenias",
                title="Gardenias",
                price=2000,
                category="flowers",
            ),
        ]
        for product in default_products:
            self.products[product.id] = product

    def search_products(self, query: str) -> ProductSearchResult:
        """Search the product catalog.

        Args:
            query: Search query string

        Returns:
            ProductSearchResult with matching products
        """
        query_lower = query.lower()
        keywords = query_lower.split()

        matching = []
        for product in self.products.values():
            searchable = f"{product.title} {product.category} {product.id}".lower()
            for keyword in keywords:
                if keyword in searchable and product not in matching:
                    matching.append(product)
                    break

        return ProductSearchResult(
            products=matching if matching else list(self.products.values()),
            query=query,
            total=len(matching) if matching else len(self.products),
        )

    def get_product(self, product_id: str) -> Product | None:
        """Get product by ID.

        Args:
            product_id: Product ID

        Returns:
            Product if found, None otherwise
        """
        return self.products.get(product_id)

    async def add_to_checkout(
        self,
        product_id: str,
        quantity: int = 1,
    ) -> CheckoutResponse:
        """Add a product to the checkout session.

        Creates a new checkout if one doesn't exist.

        Args:
            product_id: Product ID to add
            quantity: Quantity to add

        Returns:
            Updated checkout session

        Raises:
            ValueError: If product not found
        """
        product = self.get_product(product_id)
        if not product:
            raise ValueError(f"Product {product_id} not found in catalog")

        line_item = LineItemCreateRequest(
            item=ItemCreateRequest(id=product.id),
            quantity=quantity,
        )

        if self.checkout_id:
            # Get existing checkout and add item
            try:
                existing = await self.client.get_checkout(self.checkout_id)
                existing_items = list(existing.line_items) if existing.line_items else []

                # Check if item already exists
                found = False
                updated_items = []
                for item in existing_items:
                    if item.item.id == product_id:
                        # Update quantity
                        updated_items.append(
                            LineItemUpdateRequest(
                                id=item.id,
                                item=ItemUpdateRequest(id=item.item.id),
                                quantity=item.quantity + quantity,
                            )
                        )
                        found = True
                    else:
                        updated_items.append(
                            LineItemUpdateRequest(
                                id=item.id,
                                item=ItemUpdateRequest(id=item.item.id),
                                quantity=item.quantity,
                            )
                        )

                if not found:
                    # Add new item (for update, we need to use LineItemUpdateRequest without id)
                    updated_items.append(
                        LineItemUpdateRequest(
                            item=ItemUpdateRequest(id=product.id),
                            quantity=quantity,
                        )
                    )

                update_req = CheckoutUpdateRequest(
                    id=self.checkout_id,
                    currency=existing.currency,
                    line_items=updated_items,
                    payment=PaymentUpdateRequest(instruments=[]),
                )

                checkout = await self.client.update_checkout(
                    self.checkout_id, update_req
                )
            except Exception:
                # Checkout not found, create new
                checkout = await self._create_new_checkout([line_item])
        else:
            checkout = await self._create_new_checkout([line_item])

        self._checkout_cache = checkout
        return checkout

    async def _create_new_checkout(
        self, line_items: list[LineItemCreateRequest]
    ) -> CheckoutResponse:
        """Create a new checkout session.

        Args:
            line_items: Initial line items

        Returns:
            Created checkout session
        """
        create_req = CheckoutCreateRequest(
            currency="USD",
            line_items=line_items,
            payment=PaymentCreateRequest(instruments=[]),
        )
        checkout = await self.client.create_checkout(create_req)
        self.checkout_id = checkout.id
        return checkout

    async def remove_from_checkout(self, product_id: str) -> CheckoutResponse:
        """Remove a product from the checkout.

        Args:
            product_id: Product ID to remove

        Returns:
            Updated checkout session

        Raises:
            ValueError: If no active checkout
        """
        if not self.checkout_id:
            raise ValueError("No active checkout session")

        existing = await self.client.get_checkout(self.checkout_id)

        updated_items = [
            LineItemUpdateRequest(
                id=item.id,
                item=ItemUpdateRequest(id=item.item.id),
                quantity=item.quantity,
            )
            for item in (existing.line_items or [])
            if item.item.id != product_id
        ]

        update_req = CheckoutUpdateRequest(
            id=self.checkout_id,
            currency=existing.currency,
            line_items=updated_items,
            payment=PaymentUpdateRequest(instruments=[]),
        )

        checkout = await self.client.update_checkout(self.checkout_id, update_req)
        self._checkout_cache = checkout
        return checkout

    async def update_checkout_quantity(
        self, product_id: str, quantity: int
    ) -> CheckoutResponse:
        """Update the quantity of a product in checkout.

        Args:
            product_id: Product ID to update
            quantity: New quantity (0 to remove)

        Returns:
            Updated checkout session

        Raises:
            ValueError: If no active checkout
        """
        if not self.checkout_id:
            raise ValueError("No active checkout session")

        if quantity == 0:
            return await self.remove_from_checkout(product_id)

        existing = await self.client.get_checkout(self.checkout_id)

        updated_items = []
        for item in existing.line_items or []:
            if item.item.id == product_id:
                updated_items.append(
                    LineItemUpdateRequest(
                        id=item.id,
                        item=ItemUpdateRequest(id=item.item.id),
                        quantity=quantity,
                    )
                )
            else:
                updated_items.append(
                    LineItemUpdateRequest(
                        id=item.id,
                        item=ItemUpdateRequest(id=item.item.id),
                        quantity=item.quantity,
                    )
                )

        update_req = CheckoutUpdateRequest(
            id=self.checkout_id,
            currency=existing.currency,
            line_items=updated_items,
            payment=PaymentUpdateRequest(instruments=[]),
        )

        checkout = await self.client.update_checkout(self.checkout_id, update_req)
        self._checkout_cache = checkout
        return checkout

    async def get_checkout(self) -> CheckoutResponse | None:
        """Get the current checkout session.

        Returns:
            Current checkout or None if no active session
        """
        if not self.checkout_id:
            return None

        checkout = await self.client.get_checkout(self.checkout_id)
        self._checkout_cache = checkout
        return checkout

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
        """Update customer details and delivery address.

        Args:
            first_name: First name
            last_name: Last name
            street_address: Street address
            address_locality: City
            address_region: State/Region
            postal_code: Postal code
            address_country: Country code
            extended_address: Extended address (suite, apt)
            email: Email address

        Returns:
            Updated checkout session

        Raises:
            ValueError: If no active checkout
        """
        if not self.checkout_id:
            raise ValueError("No active checkout session")

        existing = await self.client.get_checkout(self.checkout_id)

        # Build line items
        line_items = [
            LineItemUpdateRequest(
                id=item.id,
                item=ItemUpdateRequest(id=item.item.id),
                quantity=item.quantity,
            )
            for item in (existing.line_items or [])
        ]

        line_item_ids = [item.id for item in (existing.line_items or [])]

        # Build fulfillment with shipping address
        fulfillment = {
            "methods": [
                {
                    "type": "shipping",
                    "line_item_ids": line_item_ids,
                    "destinations": [
                        {
                            "street_address": street_address,
                            "address_locality": address_locality,
                            "address_region": address_region,
                            "postal_code": postal_code,
                            "address_country": address_country,
                            "first_name": first_name,
                            "last_name": last_name,
                            **(
                                {"extended_address": extended_address}
                                if extended_address
                                else {}
                            ),
                        }
                    ],
                }
            ]
        }

        # Build buyer info
        buyer = None
        if email:
            buyer = Buyer(email=email, first_name=first_name, last_name=last_name)

        update_req = CheckoutUpdateRequest(
            id=self.checkout_id,
            currency=existing.currency,
            line_items=line_items,
            payment=PaymentUpdateRequest(instruments=[]),
            buyer=buyer,
            fulfillment=fulfillment,
        )

        checkout = await self.client.update_checkout(self.checkout_id, update_req)
        self._checkout_cache = checkout
        return checkout

    async def start_payment(self) -> CheckoutResponse | str:
        """Prepare checkout for payment.

        Returns:
            Checkout if ready, or message describing what's missing

        Raises:
            ValueError: If no active checkout
        """
        if not self.checkout_id:
            raise ValueError("No active checkout session")

        checkout = await self.client.get_checkout(self.checkout_id)

        # Check what's missing
        missing = []
        if not checkout.buyer:
            missing.append("buyer email address")
        if not hasattr(checkout, 'fulfillment') or not checkout.fulfillment:
            missing.append("shipping address")

        if missing:
            return f"Please provide: {', '.join(missing)}"

        self._checkout_cache = checkout
        return checkout

    async def complete_checkout(
        self,
        payment_handler_id: str = "mock_payment_handler",
        payment_token: str = "success_token",
    ) -> CheckoutResponse:
        """Complete the checkout with payment.

        Args:
            payment_handler_id: Payment handler ID
            payment_token: Payment token

        Returns:
            Completed checkout with order confirmation

        Raises:
            ValueError: If no active checkout or checkout not ready
        """
        if not self.checkout_id:
            raise ValueError("No active checkout session")

        checkout = await self.client.get_checkout(self.checkout_id)

        if checkout.status != "ready_for_complete":
            raise ValueError(
                f"Checkout not ready. Status: {checkout.status}. "
                "Please add buyer info and shipping address first."
            )

        instrument_id = f"inst_{uuid4().hex[:8]}"
        payment_data = {
            "id": instrument_id,
            "handler_id": payment_handler_id,
            "credential": {"token": payment_token},
        }

        completed = await self.client.complete_checkout(
            self.checkout_id,
            payment_data=payment_data,
            risk_signals={"device_id": "langchain_agent"},
        )

        # Clear checkout after completion
        self.checkout_id = None
        self._checkout_cache = None

        return completed

    async def cancel_checkout(self) -> CheckoutResponse:
        """Cancel the current checkout.

        Returns:
            Cancelled checkout session

        Raises:
            ValueError: If no active checkout
        """
        if not self.checkout_id:
            raise ValueError("No active checkout session")

        checkout = await self.client.cancel_checkout(self.checkout_id)

        # Clear state
        self.checkout_id = None
        self._checkout_cache = None

        return checkout

    async def get_order(self, order_id: str) -> dict[str, Any]:
        """Get order details.

        Args:
            order_id: Order ID

        Returns:
            Order data
        """
        return await self.client.get_order(order_id)

    def clear_session(self) -> None:
        """Clear the current checkout session."""
        self.checkout_id = None
        self._checkout_cache = None
