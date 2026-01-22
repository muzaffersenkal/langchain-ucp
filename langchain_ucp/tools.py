"""LangChain tools for UCP operations."""

import logging
from abc import abstractmethod
from typing import Any, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_ucp.store import UCPStore

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Formatters
# -----------------------------------------------------------------------------


def format_price(cents: int) -> str:
    """Format price from cents to dollar string."""
    return f"${cents / 100:.2f}"


def format_checkout_summary(checkout: Any) -> str:
    """Format checkout response into a readable summary."""
    lines = [
        f"**Checkout ID:** {checkout.id}",
        f"**Status:** {checkout.status}",
        f"**Currency:** {checkout.currency}",
    ]

    if checkout.line_items:
        lines.append("\n**Items in Cart:**")
        for item in checkout.line_items:
            price = item.item.price if item.item.price else 0
            lines.append(
                f"  - {item.item.title} x{item.quantity} @ {format_price(price)} each"
            )

    if checkout.totals:
        lines.append("\n**Totals:**")
        for total in checkout.totals:
            display = getattr(total, "display_text", total.type.title())
            lines.append(f"  - {display}: {format_price(total.amount)}")

    if checkout.order:
        lines.append("\n**Order Confirmed!**")
        lines.append(f"  - Order ID: {checkout.order.id}")
        if checkout.order.permalink_url:
            lines.append(f"  - Order URL: {checkout.order.permalink_url}")

    return "\n".join(lines)


def format_order_summary(order: dict[str, Any]) -> str:
    """Format order response into a readable summary."""
    lines = [
        f"**Order ID:** {order.get('id', 'N/A')}",
        f"**Checkout ID:** {order.get('checkout_id', 'N/A')}",
    ]

    if order.get("line_items"):
        lines.append("\n**Items:**")
        for item in order["line_items"]:
            item_data = item.get("item", {})
            qty = item.get("quantity", {})
            total_qty = qty.get("total", 0) if isinstance(qty, dict) else qty
            title = item_data.get("title", "Unknown")
            status = item.get("status", "unknown")
            lines.append(f"  - {title} x{total_qty} - Status: {status}")

    if order.get("totals"):
        lines.append("\n**Totals:**")
        for total in order["totals"]:
            lines.append(
                f"  - {total.get('type', '').title()}: {format_price(total.get('amount', 0))}"
            )

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Input Schemas
# -----------------------------------------------------------------------------


class SearchCatalogInput(BaseModel):
    """Input for searching the product catalog."""

    query: str = Field(description="Search query for finding products")


class AddToCheckoutInput(BaseModel):
    """Input for adding a product to checkout."""

    product_id: str = Field(description="The product ID to add")
    quantity: int = Field(default=1, description="Quantity to add")


class RemoveFromCheckoutInput(BaseModel):
    """Input for removing a product from checkout."""

    product_id: str = Field(description="The product ID to remove")


class UpdateCheckoutInput(BaseModel):
    """Input for updating product quantity in checkout."""

    product_id: str = Field(description="The product ID to update")
    quantity: int = Field(description="New quantity (0 to remove)")


class UpdateCustomerDetailsInput(BaseModel):
    """Input for updating customer details."""

    first_name: str = Field(description="First name of the recipient")
    last_name: str = Field(description="Last name of the recipient")
    street_address: str = Field(description="Street address")
    address_locality: str = Field(description="City/locality")
    address_region: str = Field(description="State/region code")
    postal_code: str = Field(description="Postal/ZIP code")
    address_country: str = Field(default="US", description="Country code")
    extended_address: Optional[str] = Field(default=None, description="Suite/apt number")
    email: Optional[str] = Field(default=None, description="Email address")


class CompleteCheckoutInput(BaseModel):
    """Input for completing checkout."""

    payment_handler_id: str = Field(
        default="mock_payment_handler",
        description="Payment handler ID",
    )
    payment_token: str = Field(
        default="success_token",
        description="Payment token",
    )


class GetOrderInput(BaseModel):
    """Input for getting order details."""

    order_id: str = Field(description="The order ID to look up")


# -----------------------------------------------------------------------------
# Base Tool
# -----------------------------------------------------------------------------


class UCPBaseTool(BaseTool):
    """Base class for UCP tools with common functionality."""

    store: UCPStore = Field(exclude=True)
    verbose: bool = Field(default=False, exclude=True)

    def _log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            logger.debug(f"[{self.name}] {message}")

    def _run(self, *args, **kwargs) -> str:
        """Sync run - not supported for async operations."""
        raise NotImplementedError("Use async version")

    @abstractmethod
    async def _arun(self, *args, **kwargs) -> str:
        """Async run - must be implemented by subclasses."""
        pass


# -----------------------------------------------------------------------------
# Tool Implementations
# -----------------------------------------------------------------------------


class SearchCatalogTool(UCPBaseTool):
    """Tool for searching the product catalog."""

    name: str = "search_shopping_catalog"
    description: str = (
        "Searches the product catalog for products that match the given query. "
        "Use this tool to find products before adding them to the cart. "
        "Returns matching products with their IDs and titles."
    )
    args_schema: Type[BaseModel] = SearchCatalogInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Search catalog synchronously."""
        self._log(f"Searching for: {query}")
        result = self.store.search_products(query)

        if not result.products:
            self._log(f"No products found for '{query}'")
            return f"No products found for '{query}'."

        self._log(f"Found {result.total} products")
        lines = [f"Found {result.total} product(s) for '{query}':\n"]
        lines.extend(
            f"  - **{p.title}** (Product ID: `{p.id}`)"
            for p in result.products
        )
        return "\n".join(lines)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Search catalog asynchronously."""
        return self._run(query, run_manager)


class AddToCheckoutTool(UCPBaseTool):
    """Tool for adding a product to the checkout session."""

    name: str = "add_to_checkout"
    description: str = (
        "Adds a product to the checkout session. "
        "Creates a new checkout if one doesn't exist. "
        "Use search_shopping_catalog first to find product IDs."
    )
    args_schema: Type[BaseModel] = AddToCheckoutInput

    async def _arun(
        self,
        product_id: str,
        quantity: int = 1,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Add to checkout asynchronously."""
        self._log(f"Adding product_id={product_id}, quantity={quantity}")
        try:
            product = self.store.get_product(product_id)
            if not product:
                self._log(f"Product not found: {product_id}")
                return f"Product '{product_id}' not found. Use search_shopping_catalog to find available products."

            checkout = await self.store.add_to_checkout(product_id, quantity)
            self._log(f"Added to checkout_id={checkout.id}")
            return f"Added {quantity}x {product.title} to cart.\n\n{format_checkout_summary(checkout)}"
        except Exception as e:
            logger.exception("Error adding to checkout")
            return f"Error adding to cart: {e}"


class RemoveFromCheckoutTool(UCPBaseTool):
    """Tool for removing a product from the checkout session."""

    name: str = "remove_from_checkout"
    description: str = "Removes a product from the checkout session."
    args_schema: Type[BaseModel] = RemoveFromCheckoutInput

    async def _arun(
        self,
        product_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Remove from checkout asynchronously."""
        self._log(f"Removing product_id={product_id}")
        try:
            checkout = await self.store.remove_from_checkout(product_id)
            self._log(f"Removed from checkout_id={checkout.id}")
            return f"Removed item from cart.\n\n{format_checkout_summary(checkout)}"
        except ValueError as e:
            return str(e)
        except Exception as e:
            logger.exception("Error removing from checkout")
            return f"Error removing from cart: {e}"


class UpdateCheckoutTool(UCPBaseTool):
    """Tool for updating product quantity in the checkout session."""

    name: str = "update_checkout"
    description: str = (
        "Updates the quantity of a product in the checkout session. "
        "Set quantity to 0 to remove the item."
    )
    args_schema: Type[BaseModel] = UpdateCheckoutInput

    async def _arun(
        self,
        product_id: str,
        quantity: int,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Update checkout asynchronously."""
        self._log(f"Updating product_id={product_id}, quantity={quantity}")
        try:
            checkout = await self.store.update_checkout_quantity(product_id, quantity)
            action = "removed from" if quantity == 0 else "updated in"
            self._log(f"Item {action} checkout_id={checkout.id}")
            return f"Item {action} cart.\n\n{format_checkout_summary(checkout)}"
        except ValueError as e:
            return str(e)
        except Exception as e:
            logger.exception("Error updating checkout")
            return f"Error updating cart: {e}"


class GetCheckoutTool(UCPBaseTool):
    """Tool for retrieving the current checkout session."""

    name: str = "get_checkout"
    description: str = "Retrieves the current checkout session with all items and totals."

    async def _arun(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get checkout asynchronously."""
        self._log("Getting current checkout")
        try:
            checkout = await self.store.get_checkout()
            if not checkout:
                self._log("No active checkout")
                return "No active checkout session. Add items first."
            self._log(f"Retrieved checkout_id={checkout.id}")
            return format_checkout_summary(checkout)
        except Exception as e:
            logger.exception("Error getting checkout")
            return f"Error getting cart: {e}"


class UpdateCustomerDetailsTool(UCPBaseTool):
    """Tool for updating customer details and delivery address."""

    name: str = "update_customer_details"
    description: str = (
        "Adds delivery address and buyer details to the checkout. "
        "Provide the recipient's name, full address, and optionally email. "
        "This prepares the checkout for payment."
    )
    args_schema: Type[BaseModel] = UpdateCustomerDetailsInput

    async def _arun(
        self,
        first_name: str,
        last_name: str,
        street_address: str,
        address_locality: str,
        address_region: str,
        postal_code: str,
        address_country: str = "US",
        extended_address: Optional[str] = None,
        email: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Update customer details asynchronously."""
        self._log(f"Updating customer: {first_name} {last_name}, {address_locality}, {address_region}")
        try:
            checkout = await self.store.update_customer_details(
                first_name=first_name,
                last_name=last_name,
                street_address=street_address,
                address_locality=address_locality,
                address_region=address_region,
                postal_code=postal_code,
                address_country=address_country,
                extended_address=extended_address,
                email=email,
            )
            self._log(f"Updated customer details for checkout_id={checkout.id}")
            return f"Updated customer details.\n\n{format_checkout_summary(checkout)}"
        except ValueError as e:
            return str(e)
        except Exception as e:
            logger.exception("Error updating customer details")
            return f"Error updating customer details: {e}"


class StartPaymentTool(UCPBaseTool):
    """Tool for preparing checkout for payment."""

    name: str = "start_payment"
    description: str = (
        "Prepares the checkout for payment. "
        "Call this after adding items and customer details. "
        "Returns the checkout status and any missing information."
    )

    async def _arun(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Start payment asynchronously."""
        self._log("Starting payment process")
        try:
            result = await self.store.start_payment()
            if isinstance(result, str):
                self._log(f"Payment not ready: {result}")
                return f"Checkout is not ready. {result}"
            self._log(f"Checkout ready for payment, checkout_id={result.id}")
            return (
                f"Checkout is ready for payment!\n\n"
                f"{format_checkout_summary(result)}\n\n"
                "Use complete_checkout to finalize the order."
            )
        except ValueError as e:
            return str(e)
        except Exception as e:
            logger.exception("Error starting payment")
            return f"Error preparing payment: {e}"


class CompleteCheckoutTool(UCPBaseTool):
    """Tool for completing checkout and placing the order."""

    name: str = "complete_checkout"
    description: str = (
        "Processes the payment and completes the checkout. "
        "Requires buyer info and shipping address to be set first. "
        "Use 'mock_payment_handler' with 'success_token' for testing."
    )
    args_schema: Type[BaseModel] = CompleteCheckoutInput

    async def _arun(
        self,
        payment_handler_id: str = "mock_payment_handler",
        payment_token: str = "success_token",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Complete checkout asynchronously."""
        self._log(f"Completing checkout with handler={payment_handler_id}")
        try:
            checkout = await self.store.complete_checkout(
                payment_handler_id=payment_handler_id,
                payment_token=payment_token,
            )
            order_id = checkout.order.id if checkout.order else "N/A"
            self._log(f"Order completed! order_id={order_id}")
            return f"Order placed successfully!\n\n{format_checkout_summary(checkout)}"
        except ValueError as e:
            return str(e)
        except Exception as e:
            logger.exception("Error completing checkout")
            return f"Error completing checkout: {e}"


class CancelCheckoutTool(UCPBaseTool):
    """Tool for cancelling the current checkout session."""

    name: str = "cancel_checkout"
    description: str = "Cancels the current checkout session and clears the cart."

    async def _arun(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Cancel checkout asynchronously."""
        self._log("Cancelling checkout")
        try:
            checkout = await self.store.cancel_checkout()
            self._log(f"Cancelled checkout_id={checkout.id}")
            return f"Checkout cancelled.\n\n{format_checkout_summary(checkout)}"
        except ValueError as e:
            return str(e)
        except Exception as e:
            logger.exception("Error cancelling checkout")
            return f"Error cancelling checkout: {e}"


class GetOrderTool(UCPBaseTool):
    """Tool for getting order details."""

    name: str = "get_order"
    description: str = "Gets details of a placed order by ID."
    args_schema: Type[BaseModel] = GetOrderInput

    async def _arun(
        self,
        order_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get order asynchronously."""
        self._log(f"Getting order_id={order_id}")
        try:
            order = await self.store.get_order(order_id)
            self._log("Retrieved order details")
            return format_order_summary(order)
        except Exception as e:
            logger.exception("Error getting order")
            return f"Error getting order: {e}"
