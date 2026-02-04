"""A2UI system prompt helpers for langchain-ucp.

This module provides system prompt generation for A2UI output formatting.
A2UI is NOT a tool - it's a standardization for output formats. The LLM
generates A2UI JSON directly in its response using a delimiter pattern.

Usage:
    >>> from langchain_ucp.a2ui import get_a2ui_system_prompt
    >>> 
    >>> # Add to your agent's system prompt
    >>> system_prompt = base_prompt + get_a2ui_system_prompt(
    ...     include_schema=True,
    ...     include_commerce_examples=True,
    ... )
"""

import json
import logging
from typing import Any

from langchain_ucp.a2ui.schema import A2UI_MESSAGE_SCHEMA, wrap_as_json_array

logger = logging.getLogger(__name__)

# The delimiter used to separate text content from A2UI JSON in responses
A2UI_DELIMITER = "---a2ui_JSON---"


def _get_commerce_examples() -> str:
    """Generate commerce UI examples using templates.
    
    Uses the helper functions from templates.py to generate consistent examples.
    These are FORMAT TEMPLATES only - actual data must come from tool calls.
    """
    # Import here to avoid circular imports
    from langchain_ucp.a2ui.templates import (
        create_product_card,
        create_product_list,
        create_checkout_ui,
        create_order_confirmation,
    )
    
    examples = []
    
    # Add critical warning at the top
    examples.append("""
⚠️ CRITICAL: The examples below show the A2UI JSON FORMAT/STRUCTURE only.
DO NOT use the example data (product names, prices, image URLs) directly!
You MUST:
1. FIRST call search_shopping_catalog tool to get REAL product data
2. THEN generate A2UI JSON using the ACTUAL data returned from the tool
3. NEVER copy example values - always use real data from tool responses
""")
    
    # Product Card Example - use obvious placeholder values
    product_card = create_product_card(
        product_id="{{PRODUCT_ID_FROM_TOOL}}",
        name="{{PRODUCT_NAME_FROM_TOOL}}",
        price="{{PRICE_FROM_TOOL}}",
        image_url="{{IMAGE_URL_FROM_TOOL}}",
        description="{{DESCRIPTION_FROM_TOOL}}",
    )
    examples.append(f"""
=== PRODUCT CARD FORMAT ===
Format template for a single product card. Replace {{{{...}}}} placeholders with REAL data from search_shopping_catalog tool:

{A2UI_DELIMITER}
{json.dumps(product_card, indent=2)}
""")
    
    # Product List Example - use obvious placeholder values
    products = [
        {"id": "{{PRODUCT_1_ID}}", "name": "{{PRODUCT_1_NAME}}", "price": "{{PRODUCT_1_PRICE}}", "imageUrl": "{{PRODUCT_1_IMAGE_URL}}"},
        {"id": "{{PRODUCT_2_ID}}", "name": "{{PRODUCT_2_NAME}}", "price": "{{PRODUCT_2_PRICE}}", "imageUrl": "{{PRODUCT_2_IMAGE_URL}}"},
        {"id": "{{PRODUCT_N_ID}}", "name": "{{PRODUCT_N_NAME}}", "price": "{{PRODUCT_N_PRICE}}", "imageUrl": "{{PRODUCT_N_IMAGE_URL}}"},
    ]
    product_list = create_product_list(
        title="Search Results",
        products=products,
    )
    examples.append(f"""
=== PRODUCT LIST FORMAT ===
Format template for product list. Replace {{{{...}}}} placeholders with REAL data from search_shopping_catalog tool.
Include ALL products returned by the tool, not just 2-3:

{A2UI_DELIMITER}
{json.dumps(product_list, indent=2)}
""")
    
    # Checkout Example - use placeholder values for cart, EMPTY for address
    items = [
        {"title": "{{ITEM_TITLE_FROM_CART}}", "quantity": "{{QUANTITY}}", "total": "{{ITEM_TOTAL}}"},
    ]
    checkout = create_checkout_ui(
        checkout_id="{{CHECKOUT_ID_FROM_TOOL}}",
        items=items,
        total="{{CART_TOTAL_FROM_TOOL}}",
    )
    examples.append(f"""
=== CHECKOUT FORM FORMAT ===
Format template for checkout form with shipping address fields.

WORKFLOW:
1. Get cart data from get_checkout tool
2. Show this form with:
   - Cart items and total from the tool response
   - ALL shipping address fields EMPTY (firstName="", lastName="", streetAddress="", city="", state="", zipCode="", email="")
3. User fills in the form fields
4. When user submits or provides address, call update_customer_details with their EXACT input
5. ONLY after update_customer_details succeeds, proceed with start_payment and complete_checkout

⚠️ CRITICAL:
- Address fields MUST be empty strings ("") - let user fill them
- NEVER pre-fill with example data like "John", "123 Main St"
- When calling update_customer_details, use the EXACT values user provided
- ⚠️ CHECKOUT CANNOT BE COMPLETED WITHOUT SHIPPING ADDRESS - ask user to provide it first!

{A2UI_DELIMITER}
{json.dumps(checkout, indent=2)}
""")
    
    # Order Confirmation Example - use placeholder values
    confirmation = create_order_confirmation(
        order_id="{{ORDER_ID_FROM_TOOL}}",
        items_summary="{{ITEMS_SUMMARY_FROM_TOOL}}",
        total="{{TOTAL_FROM_TOOL}}",
        shipping_address="{{SHIPPING_ADDRESS_FROM_USER_INPUT}}",
        primary_color="#4CAF50",  # Green for success
    )
    examples.append(f"""
=== ORDER CONFIRMATION FORMAT ===
Format template for order confirmation after successful purchase.
- Order ID, items, and total MUST come from complete_checkout tool response
- Shipping address MUST come from what the user provided earlier
- NEVER use example addresses like "123 Main St"

{A2UI_DELIMITER}
{json.dumps(confirmation, indent=2)}
""")
    
    return "\nCOMMERCE UI EXAMPLES:\n" + "\n".join(examples)


def get_a2ui_system_prompt(
    include_schema: bool = True,
    include_commerce_examples: bool = True,
) -> str:
    """Generate A2UI system prompt for the LLM.

    This function generates instructions for the LLM to output A2UI JSON
    as part of its response, using a delimiter pattern. The LLM should NOT
    call a tool - instead it directly generates the A2UI JSON.

    Args:
        include_schema: Whether to include the full A2UI schema.
        include_commerce_examples: Whether to include commerce-specific examples.

    Returns:
        System prompt string with A2UI instructions.
    """
    prompt_parts = [
        f"""
You can generate rich UIs using the A2UI (Agent-to-User Interface) format.

MANDATORY WORKFLOW:
1. ALWAYS call the appropriate tool FIRST (e.g., search_shopping_catalog for products)
2. WAIT for the tool response with REAL data
3. ONLY THEN generate A2UI JSON using the ACTUAL data from the tool response
4. NEVER generate A2UI with made-up or example data

When you need to show products, checkout information, or order details to the user,
generate A2UI JSON in your response using the following format:

A2UI MESSAGE TYPES:
- beginRendering: Start a new UI surface with root component and styles
- surfaceUpdate: Define the component tree for a surface
- dataModelUpdate: Update data values that components reference
- deleteSurface: Remove a surface

IMPORTANT RULES:
- Each UI must have a beginRendering, surfaceUpdate, and dataModelUpdate message
- The JSON must be a valid array of A2UI messages
- Do NOT wrap the JSON in markdown code blocks after the delimiter
- Only include the delimiter and JSON when you want to render a UI
- Component IDs must be unique and should NOT match data path names (e.g., use "product-name" not "name")
- When displaying product lists, include ALL products from the tool response
- ALL product data (name, price, image_url, id) MUST come from the tool response, NOT from examples
- If a field (like price) is NOT present in the tool response, either omit it or show "N/A" - NEVER make up values

⚠️ CRITICAL - USER INPUT HANDLING:
- When user provides their name, address, or any personal information, you MUST use their EXACT input
- NEVER substitute, modify, or make up names/addresses - use EXACTLY what the user typed
- For update_customer_details tool: pass the EXACT values the user provided, character for character
- Example: If user says "John Smith, 456 Oak Ave", use "John" for first_name, "Smith" for last_name, "456 Oak Ave" for street_address
- NEVER use example names like "Jane Doe" or addresses like "123 Main St" - only use what the user actually typed

RESPONSE FORMAT:
1. Your response MUST be in two parts, separated by the delimiter: `{A2UI_DELIMITER}`
2. The first part is your conversational text response.
3. The second part is a raw JSON array of A2UI messages (no markdown code blocks).

Example response structure:
Here are the flowers you requested!

{A2UI_DELIMITER}
[{{"beginRendering": ...}}, {{"surfaceUpdate": ...}}, {{"dataModelUpdate": ...}}]

"""
    ]

    if include_commerce_examples:
        prompt_parts.append(_get_commerce_examples())

    if include_schema:
        prompt_parts.append(f"""
---BEGIN A2UI JSON SCHEMA---
{json.dumps(wrap_as_json_array(A2UI_MESSAGE_SCHEMA), indent=2)}
---END A2UI JSON SCHEMA---
""")

    return "\n".join(prompt_parts)


def validate_a2ui_json(json_string: str, schema: dict[str, Any] | None = None) -> tuple[bool, list[dict[str, Any]] | None, str | None]:
    """Validate A2UI JSON string against the schema.

    Args:
        json_string: The JSON string to validate.
        schema: Optional custom schema. Uses default A2UI schema if not provided.

    Returns:
        Tuple of (is_valid, parsed_data, error_message).
    """
    if schema is None:
        schema = wrap_as_json_array(A2UI_MESSAGE_SCHEMA)

    try:
        # Clean the JSON string (remove markdown code blocks if present)
        cleaned = json_string.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.lstrip("```json").lstrip("```").rstrip("```").strip()

        # Parse JSON
        parsed = json.loads(cleaned)

        # Auto-wrap single object in list
        if isinstance(parsed, dict):
            parsed = [parsed]

        if not isinstance(parsed, list):
            return False, None, "A2UI JSON must be a list of messages"

        # Validate against schema
        try:
            import jsonschema
            jsonschema.validate(instance=parsed, schema=schema)
        except ImportError:
            logger.warning("jsonschema not installed, skipping validation")
        except jsonschema.exceptions.ValidationError as e:
            return False, None, f"Schema validation failed: {e.message}"

        return True, parsed, None

    except json.JSONDecodeError as e:
        return False, None, f"Invalid JSON: {e}"


def parse_a2ui_response(response: str) -> tuple[str, list[dict[str, Any]] | None]:
    """Parse an LLM response that may contain A2UI JSON.

    Looks for the A2UI_DELIMITER in the response and splits it into
    text content and A2UI JSON.

    Args:
        response: The full LLM response string.

    Returns:
        Tuple of (text_content, a2ui_messages).
        a2ui_messages is None if no A2UI JSON found or if parsing failed.
    """
    if A2UI_DELIMITER not in response:
        return response, None

    parts = response.split(A2UI_DELIMITER, 1)
    text_content = parts[0].strip()
    json_string = parts[1].strip() if len(parts) > 1 else ""

    if not json_string:
        return text_content, None

    is_valid, parsed, error = validate_a2ui_json(json_string)
    if not is_valid:
        logger.warning(f"A2UI JSON validation failed: {error}")
        return text_content, None

    return text_content, parsed


def get_a2ui_schema() -> dict[str, Any]:
    """Get the A2UI message schema wrapped for array validation.

    Returns:
        The A2UI schema as a Python dict, ready for jsonschema validation.
    """
    return wrap_as_json_array(A2UI_MESSAGE_SCHEMA)
