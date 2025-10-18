"""JSON schema definition for invoice extraction output."""

from typing import List, Optional
from pydantic import BaseModel, Field


class CustomerInfo(BaseModel):
    """Customer information."""

    name: Optional[str] = Field(None, description="Customer name")
    address: Optional[str] = Field(None, description="Customer address")


class InvoiceItem(BaseModel):
    """Single invoice line item."""

    name: str = Field(..., description="Item name or description")
    price: Optional[str] = Field(None, description="Item price")


class InvoiceOutput(BaseModel):
    """Complete invoice extraction output."""

    invoice_number: Optional[str] = Field(None, description="Invoice number")
    date: Optional[str] = Field(None, description="Invoice date")
    customer: CustomerInfo = Field(default_factory=CustomerInfo, description="Customer information")
    items: List[InvoiceItem] = Field(default_factory=list, description="Line items")
    total: Optional[str] = Field(None, description="Total amount")


# Export JSON schema for constrained generation
def get_json_schema() -> dict:
    """
    Get the JSON schema for invoice output.

    Returns:
        JSON schema dictionary compatible with outlines
    """
    return InvoiceOutput.model_json_schema()


# Helper function to create empty invoice
def create_empty_invoice() -> dict:
    """Create an empty invoice structure."""
    return InvoiceOutput().model_dump()


# Validation function
def validate_invoice_output(data: dict) -> tuple[bool, Optional[str]]:
    """
    Validate invoice output against schema.

    Args:
        data: Dictionary to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        InvoiceOutput(**data)
        return True, None
    except Exception as e:
        return False, str(e)
