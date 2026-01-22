"""UCP Exception classes."""

from typing import Any


class UCPError(Exception):
    """Base exception for UCP errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        details: Any = None,
    ):
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)


class UCPVersionError(UCPError):
    """Raised when UCP version is incompatible."""

    def __init__(self, client_version: str, merchant_version: str):
        self.client_version = client_version
        self.merchant_version = merchant_version
        super().__init__(
            f"UCP version {client_version} is not supported. "
            f"Merchant implements version {merchant_version}."
        )


class UCPValidationError(UCPError):
    """Raised when server returns a validation error."""

    def __init__(
        self,
        message: str,
        field_errors: list[dict[str, Any]] | None = None,
    ):
        self.field_errors = field_errors or []
        super().__init__(message)

    def __str__(self) -> str:
        if self.field_errors:
            errors = "; ".join(
                f"{e.get('field', 'unknown')}: {e.get('message', 'invalid')}"
                for e in self.field_errors
            )
            return f"Validation error: {errors}"
        return self.message


class UCPNotFoundError(UCPError):
    """Raised when a resource is not found."""

    pass


class UCPRequestError(UCPError):
    """Raised when request fails."""

    pass
