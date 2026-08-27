"""Helpers for consistent image identifiers across persistence layers."""


def canonical_image_id(image_id: str) -> str:
    """Remove leading zeroes from numeric IDs while preserving named IDs."""
    value = str(image_id).strip()
    if value.isdecimal():
        return str(int(value))
    return value
