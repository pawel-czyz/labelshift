"""Utilities for dealing with filesystem IO."""
import petname
from datetime import datetime


def generate_name() -> str:
    """Generates a name with timestamp and a random part."""

    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y%m%d-%H%M%S")
    suffix = petname.generate(separator="-", words=3)
    return f"{date_time}-{suffix}"
