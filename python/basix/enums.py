"""Enumerations."""

from enum import Enum as _Enum

__all__ = ["Enum"]


class Enum(_Enum):
    """An enum with comparisons implemented."""

    def __lt__(self, other) -> bool:
        """Less than."""
        if self.__class__ != other.__class__:
            return NotImplemented
        return self.value < other.value

    def __le__(self, other) -> bool:
        """Less than or equal."""
        if self.__class__ != other.__class__:
            return NotImplemented
        return self.value <= other.value

    def __gt__(self, other) -> bool:
        """Greater than."""
        if self.__class__ != other.__class__:
            return NotImplemented
        return self.value > other.value

    def __ge__(self, other) -> bool:
        """Greater than or equal."""
        if self.__class__ != other.__class__:
            return NotImplemented
        return self.value >= other.value
