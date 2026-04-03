"""Shared engine Protocol: defines the load/unload/ready contract."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class BaseEngine(Protocol):
    """Common interface for all ML engine wrappers."""

    @property
    def ready(self) -> bool: ...

    def load(self) -> None: ...

    def unload(self) -> None: ...
