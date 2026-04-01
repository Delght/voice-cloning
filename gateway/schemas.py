"""Shared Pydantic schemas for Gateway responses."""

from pydantic import BaseModel


class HealthStatus(BaseModel):
    service: str
    url: str
    status: str
    detail: dict | str | None = None


class GatewayHealth(BaseModel):
    gateway: str = "ok"
    services: list[HealthStatus]
