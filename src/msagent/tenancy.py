"""Tenant registry and the per-conversation security context.

``TenantContext`` is created once, from the authenticated rep's identity, when a conversation
starts. It is immutable and is the *only* source of tenant and member identity for the rest of
the system: no tool accepts a tenant or member id from the model, and credentials are looked up
by ``TenantContext.tenant_id`` alone.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import Settings


class UnknownTenant(Exception):
    pass


class MissingCredential(Exception):
    pass


@dataclass(frozen=True)
class Tenant:
    tenant_id: str
    name: str
    token_env: str
    kb_dir: str
    products: dict[str, str]  # product_code -> network_id
    directory_enabled: bool = True

    @property
    def network_ids(self) -> frozenset[str]:
        return frozenset(self.products.values())


@dataclass(frozen=True)
class TenantContext:
    """Who is asking, for which tenant, about which member. Bound at conversation start."""

    tenant: Tenant
    member_id: str
    rep_id: str
    conversation_id: str

    @property
    def tenant_id(self) -> str:
        return self.tenant.tenant_id


class TenantRegistry:
    def __init__(self, tenants: dict[str, Tenant], settings: Settings):
        self._tenants = tenants
        self._settings = settings

    @classmethod
    def load(cls, settings: Settings) -> TenantRegistry:
        raw = json.loads(Path(settings.tenants_file).read_text(encoding="utf-8"))
        tenants = {
            tid: Tenant(
                tenant_id=tid,
                name=cfg["name"],
                token_env=cfg["token_env"],
                kb_dir=cfg["kb_dir"],
                products=dict(cfg["products"]),
                directory_enabled=bool(cfg.get("directory_enabled", True)),
            )
            for tid, cfg in raw["tenants"].items()
        }
        return cls(tenants, settings)

    def get(self, tenant_id: str) -> Tenant:
        try:
            return self._tenants[tenant_id]
        except KeyError:
            raise UnknownTenant(tenant_id) from None

    def ids(self) -> list[str]:
        return sorted(self._tenants)

    def others(self, tenant_id: str) -> list[Tenant]:
        return [t for tid, t in self._tenants.items() if tid != tenant_id]

    def api_token(self, tenant_id: str) -> str:
        """Credential for exactly one tenant. There is deliberately no 'all tokens' accessor."""
        tenant = self.get(tenant_id)
        token = self._settings.secret(tenant.token_env)
        if not token:
            raise MissingCredential(f"{tenant.token_env} is not set")
        return token

    def directory_api_key(self) -> str:
        key = self._settings.secret("MSA_DIRECTORY_API_KEY")
        if not key:
            raise MissingCredential("MSA_DIRECTORY_API_KEY is not set")
        return key
