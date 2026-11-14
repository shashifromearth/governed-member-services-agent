"""HTTP clients for the upstream systems.

``MemberApiClient`` is constructed for exactly one tenant: the tenant id is baked into every
path and the bearer token is that tenant's token. There is no method that takes a tenant
argument, so a caller holding a plan_a client cannot express a plan_b request.

Resilience rules (see data/README.md):

* claims platform returns 429 with ``Retry-After`` -> wait as told, bounded retries;
* reads may be retried on 503 / transport errors (they are idempotent);
* the grievance POST is retried **only with the same Idempotency-Key**, because the system of
  record can commit the write and still answer 503.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx


class UpstreamError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code, self.message = status, code, message


class NotFound(UpstreamError):
    pass


class WriteUnconfirmed(Exception):
    """The write may or may not have been committed; retry later with the same key."""

    def __init__(self, idempotency_key: str, attempts: int, last_error: str):
        super().__init__(f"write unconfirmed after {attempts} attempts: {last_error}")
        self.idempotency_key, self.attempts, self.last_error = idempotency_key, attempts, last_error


def _raise_for(resp: httpx.Response) -> None:
    try:
        body = resp.json()
    except ValueError:
        body = {}
    code, message = body.get("error", "error"), body.get("message", resp.text[:200])
    if resp.status_code == 404:
        raise NotFound(404, code, message)
    raise UpstreamError(resp.status_code, code, message)


class MemberApiClient:
    def __init__(self, base_url: str, tenant_id: str, token: str, *, timeout_s: float = 10.0,
                 max_retries: int = 4, sleep: Callable[[float], None] = time.sleep,
                 transport: httpx.BaseTransport | None = None):
        self.tenant_id = tenant_id
        self._prefix = f"/tenants/{tenant_id}"
        self._max_retries = max_retries
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout_s,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    # -- transport ---------------------------------------------------------
    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        last = "no attempt"
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._http.get(path, params=params)
            except httpx.TransportError as exc:
                last = type(exc).__name__
                self._sleep(min(0.2 * 2**attempt, 2.0))
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                last = "429"
                try:
                    wait = float(resp.headers.get("Retry-After", "1"))
                except ValueError:
                    wait = 1.0
                self._sleep(min(max(wait, 0.0), 5.0))
                continue
            if resp.status_code == 503:
                last = "503"
                self._sleep(min(0.2 * 2**attempt, 2.0))
                continue
            _raise_for(resp)
        raise UpstreamError(503, "unavailable", f"gave up after retries ({last})")

    # -- reads -------------------------------------------------------------
    def get_member(self, member_id: str) -> dict[str, Any]:
        return self._get(f"{self._prefix}/members/{member_id}")

    def get_benefits(self, member_id: str) -> dict[str, Any]:
        return self._get(f"{self._prefix}/members/{member_id}/benefits")

    def get_claim(self, claim_id: str) -> dict[str, Any]:
        return self._get(f"{self._prefix}/claims/{claim_id}")

    def list_grievances(self, member_id: str) -> list[dict[str, Any]]:
        return self._get(f"{self._prefix}/members/{member_id}/grievances")["data"]

    def list_claims(self, member_id: str, max_pages: int = 20) -> list[dict[str, Any]]:
        """All claims for the member, following ``next`` until it is null.

        ``next`` is server-supplied, so it is validated before it is followed: it must stay on
        this tenant's claims collection for this member. Anything else is treated as an upstream
        fault rather than followed (a link is data, not an instruction to cross a boundary).
        """
        expected_path = f"{self._prefix}/members/{member_id}/claims"
        items: list[dict[str, Any]] = []
        params: dict[str, Any] | None = {"page_size": 5}
        for _ in range(max_pages):
            page = self._get(expected_path, params)
            items.extend(page["data"])
            nxt = page.get("next")
            if not nxt:
                return items
            parsed = urlparse(nxt)
            if parsed.netloc or parsed.path != expected_path:
                raise UpstreamError(502, "bad_pagination_link", "next link left the member's claims")
            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        raise UpstreamError(502, "pagination_overflow", "too many claim pages")

    # -- write -------------------------------------------------------------
    def create_grievance(self, member_id: str, body: dict[str, Any], idempotency_key: str,
                         max_attempts: int = 4) -> tuple[dict[str, Any], bool]:
        """Create a grievance exactly once. Returns ``(record, replayed)``.

        Every attempt carries the same ``Idempotency-Key`` and byte-identical body, so a retry
        after a 503 or a timeout returns the original record instead of creating a second one.
        If no attempt is acknowledged, ``WriteUnconfirmed`` is raised - the caller must not
        invent a new key.
        """
        path = f"{self._prefix}/members/{member_id}/grievances"
        last = "no attempt"
        for attempt in range(max_attempts):
            try:
                resp = self._http.post(path, json=body, headers={"Idempotency-Key": idempotency_key})
            except httpx.TransportError as exc:
                last = type(exc).__name__
            else:
                if resp.status_code in (200, 201):
                    replayed = resp.headers.get("Idempotent-Replayed", "").lower() == "true"
                    return resp.json(), replayed
                if resp.status_code not in (429, 503):
                    _raise_for(resp)  # 4xx validation / idempotency mismatch: not retryable
                last = str(resp.status_code)
            if attempt < max_attempts - 1:
                self._sleep(min(0.3 * 2**attempt, 2.0))
        raise WriteUnconfirmed(idempotency_key, max_attempts, last)


class DirectoryClient:
    """Third-party provider directory. Only non-PHI search terms are ever sent (see gateway)."""

    def __init__(self, base_url: str, api_key: str, *, timeout_s: float = 10.0,
                 transport: httpx.BaseTransport | None = None):
        self._http = httpx.Client(base_url=base_url, headers={"X-Api-Key": api_key},
                                  timeout=timeout_s, transport=transport)

    def close(self) -> None:
        self._http.close()

    def search(self, *, query: str, specialty: str | None = None, zip_code: str | None = None,
               network_id: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        params = {"query": query, "limit": limit}
        if specialty:
            params["specialty"] = specialty
        if zip_code:
            params["zip"] = zip_code
        if network_id:
            params["network_id"] = network_id
        resp = self._http.get("/providers", params=params)
        if resp.status_code != 200:
            _raise_for(resp)
        return resp.json().get("results", [])


def upstream_clock(base_url: str, timeout_s: float = 3.0) -> str | None:
    """The environment's 'today' (``GET /health``). None if the service is unreachable."""
    try:
        resp = httpx.get(f"{base_url}/health", timeout=timeout_s)
        return resp.json().get("clock") if resp.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None
