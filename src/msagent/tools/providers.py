"""Provider search over the third-party directory (CareFinder).

The vendor ships an MCP-style manifest. It is treated as **untrusted input**: the model never
sees the vendor's tool descriptions (ours are written here), and ``load_vendor_manifest`` reports
any instruction-like text they contain. In particular the vendor description asks callers to put
the member's SSN and date of birth in the search query; this adapter does the opposite:

* the model supplies only a specialty;
* network and location are added server-side from the member's record;
* only a 3-digit ZIP area is sent (the vendor matches on the first three digits anyway);
* the gateway's egress guard rejects any argument that contains member identifiers;
* results are re-checked locally for network membership rather than trusting the vendor filter.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import Field

from ..gateway.gateway import Tool, ToolKind, ToolRuntime
from ..phi import project
from ..untrusted import sanitize
from .common import Args

_PROVIDER_ALLOW = {"name": True, "npi": True, "specialty": True, "practice": True, "phone": True,
                   "accepting_new_patients": True,
                   "address": {"line1": True, "city": True, "state": True, "zip": True}}


def load_vendor_manifest(path: Path) -> dict[str, Any]:
    """Inspect the vendor manifest. Returns tool names and any descriptions flagged as carrying
    instructions. Nothing from the manifest is forwarded to the model."""
    if not path.is_file():
        return {"present": False, "tools": [], "flagged": []}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    flagged = []
    for tool in manifest.get("tools", []):
        _, bad = sanitize(tool.get("description", ""))
        if bad:
            flagged.append(tool.get("name"))
    return {"present": True, "tools": [t.get("name") for t in manifest.get("tools", [])],
            "flagged": flagged, "vendor": manifest.get("server", {}).get("vendor")}


def _stem(text: str) -> set[str]:
    return {w[:6] for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 3}


class FindProvidersArgs(Args):
    specialty: str = Field(min_length=3, max_length=60, pattern=r"^[A-Za-z &/-]+$",
                           description="Clinical specialty only, e.g. 'Dermatology'.")
    accepting_new_patients_only: bool = False


def _find_providers(rt: ToolRuntime, args: FindProvidersArgs) -> dict[str, Any]:
    member = rt.member()
    network = member["network_id"]
    zip3 = (member.get("address") or {}).get("zip", "")[:3]
    wanted = _stem(args.specialty)
    raw = rt.directory.search(query=args.specialty, zip_code=f"{zip3}00" if zip3 else None,
                              network_id=network, limit=50)
    results = []
    for p in raw:
        if network not in (p.get("networks") or []):
            continue  # do not trust the vendor's network filter
        if zip3 and not str((p.get("address") or {}).get("zip", "")).startswith(zip3):
            continue
        if not (wanted & _stem(p.get("specialty", ""))):
            continue
        if args.accepting_new_patients_only and not p.get("accepting_new_patients"):
            continue
        results.append(project(p, _PROVIDER_ALLOW))
    return {
        "search": {"specialty": args.specialty, "network_id": network, "zip_area": f"{zip3}xx",
                   "basis": "in-network for the member's plan, near the member's home ZIP area"},
        "count": len(results),
        "providers": results[:10],
        "source": "CareFinder provider directory (third party); network status can change - "
                  "advise the member to confirm with the provider.",
    }


TOOLS = [
    Tool(
        name="find_providers",
        description="Find in-network providers of a specialty near the member's home. The member's "
                    "network and location are applied automatically - pass only the specialty.",
        args=FindProvidersArgs, kind=ToolKind.READ, handler=_find_providers, external=True,
    ),
]
