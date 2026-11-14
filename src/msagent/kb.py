"""Tenant-scoped knowledge base: ingestion, retrieval and structured plan rules.

Isolation is structural, not a query filter:

* there is one ``TenantKB`` (documents + index) per tenant and no shared index, so a query can
  not rank another tenant's text however it is phrased;
* a document is admitted to a tenant only if **both** its location (``kb/<tenant>/``) and its
  front-matter ``tenant`` agree. A disagreement means broken provenance and the document is
  quarantined - it is indexed for nobody until its owner fixes it.

Retrieval is scoped further by product and plan year (a 2027 summary must not answer a 2026
question). Row-level chunking keeps each benefit row together with its column headers.

Numbers the agent needs for arithmetic (deductible, out-of-pocket maximum, visit limits, appeal
and grievance timeframes) are parsed once into ``PlanDesign`` / ``TenantPolicy`` so tools compute
with them deterministically instead of asking the model to do subtraction or date maths.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .tenancy import Tenant, TenantRegistry
from .untrusted import sanitize

# --------------------------------------------------------------------------- documents


@dataclass(frozen=True)
class KBDocument:
    tenant: str
    plan_year: int
    effective_date: str
    product_code: str  # "ALL" applies to every product of the plan
    product: str
    title: str
    source: str  # file name, for citations


@dataclass(frozen=True)
class Chunk:
    doc: KBDocument
    section: str
    text: str

    def as_context(self) -> dict[str, str | int]:
        return {
            "source": self.doc.source,
            "plan_year": self.doc.plan_year,
            "product": self.doc.product,
            "section": self.section,
            "text": self.text,
        }


@dataclass(frozen=True)
class Quarantined:
    path: str
    reason: str


@dataclass(frozen=True)
class PlanDesign:
    """Benefit design of one product for one plan year (in-network, individual)."""

    product_code: str
    plan_year: int
    source: str
    deductible_individual: float | None
    oop_max_individual: float | None
    pt_visit_limit: int | None


@dataclass(frozen=True)
class TenantPolicy:
    appeal_window_days: int | None
    grievance_resolution_days: int | None
    source: str | None


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    _, _, rest = text.partition("\n")
    header, sep, body = rest.partition("\n---")
    if not sep:
        return {}, text
    meta: dict[str, str] = {}
    for line in header.splitlines():
        key, colon, value = line.partition(":")
        if colon:
            meta[key.strip()] = value.strip()
    return meta, body.lstrip("-").lstrip("\n")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\s*\|?[\s:|-]+\|?\s*", line)) and "-" in line


def chunk_document(doc: KBDocument, body: str) -> tuple[list[Chunk], list[dict[str, str]], int]:
    """Split a document into chunks. Returns ``(chunks, table_rows, flagged_count)``.

    ``table_rows`` are the parsed rows (header -> cell) used for structured extraction.
    """
    chunks: list[Chunk] = []
    rows: list[dict[str, str]] = []
    flagged = 0
    section = doc.title
    lines = body.splitlines()
    i = 0

    def add(text: str) -> None:
        nonlocal flagged
        clean, bad = sanitize(text)
        if bad:
            flagged += 1
            return  # instruction-like KB text is dropped, not passed on
        if clean.strip():
            chunks.append(Chunk(doc, section, clean.strip()))

    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("#"):
            section = line.lstrip("#").strip()
        elif line.startswith("|") and i + 1 < len(lines) and _is_separator(lines[i + 1]):
            headers = _cells(line)
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                cells = _cells(lines[i])
                row = dict(zip(headers, cells, strict=False))
                rows.append(row)
                add(" | ".join(f"{h}: {c}" for h, c in row.items()))
                i += 1
            continue
        elif line.strip():
            add(line.lstrip("- ").strip())
        i += 1
    return chunks, rows, flagged


# --------------------------------------------------------------------------- retrieval

_STOP = frozenset(
    "a an and are as at be by can do does for from has have how i in is it its me my of on or "
    "that the their them they this to was what when which who will with would you your "
    "member members member's rep caller plan this's".split()
)
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "er": ("emergency", "room"),
    "video": ("telehealth",),
    "prescription": ("drug",),
    "rx": ("drug",),
    "medication": ("drug",),
    "pt": ("physical", "therapy"),
    "oop": ("pocket", "maximum"),
    "mri": ("imaging",),
    "ct": ("imaging",),
    "hospital": ("inpatient",),
    "doctor": ("primary", "care", "specialist"),
    "pcp": ("primary", "care"),
    "therapy": ("behavioral",),
    "counseling": ("behavioral",),
    "mental": ("behavioral",),
    "complaint": ("grievance",),
}


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        if tok in _STOP:
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


def expand_query(tokens: list[str]) -> list[str]:
    out = list(tokens)
    for tok in tokens:
        out.extend(_SYNONYMS.get(tok, ()))
    return out


class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self._k1, self._b = k1, b
        self._tf = [Counter(doc) for doc in corpus]
        self._len = [len(doc) for doc in corpus]
        self._avg = (sum(self._len) / len(corpus)) if corpus else 0.0
        df: Counter[str] = Counter()
        for doc in corpus:
            df.update(set(doc))
        n = len(corpus)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: list[str], idx: int) -> float:
        tf, length = self._tf[idx], self._len[idx]
        total = 0.0
        for term in query:
            f = tf.get(term)
            if not f:
                continue
            norm = f + self._k1 * (1 - self._b + self._b * length / (self._avg or 1))
            total += self._idf[term] * f * (self._k1 + 1) / norm
        return total


# --------------------------------------------------------------------------- tenant KB

_MONEY = re.compile(r"\$([\d,]+(?:\.\d+)?)")


def _first_money(text: str | None) -> float | None:
    m = _MONEY.search(text or "")
    return float(m.group(1).replace(",", "")) if m else None


def _first_int(text: str | None) -> int | None:
    m = re.search(r"\b(\d+)\b", text or "")
    return int(m.group(1)) if m else None


class TenantKB:
    """All knowledge for one tenant. Instances are never shared between tenants."""

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.documents: list[KBDocument] = []
        self._chunks: list[Chunk] = []
        self._rows: dict[str, list[dict[str, str]]] = {}  # source -> table rows
        self._index: BM25 | None = None
        self.flagged_chunks = 0

    def add(self, doc: KBDocument, body: str) -> None:
        if doc.tenant != self.tenant_id:  # defence in depth; the loader already checked
            raise ValueError(f"document {doc.source} belongs to {doc.tenant}, not {self.tenant_id}")
        chunks, rows, flagged = chunk_document(doc, body)
        self.documents.append(doc)
        self._chunks.extend(chunks)
        self._rows[doc.source] = rows
        self.flagged_chunks += flagged
        self._index = None

    # -- plan years ---------------------------------------------------------
    def current_plan_year(self, product_code: str, as_of: str) -> int | None:
        years = [
            d.plan_year
            for d in self.documents
            if d.product_code in (product_code, "ALL") and d.effective_date <= as_of
        ]
        return max(years) if years else None

    def plan_years(self, product_code: str) -> list[int]:
        return sorted({d.plan_year for d in self.documents if d.product_code == product_code})

    # -- retrieval ----------------------------------------------------------
    def search(self, query: str, *, product_code: str, plan_year: int, k: int = 6) -> list[Chunk]:
        """BM25 over this tenant's chunks, restricted to the member's product and plan year.

        Plan-wide documents (``product_code == ALL``) are eligible for any plan year on or after
        their own: the member-services guide is not re-issued per product.
        """
        if self._index is None:
            self._index = BM25([tokenize(f"{c.section} {c.text}") for c in self._chunks])
        q = expand_query(tokenize(query))
        scored: list[tuple[float, int]] = []
        for idx, chunk in enumerate(self._chunks):
            doc = chunk.doc
            if doc.product_code == product_code:
                if doc.plan_year != plan_year:
                    continue
            elif doc.product_code == "ALL":
                if doc.plan_year > plan_year:
                    continue
            else:
                continue
            score = self._index.score(q, idx)
            if score > 0:
                scored.append((score, idx))
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [self._chunks[idx] for _, idx in scored[:k]]

    # -- structured rules ---------------------------------------------------
    def plan_design(self, product_code: str, plan_year: int) -> PlanDesign | None:
        for doc in self.documents:
            if doc.product_code == product_code and doc.plan_year == plan_year:
                rows = self._rows.get(doc.source, [])
                by_benefit = {r.get("Benefit", "").lower(): r for r in rows}

                def cell(prefix: str, rows_by=by_benefit) -> str | None:
                    for name, row in rows_by.items():
                        if name.startswith(prefix):
                            return row.get("In-network member cost")
                    return None

                return PlanDesign(
                    product_code=product_code,
                    plan_year=plan_year,
                    source=doc.source,
                    deductible_individual=_first_money(cell("deductible")),
                    oop_max_individual=_first_money(cell("out-of-pocket maximum")),
                    pt_visit_limit=_first_int(cell("physical therapy visit limit")),
                )
        return None

    def denial_guidance(self, carc: str) -> str | None:
        """The plan's own 'what to tell the member' text for a denial code, if it publishes one."""
        for doc in self.documents:
            for row in self._rows.get(doc.source, []):
                if row.get("CARC", "").strip().upper() == carc.strip().upper():
                    text, flagged = sanitize(row.get("What to tell the member", ""))
                    return None if flagged or not text else text
        return None

    def policy(self) -> TenantPolicy:
        appeal = grievance = source = None
        for doc in self.documents:
            if doc.product_code != "ALL":
                continue
            for row in self._rows.get(doc.source, []):
                topic = row.get("Topic", "").lower()
                if topic.startswith("claim appeal filing window"):
                    appeal, source = _first_int(row.get("Rule")), doc.source
                elif topic.startswith("grievance resolution timeframe"):
                    grievance, source = _first_int(row.get("Rule")), doc.source
        return TenantPolicy(appeal, grievance, source)


@dataclass
class KnowledgeBase:
    tenants: dict[str, TenantKB] = field(default_factory=dict)
    quarantined: list[Quarantined] = field(default_factory=list)

    def for_tenant(self, tenant_id: str) -> TenantKB:
        return self.tenants[tenant_id]

    @classmethod
    def load(cls, pack_dir: Path, registry: TenantRegistry) -> KnowledgeBase:
        kb = cls()
        for tenant_id in registry.ids():
            tenant: Tenant = registry.get(tenant_id)
            tkb = TenantKB(tenant_id)
            kb.tenants[tenant_id] = tkb
            root = pack_dir / tenant.kb_dir
            if not root.is_dir():
                continue
            for path in sorted(root.glob("*.md")):
                rel = f"{tenant.kb_dir}/{path.name}"
                meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
                required = ("tenant", "plan_year", "effective_date", "product_code", "product")
                missing = [k for k in required if not meta.get(k)]
                if missing:
                    kb.quarantined.append(Quarantined(rel, f"front-matter missing {missing}"))
                    continue
                if meta["tenant"] != tenant_id:
                    kb.quarantined.append(Quarantined(
                        rel, f"front-matter tenant '{meta['tenant']}' does not match folder tenant "
                             f"'{tenant_id}'"))
                    continue
                if meta["product_code"] != "ALL" and meta["product_code"] not in tenant.products:
                    kb.quarantined.append(Quarantined(
                        rel, f"product_code '{meta['product_code']}' is not a product of {tenant_id}"))
                    continue
                title = next((ln.lstrip("# ").strip() for ln in body.splitlines()
                              if ln.startswith("# ")), path.stem)
                tkb.add(KBDocument(
                    tenant=tenant_id,
                    plan_year=int(meta["plan_year"]),
                    effective_date=meta["effective_date"],
                    product_code=meta["product_code"],
                    product=meta["product"],
                    title=title,
                    source=path.name,
                ), body)
        return kb
