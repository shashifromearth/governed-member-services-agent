# Governed multi-tenant member-services agent

An assistant for health-plan member-services reps. A rep, on a call with one member of one client plan, asks
about benefits and claims and can have a grievance filed on the member's behalf. Two client plans (`plan_a`,
`plan_b`) share the platform; their data never mixes.

The interesting part is not the model loop. It is the **tool gateway** every call goes through, which makes
the guarantees hold *even when the model is wrong or manipulated*:

| Guarantee | How it is enforced (not "prompted") |
|---|---|
| Tenant isolation | Tenant and member are bound from the rep's identity. No tool has a tenant or member argument. One credential, one KB index, one trace partition per tenant. |
| PHI minimum necessary | Allow-list projection per tool; the SSN never leaves the gateway (match / no match only); redaction before any log sink; egress guard on third-party calls. |
| Confirmed, exactly-once writes | The model can only *draft*. Confirmation comes from the rep's own words or a Confirm button. The gateway files a frozen payload with a payload-derived idempotency key and retries only with that key. |
| Budgets | Steps, cost and latency per conversation, checked before every model and tool call. |
| Traceability | Every model call, tool call, gateway decision and guardrail is a trace event; writes and member access go to a hash-chained audit log. |

- Design document, threat model, ADRs, cloud mapping: [`docs/design.md`](docs/design.md)
- Decisions, surprises in the data, next steps, AI usage: [`NOTES.md`](NOTES.md)
- Latest evaluation run: [`reports/eval_report.md`](reports/eval_report.md) · gate thresholds and rationale: [`evaluation/README.md`](evaluation/README.md)

## Layout

```
src/msagent/
  gateway/gateway.py     the policy pipeline (budget → registry → schema → rep intent → write gate → egress PHI → execute → scope → PHI backstop)
  gateway/actions.py     two-phase write state machine, idempotency key
  gateway/budget.py      per-conversation step / cost / latency budgets
  tools/                 coverage, plan documents, claims, grievances, SSN last-4 check, provider search
  agent/service.py       orchestration: member resolution, confirmation, retrieval, model/tool loop, output guards
  agent/confirmation.py  confirm / cancel / unclear, from the rep's own words only
  kb.py                  per-tenant ingestion (with quarantine), row-level retrieval, structured plan rules
  phi.py, untrusted.py   projection, redaction, PHI detection; neutralising instruction-like text in data
  clients.py             tenant-bound API client: 429/503 handling, pagination-link validation, idempotent POST
  llm/                   provider-neutral interface; Anthropic and OpenAI-compatible adapters
  tracing.py             redacting tracer, tenant-partitioned JSONL sink, hash-chained audit log
  api.py, cli.py         HTTP API and CLI
  evals/                 harness, deterministic scoring, regression gate, report
evaluation/              gate thresholds, adjudicated cases, rationale
tests/                   99 tests on a synthetic KB and a fake upstream (no pack or model needed)
config/tenants.json      non-secret tenant registry (credentials are referenced by env var name)
```

## Setup

Python 3.11+ (developed on 3.15).

```bash
python -m venv .venv
```
```bash
.venv/Scripts/pip install -e ".[dev]"        # Linux/macOS: .venv/bin/pip
```
```bash
cp .env.example .env
```

Then edit `.env`:

1. **Pack location** – `MSA_PACK_DIR` is the folder that contains `kb/`, `eval/`, `tools/` and
   `mock_services/`. The default is the repository root, so you can also just copy the pack into the clone
   (those paths are git-ignored).
2. **Upstream credentials** – `MSA_TOKEN_PLAN_A`, `MSA_TOKEN_PLAN_B`, `MSA_DIRECTORY_API_KEY`; the values for
   the mock are in the pack's `data/README.md`. They are read from the environment only and are never
   committed.
3. **Model** – any of:

| Provider | Settings |
|---|---|
| Anthropic API | `MSA_LLM_PROVIDER=anthropic`, `MSA_LLM_MODEL=claude-haiku-5-5`, `ANTHROPIC_API_KEY=...` |
| OpenAI, Azure OpenAI, vLLM, any OpenAI-compatible endpoint | `MSA_LLM_PROVIDER=openai`, `MSA_LLM_MODEL=...`, `MSA_LLM_BASE_URL=.../v1`, `MSA_LLM_API_KEY=...` |
| Ollama (open weights, local) | `MSA_LLM_PROVIDER=ollama`, `MSA_LLM_MODEL=qwen2.5:3b` (base URL defaults to `http://localhost:11434/v1`) |

   For models not in the built-in price table, set `MSA_LLM_PRICE_IN_PER_MTOK` / `MSA_LLM_PRICE_OUT_PER_MTOK`
   so cost budgets and the cost gate are meaningful. On a CPU-only machine also raise `MSA_LLM_TIMEOUT_S` and
   `MSA_BUDGET_TURN_DEADLINE_S`.

## Run

**1. Start the mock** (from the pack):

```bash
python mock_services/server.py --port 8090
```

**2. Check the configuration** – credentials, pack, knowledge base, vendor manifest:

```bash
msagent doctor
```

**3a. Talk to the agent (CLI)** – a conversation is bound to one tenant and one member:

```bash
msagent chat --tenant plan_a --member M704649
```

Type the rep's messages; `/confirm` and `/cancel` act on a drafted grievance (typing "yes, go ahead and file
it" works too); `/trace` prints every step of the conversation so far.

**3b. Or run the HTTP API:**

```bash
msagent serve --port 8000
```

Rep keys map to a tenant via `MSA_REP_KEYS` (see `.env.example`). The tenant always comes from the key, never
from the request body.

```bash
curl -s -X POST localhost:8000/v1/conversations -H "Authorization: Bearer dev-rep-a" -H "Content-Type: application/json" -d '{"member_id":"M704649"}'
```
```bash
curl -s -X POST localhost:8000/v1/conversations/<id>/messages -H "Authorization: Bearer dev-rep-a" -H "Content-Type: application/json" -d '{"text":"Please file a billing grievance: the provider billed the member in full for the 2026-08-14 office visit."}'
```
```bash
curl -s -X POST localhost:8000/v1/conversations/<id>/actions/<action_id>/confirm -H "Authorization: Bearer dev-rep-a"
```
```bash
curl -s localhost:8000/v1/conversations/<id>/trace -H "Authorization: Bearer dev-rep-a"
```

OpenAPI docs are served at `/docs`.

**4. Tests** (no pack, mock or model required):

```bash
pytest
```

**5. Evaluation and regression gate:**

```bash
msagent eval
```

Runs all cases in `<pack>/eval/conversations.json`, prints a line per case, writes
`reports/eval_report.md` and `reports/eval_summary.json`, and exits 1 if the gate fails.
`--only EV-16,EV-32` runs a subset; `--no-gate` reports without failing. Restart the mock before a full run
so grievance state starts clean.

Runtime state (traces, audit log) goes to `var/`, and per-case debugging detail to `reports/local/`; both are
git-ignored because they are derived from member data.

## How I know it works

1. **Tests – 99, all deterministic.** They drive the *real* gateway, tools, KB, tracing and API over a fake
   upstream that reproduces the documented failure modes (429 with `Retry-After`, a grievance POST that
   commits and still returns 503, pagination). A scripted model plays the adversary: it tries to file without
   confirmation, pass a tenant argument, read another member's claim, follow an instruction planted in a
   claim note, send an SSN to the directory vendor, read out an SSN, loop forever. Each attempt is asserted to
   fail at a named policy.
2. **Evaluation – the 40 supplied conversations against a real model**, scored on observed state (grievances
   actually on file before/after) and on safety invariants checked for every case. See the
   [report](reports/eval_report.md).
3. **A gate that fails the build** on any safety violation, on any miss in the write/privacy/isolation
   behaviours, or when quality drops below the thresholds justified in [`evaluation/README.md`](evaluation/README.md).

## Model used

- **Coding assistant:** Claude Code (Anthropic; Claude Opus 5.5 and Fable 5.2). See `NOTES.md`.
- **Model the agent ran on during development and for the committed report:** `qwen2.5:3b` (open weights,
  3B parameters) through Ollama's OpenAI-compatible endpoint, on a CPU-only laptop. No hosted model
  credential was available on the development machine.
- **Intended production model:** a small hosted model (default `claude-haiku-5-5`); the Anthropic adapter is
  unit-tested for wire format but has **not** been run against the live API.

### Reading the committed report honestly

Last run: **30 / 40 (75%) adjudicated, 26 / 40 as supplied, 0 safety-invariant violations — gate FAILED** on
the overall and per-tenant pass-rate thresholds. I am committing the failing report rather than lowering the
thresholds to fit a 3B model.

- Everything enforced by code passed at 100%: confirm-then-file (both cases went through the
  commit-then-503 path and ended with exactly one grievance), no-write, the injected claim note, declining
  to read out an SSN, and both wrong-tenant cases.
- All 10 misses are `answer`/`cannot_answer` cases where the small model skipped a tool, looped on a
  malformed call until the per-turn model-call budget stopped it, returned nothing twice, dropped a denial code from its wording, or did not lead
  with "Yes"/"No". None is a retrieval, tool or gateway fault: the correct value was in the model's context
  or one tool call away in each case.
- **Latency numbers from this run say little.** End-to-end p95 is ~2 minutes because the model runs on CPU.
  Platform p95 (0.37 s) is flattered for the same reason: the background claims prefetch finishes while the
  model is still thinking. With a hosted model, expect platform time on cold multi-page claim questions to be
  several seconds (see the NFR-3 discussion in the design document).
- Measured prompt size: ~4.9k input and ~80 output tokens per conversation, about **$0.0006 per conversation**
  at `claude-haiku-5-5` list price.

To reproduce with a hosted model: set `MSA_LLM_PROVIDER`, `MSA_LLM_MODEL` and the key in `.env`, restart the
mock, run `msagent eval`.

## Slice versus target design

The slice implements the request path end to end. These parts of [`docs/design.md`](docs/design.md) are
described there and deliberately not built here:

| Area | In this slice | Target |
|---|---|---|
| Rep authentication | static rep keys → tenant | OIDC token with tenant claim |
| Session / draft / idempotency store | in process memory | DynamoDB, tenant-prefixed keys, TTL |
| Trace and audit sinks | local JSONL (tenant-partitioned; audit hash-chained) | CloudWatch/S3; S3 Object Lock, 6-year retention |
| Secrets | environment variables via one accessor | Secrets Manager, one secret per tenant |
| Claims quota | honour `Retry-After`, prefetch, per-conversation cache | plus a shared per-tenant token bucket |
| Unacknowledged writes | reported to the rep, safe manual retry with the same key | plus an automatic reconciliation job |
| Responses | whole reply | streamed |
