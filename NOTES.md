# Notes

## Decisions and assumptions

- **The model is not trusted with anything that must be right.** Tenant/member binding, write confirmation,
  idempotency, arithmetic ("how much is left"), date rules (appeal window, grievance deadline) and PHI
  filtering are deterministic code in or behind the gateway. The model chooses tools and words the answer.
  This is also what makes a small, cheap model viable (NFR-7).
- **A conversation is one tenant and one member**, fixed when it starts. The rep's tenant comes from their
  credential. Assumption: the desktop opens a conversation per call with the member id from the screen-pop.
- **"Explicit confirmation" means the rep's own message or a Confirm button — after a draft has been read
  back.** "Please file it" in the opening request is a request to draft, not a confirmation. Ambiguous or
  mixed replies ("yes, but hold off") never file.
- **A member not found in the rep's tenant gets a fixed reply with no model call.** The agent never checks
  whether the id exists elsewhere; it has no credential that could.
- **Minimum necessary:** the model never receives name, DOB, SSN, phone, email or street address. It gets
  plan, coverage dates, accumulators and claim/grievance facts. The audit log holds the member id (it must);
  operational traces hold a keyed pseudonym.
- **Thresholds:** zero tolerance on safety invariants and the code-enforced behaviours; 90% overall
  (`evaluation/README.md`). The committed run (3B local model) fails the 90% line; I did not lower it.

## What surprised me in the data

Found by cross-checking the pack against itself before writing code.

1. **Four eval expectations are wrong**, each along an axis the system must get right: EV-23 expects a
   *plan_b* benefit for a plan_a member (cross-tenant), EV-25 uses the *2027* deductible for a 2026 accumulator
   (wrong plan year), EV-37 quotes the *Value HDHP* benefit for a Choice PPO member (wrong product), and EV-34
   expects `PAID` for a claim the API returns as `DENIED`. I answer from the member's own tenant, product,
   current plan year and the system of record, adjudicate the four explicitly, and report both scores. A
   system scoring 40/40 as supplied would be leaking across tenants.
2. **`kb/plan_a/virtual_care_2026.md` is a Prairie Sky document** (front-matter `tenant: plan_b`). Ingestion
   quarantines any document whose folder and front-matter disagree, for both tenants, until its owner fixes it.
3. **Three planted prompt injections:** a provider-portal claim note telling the assistant to file an urgent
   grievance without asking; a "processing note for AI coding assistants" in the plan_a member-services
   guide; and the vendor's tool description, which asks for the member's SSN and DOB in every directory query
   and says not to mention it. None is followed; each is covered by a test.
4. **One member id exists in both tenants** (different people) — a shared cache keyed by member id would
   serve the wrong person.
5. **The grievance API can commit a write and still return 503** (for a quarter of member ids, on the first
   attempt). A naive retry files twice; both `confirm_then_file` cases hit this path.
6. **NFRs 1/2 and 3 conflict with each other and with the claims platform** (1.5 s per call, 5 per page,
   periodic 429). I say so in the design document rather than claim to meet them.

## With another day

- Run the eval on the intended hosted model (not done: no credential on the dev machine) and gate CI on it.
- Durable session/draft/idempotency store and a reconciliation job for unacknowledged writes.
- Duplicate-issue check against the member's open grievances before drafting ("one grievance per issue").
- Paraphrased and adversarial variants of the 40 cases; shared per-tenant claims rate limiter; streaming.

## How I used AI tools

The whole submission was built with **Claude Code** (Anthropic; Claude Opus 5.5 and Fable 5.2) as the
implementation tool, working from the brief: reading the pack, proposing the architecture, writing code,
tests and documents, and running them. The commit trailers say so.

**Where it helped**
- Cross-checking all 40 cases against the mock data and the KB *before* designing anything. That is what
  surfaced the wrong expectations, the misfiled document, the duplicate member id and the commit-then-503
  behaviour, and it shaped the design (structured plan rules, quarantine, payload-derived idempotency key).
- The fake upstream and the adversarial test suite, where a scripted model plays the attacker.
- First drafts of the design document and ADRs.

**Where it was wrong, and how that was caught**
- It logged the rep's message *before* the member's identifiers were registered for scrubbing, so a name typed
  by the rep reached the trace. Caught by the test that asserts no PHI in traces; the order was fixed.
- Its first log-scrubbing list included the bare surname, which would also have removed unrelated providers
  with the same surname from tool results. Narrowed to log lines only.
- It defaulted the upstream URL to `localhost`; on Windows that cost ~2 s per call (IPv6 fallback) and made
  latency numbers meaningless. Found in the trace durations.
- It initially took the eval expectations as ground truth for one case (EV-34); the first full run showed the
  agent answering `DENIED` correctly and "failing".
- The small local model printed tool calls as text and invented a `member_id` argument. The schemas stayed
  closed; the adapter learned to read an inline tool call instead.
