"""Command line entry point: ``msagent serve | chat | eval | doctor``."""
from __future__ import annotations

import argparse
import json
import sys

from .config import Settings


def _chat(args: argparse.Namespace) -> int:
    from .agent.service import AgentService

    svc = AgentService.build()
    session = svc.start(args.tenant, args.member, args.rep)
    print(f"{session.ctx.tenant.name} | member {args.member} | model {svc.model.model} | "
          f"today {svc.as_of}\nCommands: /confirm  /cancel  /trace  /quit\n")
    while True:
        try:
            text = input("rep> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text == "/quit":
            return 0
        if text == "/trace":
            for e in session.tracer.export():
                print(f"  {e['turn']}.{e['seq']:<3} {e['kind']:<17} {e['name']:<24} {e['status']:<7} "
                      f"{e['duration_ms'] or '':>8}  {json.dumps(e['attrs'])[:110]}")
            continue
        if text in ("/confirm", "/cancel"):
            draft = session.actions.open_draft() or session.actions.unconfirmed()
            if draft is None:
                print("  (no draft waiting)")
                continue
            reply = (svc.confirm if text == "/confirm" else svc.cancel)(session, draft.action_id)
        else:
            reply = svc.send(session, text)
        print(f"\nassistant> {reply.text}\n  [{reply.outcome} | {reply.usage}]\n")


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("msagent.api:app", host=args.host, port=args.port, log_level="info")
    return 0


def _doctor(_: argparse.Namespace) -> int:
    """Check configuration, credentials, the pack and the knowledge base before a run."""
    from .clients import upstream_clock
    from .kb import KnowledgeBase
    from .tenancy import MissingCredential, TenantRegistry
    from .tools.providers import load_vendor_manifest

    settings = Settings.from_env()
    registry = TenantRegistry.load(settings)
    ok = True
    print(f"pack dir        {settings.pack_dir}")
    print(f"upstream clock  {upstream_clock(settings.api_base_url) or 'UNREACHABLE ' + settings.api_base_url}")
    llm = settings.llm
    if llm.provider == "anthropic" or (llm.provider == "openai" and llm.api_key):
        print(f"model           HOSTED  {llm.provider} / {llm.model}  (an API key was supplied)")
    else:
        import httpx

        try:
            listed = httpx.get(f"{llm.base_url}/models", timeout=5).json().get("data", [])
            present = any(m.get("id") == llm.model for m in listed)
            state = "available" if present else f"NOT PULLED - run: ollama pull {llm.model}"
        except (httpx.HTTPError, ValueError):
            present, state = False, f"model server UNREACHABLE at {llm.base_url}"
        ok = ok and present
        print(f"model           LOCAL   {llm.model} at {llm.base_url}  ({state}; no API key in use)")
    for tid in registry.ids():
        try:
            registry.api_token(tid)
            cred = "credential ok"
        except MissingCredential as exc:
            cred, ok = f"MISSING ({exc})", False
        print(f"tenant {tid:<9} {cred}")
    kb = KnowledgeBase.load(settings.pack_dir, registry)
    for tid in registry.ids():
        tkb = kb.for_tenant(tid)
        print(f"kb {tid:<13} {len(tkb.documents)} documents, {tkb.flagged_chunks} instruction-like "
              f"chunk(s) dropped, policy {tkb.policy()}")
        ok = ok and bool(tkb.documents)
    for q in kb.quarantined:
        print(f"kb QUARANTINED   {q.path}: {q.reason}")
    manifest = load_vendor_manifest(settings.pack_dir / "tools" / "third_party_provider_directory.json")
    print(f"vendor manifest  {manifest}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="msagent", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    chat = sub.add_parser("chat", help="interactive rep conversation")
    chat.add_argument("--tenant", required=True)
    chat.add_argument("--member", required=True)
    chat.add_argument("--rep", default="rep-cli")
    chat.set_defaults(fn=_chat)

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(fn=_serve)

    doctor = sub.add_parser("doctor", help="check config, credentials, pack and KB")
    doctor.set_defaults(fn=_doctor)

    ev = sub.add_parser("eval", help="run the evaluation harness and the regression gate",
                        add_help=False)
    ev.set_defaults(fn=None)

    args, rest = parser.parse_known_args(argv)
    if args.cmd == "eval":
        from .evals.run import main as eval_main

        return eval_main(rest)
    if rest:
        parser.error(f"unrecognised arguments: {' '.join(rest)}")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
