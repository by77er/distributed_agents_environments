"""`agents`: run and manage agent sessions.

    agents serve [--state DIR] [--port 8421]      run the server (sessions survive restarts)
                 [--environment local]        sessions work on this machine, not in a sandbox
    agents list                                   sessions and their status
    agents new NAME "instructions"                start a session
    agents send NAME "text" [--urgent]            message a session (urgent interrupts it)
    agents tail NAME [--follow]                   a session's activity
    agents stop NAME                              stop a session; its environment is destroyed
    agents board [--channel C] [--status S]       the board
    agents post CHANNEL "title" "body" [--task]   post to the board as the operator
    agents inbox                                  messages sessions sent you

The client commands talk to the server at $AGENTS_URL (default http://127.0.0.1:8421).
"""

import argparse
import importlib
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx

DEFAULT_URL = "http://127.0.0.1:8421"
STATUS_MARK = {"working": "●", "waiting": "○", "sleeping": "z", "starting": "◌", "stopped": "-"}


def main() -> None:
    parser = argparse.ArgumentParser(prog="agents", description="Run and manage agent sessions.")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="run the server")
    serve.add_argument("--state", type=Path, default=Path.home() / ".local" / "state" / "agent-sessions")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8421)
    serve.add_argument("--model", default="gpt-6-astra")
    serve.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high"])
    serve.add_argument("--in-memory", action="store_true", help="use the LocalRunner: nothing survives a restart")
    serve.add_argument(
        "--environment",
        choices=["namespaces", "local"],
        default="namespaces",
        help="namespaces: a private Alpine computer per session; local: a workspace directory on this machine, "
        "with no sandbox",
    )
    serve.add_argument(
        "--evict-after", type=float, default=300, help="seconds of waiting before a session is unloaded from memory"
    )
    serve.add_argument(
        "--database", help="a Postgres URL shared by several servers, which share --state too (default: SQLite)"
    )
    serve.add_argument(
        "--blobs", help="where images are kept: s3://bucket/prefix (endpoint, credentials: AWS_*); default: --state"
    )
    serve.add_argument("--runner-id", help="this server's name among those sharing --database; keep it across restarts")
    serve.add_argument("--takeover-after", type=float, default=15, help=argparse.SUPPRESS)
    serve.add_argument("--providers", help=argparse.SUPPRESS)  # module:attribute of model providers, for tests
    serve.add_argument("--model-ledger", type=Path, help=argparse.SUPPRESS)
    serve.add_argument("--command-ledger", type=Path, help=argparse.SUPPRESS)
    faults = commands.add_parser("faults", help="kill the server during a fan-out and check nothing was lost")
    faults.add_argument("--kills", type=int, default=3)
    faults.add_argument("--seed", type=int, default=1)
    faults.add_argument("--environment", choices=["namespaces", "local"], default="namespaces")
    faults.add_argument("--servers", type=int, default=1, help="servers sharing a throwaway Postgres (needs pgembed)")
    commands.add_parser("list", help="sessions and their status")
    new = commands.add_parser("new", help="start a session")
    new.add_argument("name")
    new.add_argument("instructions")
    send = commands.add_parser("send", help="message a session")
    send.add_argument("name")
    send.add_argument("text")
    send.add_argument("--urgent", action="store_true", help="interrupt what the session is doing")
    tail = commands.add_parser("tail", help="a session's activity")
    tail.add_argument("name")
    tail.add_argument("-f", "--follow", action="store_true")
    stop = commands.add_parser("stop", help="stop a session")
    stop.add_argument("name")
    board = commands.add_parser("board", help="the board")
    board.add_argument("--channel")
    board.add_argument("--status", choices=["open", "claimed", "done"])
    post = commands.add_parser("post", help="post to the board")
    post.add_argument("channel")
    post.add_argument("title")
    post.add_argument("body")
    post.add_argument("--task", action="store_true")
    commands.add_parser("inbox", help="messages sessions sent you")
    arguments = parser.parse_args()

    if arguments.command == "serve":
        _serve(arguments)
        return
    if arguments.command == "faults":
        import asyncio

        from agent_sessions.faults import run_faults

        report = asyncio.run(
            run_faults(
                kills=arguments.kills, seed=arguments.seed, environment=arguments.environment, servers=arguments.servers
            )
        )
        print(json.dumps(report, indent=2))
        sys.exit(0 if report["passed"] else 1)
    client = httpx.Client(base_url=os.environ.get("AGENTS_URL", DEFAULT_URL), timeout=60)
    try:
        _client_command(client, arguments)
    except httpx.ConnectError:
        sys.exit(f"cannot reach the server at {client.base_url}; start it with `agents serve`")


def _serve(arguments: argparse.Namespace) -> None:
    import uvicorn

    from agent_sessions.http import create_app
    from agent_sessions.service import SessionsService, Settings
    from rollout_openai import codex_provider

    settings = Settings(
        state=arguments.state,
        model=arguments.model,
        reasoning_effort=arguments.reasoning_effort,
        durable=not arguments.in_memory,
        model_ledger=arguments.model_ledger,
        command_ledger=arguments.command_ledger,
        evict_after=timedelta(seconds=arguments.evict_after) if arguments.evict_after > 0 else None,
        environment=arguments.environment,
        database=arguments.database,
        blob_store=arguments.blobs,
        runner_id=arguments.runner_id,
        takeover_after=timedelta(seconds=arguments.takeover_after),
    )
    if arguments.providers:
        module, _, attribute = arguments.providers.partition(":")
        providers = getattr(importlib.import_module(module), attribute)
    else:
        providers = {"codex": codex_provider(blobs=settings.blobs())}
    service = SessionsService(settings, providers=providers)
    print(f"agent sessions: state in {settings.state}", file=sys.stderr)
    uvicorn.run(create_app(service), host=arguments.host, port=arguments.port, log_level="warning")


def _client_command(client: httpx.Client, arguments: argparse.Namespace) -> None:
    command = arguments.command
    if command == "list":
        sessions = _json(client.get("/sessions"))["sessions"]
        for session in sessions:
            mark = STATUS_MARK.get(session["status"], "?")
            parent = session["parent"] or "?"
            print(f"{mark} {session['name']:<20} {session['status']:<9} from {parent:<14} {session['purpose'][:60]}")
        if not sessions:
            print('no sessions; start one with `agents new NAME "instructions"`')
    elif command == "new":
        _json(client.post("/sessions", json={"name": arguments.name, "instructions": arguments.instructions}))
        print(f"started {arguments.name}")
    elif command == "send":
        _json(
            client.post(
                f"/sessions/{arguments.name}/messages", json={"text": arguments.text, "urgent": arguments.urgent}
            )
        )
        print("sent")
    elif command == "tail":
        if arguments.follow:
            with client.stream("GET", f"/sessions/{arguments.name}/stream", timeout=None) as response:
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        _print_entry(json.loads(line[6:]))
        else:
            for entry in _json(client.get(f"/sessions/{arguments.name}/activity"))["activity"]:
                _print_entry(entry)
    elif command == "stop":
        stopped = _json(client.post(f"/sessions/{arguments.name}/stop"))["stopped"]
        print("stopped" if stopped else "not running")
    elif command == "board":
        params = {key: value for key in ("channel", "status") if (value := getattr(arguments, key))}
        for post in _json(client.get("/board", params=params))["posts"]:
            owner = f", {post['claimed_by']}" if post["claimed_by"] else ""
            state = f"{post['kind']} ({post['status']}{owner})"
            print(f"#{post['id']} [{post['channel']}] {state} {post['author']}: {post['title']}")
            print(f"    {post['body']}")
            if post["result"]:
                print(f"    result: {post['result']}")
    elif command == "post":
        body = {"channel": arguments.channel, "title": arguments.title, "body": arguments.body}
        print(_json(client.post("/board", json={**body, "kind": "task" if arguments.task else "note"}))["result"])
    elif command == "inbox":
        for message in _json(client.get("/inbox"))["messages"]:
            print(f"{message['at']}  {message['from']}: {message['text']}")


def _print_entry(entry: dict[str, Any]) -> None:
    time = entry["at"][11:19]
    if entry["kind"] == "message":
        print(f"{time}  ← {entry['from']}: {entry['text']}")
    elif entry["kind"] == "reply":
        print(f"{time}  → {entry['text']}")
    elif entry["kind"] == "tools":
        print(f"{time}    ⚙ {entry['text']}")
    else:
        print(f"{time}  ■ {entry['text']}")
    sys.stdout.flush()


def _json(response: httpx.Response) -> dict[str, Any]:
    if response.status_code >= 400:
        sys.exit(f"error {response.status_code}: {response.text}")
    return response.json()


if __name__ == "__main__":
    main()
