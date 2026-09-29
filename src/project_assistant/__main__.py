"""`project-assistant serve --repository PATH`: run the project assistant's HTTP API on the local Codex login."""

import argparse
from pathlib import Path

import uvicorn

from project_assistant.http import create_app
from project_assistant.service import AssistantService, Settings
from rollout.adapters.responses import codex_provider


def main() -> None:
    parser = argparse.ArgumentParser(prog="project-assistant")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="serve the HTTP API")
    serve.add_argument("--repository", type=Path, default=Path.cwd())
    serve.add_argument("--model", default="gpt-6-astra")
    serve.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high"])
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8420)
    arguments = parser.parse_args()

    settings = Settings(
        repository=arguments.repository, model=arguments.model, reasoning_effort=arguments.reasoning_effort
    )
    service = AssistantService(settings, providers={"codex": codex_provider()})
    uvicorn.run(create_app(service), host=arguments.host, port=arguments.port)


if __name__ == "__main__":
    main()
