"""`project-assistant serve --repository PATH`: run the project assistant's HTTP API on the local Codex login."""

import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path

import uvicorn

from project_assistant.evaluation.harness import EvaluationSettings, evaluate, save, summarize
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
    evaluate = commands.add_parser("evaluate", help="run the evaluation scenarios")
    evaluate.add_argument("--model", default="gpt-6-astra")
    evaluate.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high"])
    evaluate.add_argument("--repeats", type=int, default=1)
    evaluate.add_argument("--concurrency", type=int, default=3)
    evaluate.add_argument("--no-judge", action="store_true")
    evaluate.add_argument("--scenario", action="append", default=[], help="run only this scenario (repeatable)")
    evaluate.add_argument("--output", type=Path, default=None, help="where to save the results as JSON")
    arguments = parser.parse_args()

    if arguments.command == "evaluate":
        asyncio.run(_evaluate(arguments))
        return
    settings = Settings(
        repository=arguments.repository, model=arguments.model, reasoning_effort=arguments.reasoning_effort
    )
    service = AssistantService(settings, providers={"codex": codex_provider()})
    uvicorn.run(create_app(service), host=arguments.host, port=arguments.port)


async def _evaluate(arguments: argparse.Namespace) -> None:
    settings = EvaluationSettings(
        model=arguments.model,
        reasoning_effort=arguments.reasoning_effort,
        repeats=arguments.repeats,
        concurrency=arguments.concurrency,
        judge=not arguments.no_judge,
        scenarios=arguments.scenario,
    )
    runs = await evaluate(settings, providers={"codex": codex_provider()})
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = arguments.output or Path(".rollout") / "evaluations" / f"{stamp}.json"
    save(runs, output)
    print(summarize(runs))
    print(f"\nResults: {output}")


if __name__ == "__main__":
    main()
