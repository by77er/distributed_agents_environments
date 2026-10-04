"""Prime Intellect's verifiers environments as environments of this system (docs/implementations/rollout-verifiers.md).

A verifiers v1 environment is a taskset (its tasks and how each is scored) played by a harness (the program the model
runs in) in a runtime, one rollout per task. `VerifiersEnvironment` wraps one as a catalog:

- **Rows and starts.** A taskset configuration (a split, say) is a row; its tasks are its starts, one drawn per group
  with the group's random generator. A start carries the task's data whole, so whoever plays it needs no dataset.
- **Train and eval data.** The catalog's row is the training data (`train`, settings of the taskset's config);
  `evaluation()` is the same environment over the eval data (`eval`).
- **An episode** (`VerifiersProgram`) is one verifiers episode, played in its process: verifiers' own interception
  server stands between the harness and the model, and relays the harness's requests, in the harness's own API, to
  the episode's model address. The recorder samples them, so the tokens and logprobs are recorded by this system. The
  episode's reward is the task's (its rewards' weighted sum); its info has each reward and metric, and `solved`.
"""

import importlib
import os
import random
import secrets
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import Any

from pydantic import JsonValue

from rollout.catalog import Row
from rollout.contracts import ModelAddress
from rollout.harness import Program, ProgramReference, RunContext, register

__all__ = ["VerifiersEnvironment", "VerifiersProgram", "play"]


class VerifiersEnvironment:
    """A verifiers v1 environment as a catalog: one row of training data whose starts are its tasks."""

    def __init__(
        self,
        taskset: str,
        *,
        harness: str = "null",
        train: Mapping[str, JsonValue] | None = None,
        eval: Mapping[str, JsonValue] | None = None,
        runtime: Mapping[str, JsonValue] | None = None,
        environment: Mapping[str, JsonValue] | None = None,
        limit: int | None = None,
        solved_at: float = 1.0,
        name: str = "train",
    ) -> None:
        """`taskset` is its id (`owner/name` of an Environments Hub package, which must be installed, or a built-in);
        `harness` the id of the harness that plays it. `train` and `eval` are settings of the taskset's config (its
        split, say) for the training and the eval data. `runtime` is where each rollout runs (`subprocess` by
        default); `environment` holds any other settings of verifiers' environment config (`timeout`, `retries`).
        `limit` takes the first tasks only (an infinite taskset needs it). An episode whose reward reaches
        `solved_at` solved its task."""
        self.taskset, self.harness, self.limit, self.solved_at, self.name = taskset, harness, limit, solved_at, name
        self.train, self.eval = dict(train or {}), dict(eval or {})
        self.runtime: dict[str, JsonValue] = dict(runtime or {"type": "subprocess"})
        self.environment = dict(environment or {})
        self.program = ProgramReference(program=register(VerifiersProgram))

    def configuration(self) -> dict[str, JsonValue]:
        """verifiers' environment config for this catalog's data."""
        taskset: dict[str, JsonValue] = {"id": self.taskset, **(self.eval if self.name == "eval" else self.train)}
        agent: dict[str, JsonValue] = {"harness": {"id": self.harness}, "runtime": self.runtime}
        return {**self.environment, "taskset": taskset, "agent": agent}

    def evaluation(self) -> "VerifiersEnvironment":
        """The same environment over its eval data."""
        return VerifiersEnvironment(
            self.taskset, harness=self.harness, train=self.train, eval=self.eval, runtime=self.runtime,
            environment=self.environment, limit=self.limit, solved_at=self.solved_at, name="eval",
        )  # fmt: skip

    def rows(self) -> Sequence[Row]:
        settings = ", ".join(
            f"{key} {value}" for key, value in (self.eval if self.name == "eval" else self.train).items()
        )
        title = f"{self.taskset} ({settings})" if settings else self.taskset
        return [Row(self.name, f"{title}: {len(self.tasks)} tasks")]

    def start(self, row: Row, rng: random.Random) -> JsonValue:
        task = self.tasks[rng.randrange(len(self.tasks))]
        return {"environment": self.configuration(), "task": task, "solved_at": self.solved_at}

    @cached_property
    def tasks(self) -> list[dict[str, JsonValue]]:
        """Each task's data, as verifiers sends it to the process that plays it."""
        tasks = loaded_environment(self.configuration()).taskset
        if self.limit is not None:
            tasks = tasks.take(self.limit)
        elif not tasks.bounded:
            raise ValueError(f"{self.taskset} has tasks without end: give a limit")
        loaded: list[dict[str, JsonValue]] = [task.data.model_dump(mode="json") for task in tasks]
        if not loaded:
            raise ValueError(f"{self.taskset} has no tasks with {self.configuration()['taskset']}")
        return loaded


class VerifiersProgram(Program):
    """One verifiers episode of one task, its model reached at the run's model address."""

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        self.environment: dict[str, Any] = dict(parameters["environment"])
        self.task: dict[str, Any] = dict(parameters["task"])
        self.solved_at = float(parameters.get("solved_at", 1.0))

    async def main(self, run: RunContext) -> None:
        reward, info = await play(self.environment, self.task, run.model.address())
        run.reward(reward)
        await run.emit("result", {**info, "solved": reward >= self.solved_at})


async def play(
    environment: Mapping[str, Any], task: Mapping[str, Any], address: ModelAddress
) -> tuple[float, dict[str, JsonValue]]:
    """Play one episode of `task` in the verifiers environment `environment` describes, its model at `address`:
    its reward, and each of its rewards and metrics. A failed episode raises, with what verifiers said."""
    v1: Any = importlib.import_module("verifiers.v1")
    env = loaded_environment(environment)
    kind: Any = env.taskset.task_type()
    played: Any = kind(kind.data_type().model_validate(dict(task)), env.config.taskset.task)
    # verifiers reads a client's key from the environment; the name keeps it out of the harness's (no `API_KEY`
    # name reaches a subprocess runtime).
    variable = f"ROLLOUT_VERIFIERS_API_KEY_{secrets.token_hex(8)}"
    os.environ[variable] = address.api_key
    try:
        client = v1.EvalClientConfig(base_url=address.base_url, api_key_var=variable)
        async with env.serving():
            episode = await env.run_episode(played, v1.ModelContext(model=address.model, client=client))
    finally:
        os.environ.pop(variable, None)
    if not episode.ok or not episode.traces:
        said = "; ".join(f"{error.type}: {error.message}" for error in episode.errors)
        failed = [f"{error.type}: {error.message}" for trace in episode.traces for error in trace.errors]
        raise RuntimeError(f"the verifiers episode failed: {said or '; '.join(failed) or 'no trace'}")
    (trace,) = episode.traces
    info: dict[str, JsonValue] = {
        "rewards": {name: reward.value if reward else None for name, reward in trace.rewards.items()},
        "metrics": dict(trace.metrics),
        "turns": int(trace.num_turns),
    }
    return float(trace.reward), info


def loaded_environment(configuration: Mapping[str, Any]) -> Any:
    """verifiers' environment for its config. (verifiers ships no type information, so what it gives is `Any`.)"""
    loaders: Any = importlib.import_module("verifiers.v1.utils.loaders")
    return loaders.load_environment(loaders.resolve_env_config(dict(configuration)))
