"""Prime Intellect's verifiers environments as environments of this system (docs/implementations/rollout-verifiers.md).

A verifiers v1 environment is a taskset (its tasks and how each is scored) played by a harness (the program the model
runs in) in a runtime, one rollout per task. `VerifiersEnvironment` wraps one as an `Environment`:

- **Rows and starts.** The training data (`train`, settings of the taskset's config: a split, say) is one row; its
  tasks are its starts, one drawn per group with the group's random generator. A start carries the task's data
  whole, so whoever plays it needs no dataset.
- **Eval data.** The tasks of the eval data (`eval`), in order, are one named list of starts: a suite once frozen.
- **An episode** (`VerifiersProgram`) is one verifiers episode, played in its process: verifiers' own interception
  server stands between the harness and the model, and relays the harness's requests, in the harness's own API, to
  the episode's model address. The gateway samples them, so the tokens and logprobs are recorded by this system. The
  episode's reward is the task's (its rewards' weighted sum); its info has each reward and metric, and `solved`.
"""

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import random
import re
import secrets
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import Any

from pydantic import JsonValue

from rollout.contracts import ModelAddress
from rollout.environment import Description, Row, Start
from rollout.harness import Program, ProgramReference, RunContext, register

__all__ = ["VerifiersEnvironment", "VerifiersProgram", "play"]


class VerifiersEnvironment:
    """A verifiers v1 environment: one row of training data whose starts are its tasks, and its eval data's tasks."""

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
        eval_size: int | None = None,
        solved_at: float = 1.0,
    ) -> None:
        """`taskset` is its id (`owner/name` of an Environments Hub package, which must be installed, or a built-in);
        `harness` the id of the harness that plays it. `train` and `eval` are settings of the taskset's config (its
        split, say) for the training and the eval data; without `eval` there is no eval data. `runtime` is where
        each rollout runs (`subprocess` by default); `environment` holds any other settings of verifiers' environment
        config (`timeout`, `retries`). `limit` takes the first training tasks only (an infinite taskset needs it), and
        `eval_size` the first eval tasks. An episode whose reward reaches `solved_at` solved its task."""
        self.taskset, self.harness, self.solved_at = taskset, harness, solved_at
        self.limit, self.eval_size = limit, eval_size
        self.train, self.eval = dict(train or {}), dict(eval or {})
        self.runtime: dict[str, JsonValue] = dict(runtime or {"type": "subprocess"})
        self.environment = dict(environment or {})
        self.program = ProgramReference(program=register(VerifiersProgram))

    def configuration(self, settings: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """verifiers' environment config, for the taskset with `settings` (`train` or `eval`)."""
        taskset: dict[str, JsonValue] = {"id": self.taskset, **settings}
        agent: dict[str, JsonValue] = {"harness": {"id": self.harness}, "runtime": self.runtime}
        return {**self.environment, "taskset": taskset, "agent": agent}

    @property
    def version(self) -> str:
        return self._version

    @property
    def description(self) -> Description:
        return self._description

    @cached_property
    def _version(self) -> str:
        """The taskset's package and its version, verifiers' version, and a digest of the settings."""
        module = self.taskset.rsplit("/", 1)[-1].split("@", 1)[0].replace("-", "_").lower()  # (as verifiers names it)
        verifiers = importlib.metadata.version("verifiers")
        if importlib.util.find_spec(f"verifiers.v1.tasksets.{module}") is not None:
            package = f"verifiers {verifiers}"
        else:
            distribution = (importlib.metadata.packages_distributions().get(module) or [module])[0]
            try:
                package = f"{distribution} {importlib.metadata.version(distribution)}, verifiers {verifiers}"
            except importlib.metadata.PackageNotFoundError:  # (a module of no installed package: its settings say all)
                package = f"{module}, verifiers {verifiers}"
        settings = [self.harness, self.train, self.eval, self.runtime, self.environment, self.limit, self.eval_size]
        digest = hashlib.sha256(json.dumps([*settings, self.solved_at], sort_keys=True).encode()).hexdigest()
        return f"{package}, settings {digest[:8]}"

    @cached_property
    def _description(self) -> Description:
        """Rewards from the sum of the task's reward weights' negative parts to that of their positive parts (verifiers
        scores fall in [0, 1] by convention, not by contract); results say `solved`, and the turns as `duration`."""
        env = loaded_environment(self.configuration(self.train))
        kind: Any = env.taskset.task_type()
        task: Any = kind(kind.data_type().model_validate(self.tasks[0]), env.config.taskset.task)
        weights = [float(getattr(reward, "_vf_weight", 1.0)) for reward in task.hooks("reward")]
        low, high = sum(min(weight, 0.0) for weight in weights), sum(max(weight, 0.0) for weight in weights)
        return Description(rewards=(low, high if high > low else low + 1.0), solved=True, duration="turns")

    def rows(self) -> Sequence[Row]:
        settings = ", ".join(f"{key} {value}" for key, value in self.train.items())
        title = f"{self.taskset} ({settings})" if settings else self.taskset
        return [Row("train", f"{title}: {len(self.tasks)} tasks")]

    def start(self, row: Row, rng: random.Random) -> JsonValue:
        task = self.tasks[rng.randrange(len(self.tasks))]
        return {"environment": self.configuration(self.train), "task": task, "solved_at": self.solved_at}

    def evals(self) -> Mapping[str, Sequence[Start]]:
        """The eval data's tasks, in order, named after the taskset and its eval settings (`gsm8k-test`)."""
        if not self.eval:
            return {}
        words = [self.taskset.rsplit("/", 1)[-1], *map(str, self.eval.values())]
        words += [str(self.eval_size)] if self.eval_size is not None else []
        name = re.sub(r"[^a-z0-9.]+", "-", "-".join(words).lower()).strip("-")
        configuration = self.configuration(self.eval)
        return {
            name: [
                Start("eval", f"task {task['idx']}", int(str(task["idx"])), {
                    "environment": configuration, "task": task, "solved_at": self.solved_at,
                })
                for task in self.eval_tasks
            ]
        }  # fmt: skip

    @cached_property
    def tasks(self) -> list[dict[str, JsonValue]]:
        """Each training task's data, as verifiers sends it to the process that plays it."""
        return self._loaded(self.train, self.limit)

    @cached_property
    def eval_tasks(self) -> list[dict[str, JsonValue]]:
        return self._loaded(self.eval, self.eval_size)

    def _loaded(self, settings: Mapping[str, JsonValue], limit: int | None) -> list[dict[str, JsonValue]]:
        tasks = loaded_environment(self.configuration(settings)).taskset
        if limit is not None:
            tasks = tasks.take(limit)
        elif not tasks.bounded:
            raise ValueError(f"{self.taskset} has tasks without end: give a limit")
        loaded: list[dict[str, JsonValue]] = [task.data.model_dump(mode="json") for task in tasks]
        if not loaded:
            raise ValueError(f"{self.taskset} has no tasks with {dict(settings)}")
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
        await run.emit("result", {**info, "solved": reward >= self.solved_at, "duration": info["turns"]})


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
