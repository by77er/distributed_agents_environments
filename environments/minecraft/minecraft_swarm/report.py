"""Report a training run's progress: a chart of the climb through the curriculum, and a summary in words.

Reads what `train` writes (`metrics.jsonl`, `curriculum.json`) and can post both to a Discord webhook, once or after
every iteration (`minecraft-swarm report RUN --watch`). The webhook's address comes from `--webhook` or the
environment variable `DISCORD_WEBHOOK_URL`; it is a secret and is never written anywhere.
"""

import asyncio
import io
import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx

from minecraft_swarm.curriculum import Curriculum
from minecraft_swarm.tasks import Task, catalog

MAX_MESSAGE = 1900
"""Discord accepts 2,000 characters."""


def load(directory: Path) -> list[dict[str, Any]]:
    """The run's iterations, in order (empty if none has finished)."""
    path = directory / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def hours(iterations: Sequence[Mapping[str, Any]]) -> list[float]:
    """When each iteration ended, in hours since the run began: from the times they recorded (groups overlap, so
    their durations do not add up), or, for a run that recorded none, from summed durations."""
    if iterations and all("time" in line for line in iterations):
        began = float(iterations[0]["time"]) - float(iterations[0].get("seconds", 0.0))
        return [(float(line["time"]) - began) / 3600 for line in iterations]
    elapsed: list[float] = []
    total = 0.0
    for line in iterations:
        total += float(line.get("seconds", 0.0))
        elapsed.append(total / 3600)
    return elapsed


def update_of(line: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The trainer's statistics, if this iteration trained (a skipped update is recorded as a sentence)."""
    update = line.get("update")
    return update if isinstance(update, Mapping) else None  # pyright: ignore[reportUnknownVariableType]


def curriculum_of(directory: Path, tasks: Sequence[Task]) -> Curriculum:
    curriculum = Curriculum(tasks)
    if (directory / "curriculum.json").exists():
        curriculum.load(directory / "curriculum.json")
    return curriculum


def summary(directory: Path, iterations: Sequence[Mapping[str, Any]], tasks: Sequence[Task] | None = None) -> str:
    """The run in words: the latest group, the task set with each task's record, and the trainer's statistics."""
    tasks = list(tasks) if tasks is not None else catalog()
    if not iterations:
        return f"**Minecraft swarm · {directory.name}** — no iteration has finished yet."
    last = iterations[-1]
    titles = {task.id: task.title for task in tasks}
    updates = [update for line in iterations if (update := update_of(line)) is not None]
    lines = [
        f"**Minecraft swarm · {directory.name}** — iteration {last['iteration']}, "
        f"{hours(iterations)[-1]:.1f} h in, {len(updates)} updates (adapter step {last.get('adapter_step', 0)})"
    ]
    rewards = [float(reward) for reward in last.get("rewards", [])]
    solved = [bool(value) for value in last.get("solved", [])]
    lines.append(
        f"**Latest group:** {last['task']} ({titles.get(str(last['task']), '?')}) — rewards "
        f"{' / '.join(f'{reward:g}' for reward in sorted(rewards)) or 'none'}"
        f" ({_statistics(rewards)}); solved {sum(solved)}/{len(solved)}"
        + (f"; {last['failed']} episodes failed" if last.get("failed") else "")
    )
    update = update_of(last)
    if update is None:
        lines.append(f"**Update:** {last.get('update', 'none')}")
    else:
        lines.append("**Update:** " + _update(update))
    curriculum = curriculum_of(directory, tasks)
    unlocked = curriculum.unlocked()
    lines.append(f"**Task set:** {len(unlocked)} of {len(tasks)} unlocked")
    for task in unlocked:
        record = curriculum.record(task)
        if record.attempts:
            lines.append(
                f"`{task.id}` {task.title} — {record.attempts} groups, solved {record.success:.0%}, "
                f"mean reward {record.reward:.1f}"
            )
        else:
            lines.append(f"`{task.id}` {task.title} — not tried yet")
    inference: Mapping[str, Any] = last.get("inference") or {}
    if inference:
        lines.append(
            f"**Inference:** {inference.get('tokens_per_second', 0):g} tokens/s while generating "
            f"(mean {inference.get('mean_concurrency', 0):g} agents at once), "
            f"{inference.get('generated_tokens', 0):,} tokens this group"
        )
    text = "\n".join(lines)
    return text if len(text) <= MAX_MESSAGE else text[: MAX_MESSAGE - 1] + "…"


def _statistics(rewards: Sequence[float]) -> str:
    if not rewards:
        return "no episode finished"
    spread = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    return f"mean {statistics.fmean(rewards):.2f}, sd {spread:.2f}"


def _update(update: Mapping[str, Any]) -> str:
    parts = [f"{update.get('sequences', 0):g} turns, {update.get('tokens', 0):g} sampled tokens"]
    if "approx_kl" in update:
        parts.append(f"KL to the sampling policy ≈ {update['approx_kl']:.4f}")
    parts.append(f"clipped {float(update.get('clip_fraction', 0.0)):.1%}")
    parts.append(f"mean ratio {float(update.get('mean_ratio', 1.0)):.4f}")
    if "gradient_norm" in update:
        parts.append(f"gradient norm {update['gradient_norm']:.2f}")
    parts.append(f"loss {float(update.get('loss', 0.0)):.4f}")
    return ", ".join(parts)


def chart(iterations: Sequence[Mapping[str, Any]], tasks: Sequence[Task] | None = None, *, title: str = "") -> bytes:
    """The climb as a PNG: which task each group trained on and how far the curriculum has unlocked; each group's
    rewards; and the trainer's statistics per update."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    tasks = list(tasks) if tasks is not None else catalog()
    numbers = {task.id: index for index, task in enumerate(tasks, start=1)}
    when = hours(iterations)
    made: Any = pyplot.subplots(3, 1, figsize=(9, 9), sharex=True, height_ratios=[3, 2, 2], layout="constrained")
    figure: Any = made[0]
    panels: Any = made[1]
    climb: Any = panels[0]
    reward: Any = panels[1]
    training: Any = panels[2]

    trained = [numbers.get(str(line["task"]), 0) for line in iterations]
    share = [sum(line.get("solved") or [0]) / max(len(line.get("solved") or [0]), 1) for line in iterations]
    climb.step(when, [line.get("unlocked", 0) for line in iterations], where="post", color="#888888", label="unlocked")
    points = climb.scatter(
        when, trained, c=share, cmap="viridis", vmin=0, vmax=1, s=46, zorder=3, edgecolors="#222222", linewidths=0.5
    )
    figure.colorbar(  # beside all three panels, so that their time axes line up
        points, ax=list(panels), label="share of the group that solved it", pad=0.01, shrink=0.4, anchor=(0.0, 1.0)
    )
    climb.set_ylabel(f"task (1 to {len(tasks)}, harder upward)")
    climb.set_ylim(0, max([*trained, *[line.get("unlocked", 0) for line in iterations], 5]) + 1)
    climb.legend(loc="upper left", frameon=False)
    climb.set_title(title or "Climb through the curriculum")

    for at, line in zip(when, iterations, strict=True):
        rewards = [float(value) for value in line.get("rewards", [])]
        if not rewards:
            continue
        reward.vlines(at, min(rewards), max(rewards), color="#9aa5b1", linewidth=2)
        reward.scatter([at] * len(rewards), rewards, color="#9aa5b1", s=10, zorder=2)
        reward.scatter([at], [statistics.fmean(rewards)], color="#d1495b", s=30, zorder=3)
        reward.annotate(
            str(line["task"]), (at, max(rewards)), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7
        )
    reward.set_ylabel("reward per episode\n(mean in red; tasks differ)")

    trainings = [(at, update) for at, line in zip(when, iterations, strict=True) if (update := update_of(line))]
    if trainings:
        times = [at for at, _ in trainings]
        if any("approx_kl" in update for _, update in trainings):
            training.plot(
                times,
                [update.get("approx_kl", float("nan")) for _, update in trainings],
                marker="o",
                color="#00798c",
                label="approximate KL",
            )
        training.plot(
            times,
            [update.get("clip_fraction", 0.0) for _, update in trainings],
            marker="s",
            color="#edae49",
            label="clipped share of tokens",
        )
        training.legend(loc="best", frameon=False, ncols=2)
    else:
        training.text(
            0.5, 0.5, "no update yet", transform=training.transAxes, ha="center", va="center", color="#888888"
        )
    training.set_ylabel("per update")
    training.set_xlabel("hours since the run began")
    for panel in (climb, reward, training):
        panel.grid(True, color="#e5e5e5", linewidth=0.6)
        panel.spines[["top", "right"]].set_visible(False)
    image = io.BytesIO()
    figure.savefig(image, format="png", dpi=130)
    pyplot.close(figure)
    return image.getvalue()


async def post(webhook: str, text: str, image: bytes | None, *, client: httpx.AsyncClient | None = None) -> None:
    """Send a message, with the chart attached, to a Discord webhook."""
    owned = client is None
    client = client or httpx.AsyncClient(timeout=30)
    try:
        payload = {"payload_json": json.dumps({"content": text})}
        files = {"files[0]": ("progress.png", image, "image/png")} if image is not None else None
        if files is None:
            response = await client.post(webhook, json={"content": text})
        else:
            response = await client.post(webhook, data=payload, files=files)
        response.raise_for_status()
    finally:
        if owned:
            await client.aclose()


async def report(directory: Path, webhook: str | None, *, watch: bool = False, interval: float = 30.0) -> None:
    """Write `progress.png` and `progress.md` in the run's directory, and post them if a webhook is given; with
    `watch`, again after every new iteration until the run ends."""
    reported = -1
    while True:
        iterations = load(directory)
        if len(iterations) != reported:
            reported = len(iterations)
            text = summary(directory, iterations)
            image = chart(iterations, title=f"{directory.name}: climb through the curriculum") if iterations else None
            (directory / "progress.md").write_text(text + "\n")
            if image is not None:
                (directory / "progress.png").write_bytes(image)
            if webhook:
                await post(webhook, text, image)
            print(text, flush=True)
        if not watch or _ended(directory):
            return
        await asyncio.sleep(interval)


def _ended(directory: Path) -> bool:
    """Whether the training script has written its exit status (the last line of `train.log`)."""
    log = directory / "train.log"
    if not log.exists():
        return False
    lines = log.read_text(errors="replace").rstrip().splitlines()
    return bool(lines) and lines[-1].startswith("exit ")
