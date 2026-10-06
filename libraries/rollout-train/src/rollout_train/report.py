"""Report a training run's progress: a chart of the climb through the curriculum, and a summary in words.

Reads the run's results, steps and checkpoints from its ledger and can post both to a Discord webhook, once
or after every group (`rollout report RUN ENVIRONMENT --watch`). The webhook's address comes from `--webhook` or the
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

from rollout.curriculum import Curriculum
from rollout.environment import Row
from rollout_train.checkpoints import Checkpoint, checkpoints_in
from rollout_train.ledger import of_run
from rollout_train.record import Result, Trained, results, trained
from rollout_train.registry import registry_of, run_of

MAX_MESSAGE = 1900
"""Discord accepts 2,000 characters."""


def hours(lines: Sequence[Result]) -> list[float]:
    """When each group ended, in hours since the run began (groups overlap, so their durations do not add up)."""
    if not lines:
        return []
    began = lines[0].time - lines[0].rollout_seconds
    return [(line.time - began) / 3600 for line in lines]


def summary(
    name: str,
    lines: Sequence[Result],
    curriculum: Curriculum,
    steps: Mapping[int, Trained] | None = None,
    checkpoints: Mapping[str, Checkpoint] | None = None,
) -> str:
    """The run in words: the latest group, what was done with it, and each unlocked row's record. `steps` says
    what was done with each group, and `checkpoints` holds the checkpoints those steps made, by id."""
    if not lines:
        return f"**{name}** — no group has finished yet."
    steps, checkpoints = steps or {}, checkpoints or {}
    last = lines[-1]
    titles = {row.key: row.title for row in curriculum.rows}
    names = {outcome.checkpoint for outcome in steps.values() if outcome.checkpoint}
    made: list[str] = sorted((name for name in names if name in checkpoints), key=lambda name: checkpoints[name].made)
    serving = f" (serving {made[-1]})" if made else ""
    text = [f"**{name}** — group {last.group}, {hours(lines)[-1]:.1f} h in, {len(made)} updates{serving}"]
    text.append(
        f"**Latest group:** {last.task} ({titles.get(last.task, '?')}) — rewards "
        f"{' / '.join(f'{reward:g}' for reward in sorted(last.rewards)) or 'none'}"
        f" ({_statistics(last.rewards)}); solved {sum(last.solved)}/{len(last.solved)}"
        + (f"; {last.failed} episodes failed" if last.failed else "")
    )
    outcome = steps.get(last.group)
    checkpoint = checkpoints.get(outcome.checkpoint or "") if outcome is not None else None
    if checkpoint is not None:
        text.append("**Update:** " + _update(last, checkpoint))
    elif outcome is not None and outcome.error:
        text.append(f"**Update:** failed: {outcome.error}")
    elif last.segments:
        text.append(f"**Update:** {last.segments} segments waiting for a step")
    else:
        text.append(f"**Update:** skipped: {last.skipped or ''}")
    unlocked = curriculum.unlocked()
    text.append(f"**Rows:** {len(unlocked)} of {len(curriculum.rows)} unlocked")
    for row in unlocked:
        record = curriculum.record(row)
        if record.attempts:
            text.append(
                f"`{row.key}` {row.title} — {record.attempts} groups, solved {record.success:.0%}, "
                f"mean reward {record.reward:.1f}"
            )
        else:
            text.append(f"`{row.key}` {row.title} — not tried yet")
    joined = "\n".join(text)
    return joined if len(joined) <= MAX_MESSAGE else joined[: MAX_MESSAGE - 1] + "…"


def _statistics(rewards: Sequence[float]) -> str:
    if not rewards:
        return "no episode finished"
    spread = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    return f"mean {statistics.fmean(rewards):.2f}, sd {spread:.2f}"


def _update(line: Result, checkpoint: Checkpoint) -> str:
    update = checkpoint.metrics
    of = f" of {line.segments_recorded}" if line.segments_recorded else ""
    trained = update.get("segments", line.segments)
    tokens = update.get("tokens", 0)
    parts = [f"{checkpoint.id[:8]} (depth {checkpoint.depth}), {trained:g}{of} segments, {tokens:g} sampled tokens"]
    if "kl_moved" in update:
        parts.append(f"moved the policy by KL ≈ {update['kl_moved']:.4f} (floor {update.get('kl_floor', 0.0):.4f})")
        parts.append(f"{update.get('optimizer_steps', 0):g} steps")
    parts.append(f"clipped {update.get('clip_fraction', 0.0):.1%}")
    parts.append(f"mean ratio {update.get('mean_ratio', 1.0):.4f}")
    if "gradient_norm" in update:
        parts.append(f"gradient norm {update['gradient_norm']:.2f}")
    if "segment_tokens_per_second" in update:
        packed = f" in {update.get('packs', 0):g} packs" if update.get("packed") else " a segment at a time"
        parts.append(f"{update['segment_tokens_per_second']:,.0f} tokens/s{packed}")
    parts.append(f"loss {update.get('loss', 0.0):.4f}")
    if took := _seconds(checkpoint):
        parts.append(took)
    return ", ".join(parts)


TIMINGS = (
    ("step_seconds", "the step"),
    ("train_seconds", "training"),
    ("save_adapter_seconds", "saving the weights"),
    ("snapshot_seconds", "copying the state"),
    ("upload_adapter_seconds", "uploading the weights"),
)
"""The stages a step's metrics time, and what a step's line calls each."""


def _seconds(checkpoint: Checkpoint) -> str:
    """Where a step's seconds went, as its metrics and its state's completion say: empty where they say nothing."""
    said = [f"{name} {checkpoint.metrics[key]:.1f}" for key, name in TIMINGS if key in checkpoint.metrics]
    if checkpoint.state_seconds is not None:
        said.append(f"keeping the state {checkpoint.state_seconds:.1f} after")
    elif not checkpoint.state_complete:
        said.append("the state not kept yet")
    return f"seconds: {', '.join(said)}" if said else ""


def chart(
    lines: Sequence[Result], rows: Sequence[Row], checkpoints: Sequence[Checkpoint] = (), *, title: str = ""
) -> bytes:
    """The climb as a PNG: which row each group trained on and how far the curriculum has unlocked; each group's
    rewards; and the trainer's statistics per update (each of `checkpoints`, when it was made)."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    numbers = {row.key: index for index, row in enumerate(rows, start=1)}
    when = hours(lines)
    made: Any = pyplot.subplots(3, 1, figsize=(9, 9), sharex=True, height_ratios=[3, 2, 2], layout="constrained")
    figure: Any = made[0]
    panels: Any = made[1]
    climb: Any = panels[0]
    reward: Any = panels[1]
    training: Any = panels[2]

    trained = [numbers.get(line.task, 0) for line in lines]
    share = [sum(line.solved) / max(len(line.solved), 1) for line in lines]
    climb.step(when, [line.unlocked for line in lines], where="post", color="#888888", label="unlocked")
    points = climb.scatter(
        when, trained, c=share, cmap="viridis", vmin=0, vmax=1, s=46, zorder=3, edgecolors="#222222", linewidths=0.5
    )
    figure.colorbar(  # beside all three panels, so that their time axes line up
        points, ax=list(panels), label="share of the group that solved it", pad=0.01, shrink=0.4, anchor=(0.0, 1.0)
    )
    climb.set_ylabel(f"row (1 to {len(rows)}, harder upward)")
    climb.set_ylim(0, max([*trained, *[line.unlocked for line in lines], 5]) + 1)
    climb.legend(loc="upper left", frameon=False)
    climb.set_title(title or "Climb through the curriculum")

    for at, line in zip(when, lines, strict=True):
        rewards = line.rewards
        if not rewards:
            continue
        reward.vlines(at, min(rewards), max(rewards), color="#9aa5b1", linewidth=2)
        reward.scatter([at] * len(rewards), rewards, color="#9aa5b1", s=10, zorder=2)
        reward.scatter([at], [statistics.fmean(rewards)], color="#d1495b", s=30, zorder=3)
        reward.annotate(
            line.task, (at, max(rewards)), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7
        )
    reward.set_ylabel("reward per episode\n(mean in red; rows differ)")

    began = lines[0].time - lines[0].rollout_seconds if lines else 0.0
    trainings = [
        ((checkpoint.made - began) / 3600, checkpoint.metrics) for checkpoint in checkpoints if checkpoint.made >= began
    ]
    if trainings:
        times = [at for at, _ in trainings]
        if any("kl_moved" in update for _, update in trainings):
            training.plot(
                times,
                [update.get("kl_moved", float("nan")) for _, update in trainings],
                marker="o",
                color="#00798c",
                label="KL the update moved the policy",
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


async def report(
    directory: Path,
    rows: Sequence[Row],
    webhook: str | None,
    *,
    run: str | None = None,
    watch: bool = False,
    interval: float = 30.0,
) -> None:
    """Write `progress.png` and `progress.md` in the run's directory, and post them if a webhook is given; with
    `watch`, again after every new group, until interrupted. The run's results, steps and checkpoints are read from
    its ledger, wherever its directory says it is; the run is `run` (its id), by default the one in the directory."""
    ledger = of_run(directory)
    entry = await run_of(directory, ledger, registry_of(ledger))
    run, title = run or entry.id, entry.name
    reported: tuple[int, int] = (-1, -1)
    while True:
        lines = await results(ledger, run)
        steps = await trained(ledger, run)
        checkpoints = [checkpoint for checkpoint in await checkpoints_in(ledger) if checkpoint.run == run]
        if (len(lines), len(checkpoints)) != reported:
            reported = (len(lines), len(checkpoints))
            curriculum = Curriculum(rows)
            for line in lines:
                curriculum.recorded(line)
            text = summary(title, lines, curriculum, steps, {checkpoint.id: checkpoint for checkpoint in checkpoints})
            image = chart(lines, rows, checkpoints, title=f"{title}: climb through the curriculum") if lines else None
            (directory / "progress.md").write_text(text + "\n")
            if image is not None:
                (directory / "progress.png").write_bytes(image)
            if webhook:
                await post(webhook, text, image)
            print(text, flush=True)
        if not watch:
            return
        await asyncio.sleep(interval)
