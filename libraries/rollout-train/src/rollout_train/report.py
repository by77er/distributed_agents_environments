"""Report a training run's progress: a chart of the climb through the curriculum, and a summary in words.

Reads the run's iterations from its ledger and can post both to a Discord webhook, once
or after every group (`rollout report RUN CATALOG --watch`). The webhook's address comes from `--webhook` or the
environment variable `DISCORD_WEBHOOK_URL`; it is a secret and is never written anywhere.
"""

import asyncio
import io
import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from rollout.catalog import Row
from rollout_train.curriculum import Curriculum
from rollout_train.layout import LEDGER
from rollout_train.ledger import FileLedger, Ledger
from rollout_train.record import Iteration, iterations

MAX_MESSAGE = 1900
"""Discord accepts 2,000 characters."""


def hours(lines: Sequence[Iteration]) -> list[float]:
    """When each group ended, in hours since the run began (groups overlap, so their durations do not add up)."""
    if not lines:
        return []
    began = lines[0].time - lines[0].seconds
    return [(line.time - began) / 3600 for line in lines]


def summary(name: str, lines: Sequence[Iteration], curriculum: Curriculum) -> str:
    """The run in words: the latest group, what the update did, and each unlocked row's record."""
    if not lines:
        return f"**{name}** — no group has finished yet."
    last = lines[-1]
    titles = {row.key: row.title for row in curriculum.rows}
    updates = sum(1 for line in lines if line.update is not None)
    serving = next((f" (serving {line.adapter})" for line in reversed(lines) if line.adapter), "")
    text = [f"**{name}** — group {last.iteration}, {hours(lines)[-1]:.1f} h in, {updates} updates{serving}"]
    text.append(
        f"**Latest group:** {last.task} ({titles.get(last.task, '?')}) — rewards "
        f"{' / '.join(f'{reward:g}' for reward in sorted(last.rewards)) or 'none'}"
        f" ({_statistics(last.rewards)}); solved {sum(last.solved)}/{len(last.solved)}"
        + (f"; {last.failed} episodes failed" if last.failed else "")
    )
    if last.update is not None:
        text.append("**Update:** " + _update(last))
    else:
        text.append(f"**Update:** {'failed: ' + last.error if last.error else 'skipped: ' + (last.skipped or '')}")
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


def _update(line: Iteration) -> str:
    update = line.update or {}
    of = f" of {line.segments_recorded}" if line.segments_recorded else ""
    parts = [f"{line.segments_trained}{of} segments, {update.get('tokens', 0):g} sampled tokens"]
    if "kl_moved" in update:
        parts.append(f"moved the policy by KL ≈ {update['kl_moved']:.4f} (floor {update.get('kl_floor', 0.0):.4f})")
        parts.append(f"{update.get('optimizer_steps', 0):g} steps")
    parts.append(f"clipped {update.get('clip_fraction', 0.0):.1%}")
    parts.append(f"mean ratio {update.get('mean_ratio', 1.0):.4f}")
    if "gradient_norm" in update:
        parts.append(f"gradient norm {update['gradient_norm']:.2f}")
    parts.append(f"loss {update.get('loss', 0.0):.4f}")
    return ", ".join(parts)


def chart(lines: Sequence[Iteration], rows: Sequence[Row], *, title: str = "") -> bytes:
    """The climb as a PNG: which row each group trained on and how far the curriculum has unlocked; each group's
    rewards; and the trainer's statistics per update."""
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

    trainings = [(at, line.update) for at, line in zip(when, lines, strict=True) if line.update is not None]
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
    ledger: Ledger | None = None,
    run: str = "train",
    watch: bool = False,
    interval: float = 30.0,
) -> None:
    """Write `progress.png` and `progress.md` in the run's directory, and post them if a webhook is given; with
    `watch`, again after every new group, until interrupted. The run's iterations are read from `ledger` (by
    default the one in files under the run's directory)."""
    ledger = ledger or FileLedger(directory / LEDGER)
    reported = -1
    while True:
        lines = await iterations(ledger, run)
        if len(lines) != reported:
            reported = len(lines)
            curriculum = Curriculum(rows)
            for line in lines:
                curriculum.recorded(line)
            text = summary(directory.name, lines, curriculum)
            image = chart(lines, rows, title=f"{directory.name}: climb through the curriculum") if lines else None
            (directory / "progress.md").write_text(text + "\n")
            if image is not None:
                (directory / "progress.png").write_bytes(image)
            if webhook:
                await post(webhook, text, image)
            print(text, flush=True)
        if not watch:
            return
        await asyncio.sleep(interval)
