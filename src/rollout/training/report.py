"""Report a training run's progress: a chart of the climb through the curriculum, and a summary in words.

Reads what the training loop writes (`metrics.jsonl`, `curriculum.json`) and can post both to a Discord webhook, once
or after every group (`rollout report RUN CATALOG --watch`). The webhook's address comes from `--webhook` or the
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

from rollout.rollouts import Row
from rollout.training.curriculum import Curriculum
from rollout.training.loop import Directory, iterations

MAX_MESSAGE = 1900
"""Discord accepts 2,000 characters."""


def hours(lines: Sequence[Mapping[str, Any]]) -> list[float]:
    """When each group ended, in hours since the run began (groups overlap, so their durations do not add up)."""
    if not lines:
        return []
    began = float(lines[0]["time"]) - float(lines[0].get("seconds", 0.0))
    return [(float(line["time"]) - began) / 3600 for line in lines]


def update_of(line: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The trainer's statistics, if this group was trained on (a skipped or failed update is a sentence)."""
    update = line.get("update")
    return update if isinstance(update, Mapping) else None  # pyright: ignore[reportUnknownVariableType]


def summary(name: str, lines: Sequence[Mapping[str, Any]], curriculum: Curriculum) -> str:
    """The run in words: the latest group, what the update did, and each unlocked row's record."""
    if not lines:
        return f"**{name}** — no group has finished yet."
    last = lines[-1]
    titles = {row.key: row.title for row in curriculum.rows}
    updates = [update for line in lines if (update := update_of(line)) is not None]
    serving = next((str(line["adapter"]) for line in reversed(lines) if line.get("adapter")), "the base model")
    took = f"{hours(lines)[-1]:.1f} h in, {len(updates)} updates"
    text = [f"**{name}** — group {last['iteration']}, {took} (serving {serving})"]
    rewards = [float(reward) for reward in last.get("rewards", [])]
    solved = [bool(value) for value in last.get("solved", [])]
    text.append(
        f"**Latest group:** {last['task']} ({titles.get(str(last['task']), '?')}) — rewards "
        f"{' / '.join(f'{reward:g}' for reward in sorted(rewards)) or 'none'}"
        f" ({_statistics(rewards)}); solved {sum(solved)}/{len(solved)}"
        + (f"; {last['failed']} episodes failed" if last.get("failed") else "")
    )
    update = update_of(last)
    text.append(
        f"**Update:** {last.get('update', 'none')}" if update is None else "**Update:** " + _update(last, update)
    )
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


def _update(line: Mapping[str, Any], update: Mapping[str, Any]) -> str:
    trained, recorded = line.get("sequences_trained", update.get("sequences", 0)), line.get("sequences_recorded")
    of = f" of {recorded:g}" if recorded else ""
    parts = [f"{trained:g}{of} sequences, {update.get('tokens', 0):g} sampled tokens"]
    if "kl_moved" in update:
        parts.append(f"moved the policy by KL ≈ {update['kl_moved']:.4f} (floor {update.get('kl_floor', 0.0):.4f})")
        parts.append(f"{update.get('optimizer_steps', 0):g} steps")
    parts.append(f"clipped {float(update.get('clip_fraction', 0.0)):.1%}")
    parts.append(f"mean ratio {float(update.get('mean_ratio', 1.0)):.4f}")
    if "gradient_norm" in update:
        parts.append(f"gradient norm {update['gradient_norm']:.2f}")
    parts.append(f"loss {float(update.get('loss', 0.0)):.4f}")
    return ", ".join(parts)


def chart(lines: Sequence[Mapping[str, Any]], rows: Sequence[Row], *, title: str = "") -> bytes:
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

    trained = [numbers.get(str(line["task"]), 0) for line in lines]
    share = [sum(line.get("solved") or [0]) / max(len(line.get("solved") or [0]), 1) for line in lines]
    climb.step(when, [line.get("unlocked", 0) for line in lines], where="post", color="#888888", label="unlocked")
    points = climb.scatter(
        when, trained, c=share, cmap="viridis", vmin=0, vmax=1, s=46, zorder=3, edgecolors="#222222", linewidths=0.5
    )
    figure.colorbar(  # beside all three panels, so that their time axes line up
        points, ax=list(panels), label="share of the group that solved it", pad=0.01, shrink=0.4, anchor=(0.0, 1.0)
    )
    climb.set_ylabel(f"row (1 to {len(rows)}, harder upward)")
    climb.set_ylim(0, max([*trained, *[line.get("unlocked", 0) for line in lines], 5]) + 1)
    climb.legend(loc="upper left", frameon=False)
    climb.set_title(title or "Climb through the curriculum")

    for at, line in zip(when, lines, strict=True):
        rewards = [float(value) for value in line.get("rewards", [])]
        if not rewards:
            continue
        reward.vlines(at, min(rewards), max(rewards), color="#9aa5b1", linewidth=2)
        reward.scatter([at] * len(rewards), rewards, color="#9aa5b1", s=10, zorder=2)
        reward.scatter([at], [statistics.fmean(rewards)], color="#d1495b", s=30, zorder=3)
        reward.annotate(
            str(line["task"]), (at, max(rewards)), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7
        )
    reward.set_ylabel("reward per episode\n(mean in red; rows differ)")

    trainings = [(at, update) for at, line in zip(when, lines, strict=True) if (update := update_of(line))]
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
    directory: Path, rows: Sequence[Row], webhook: str | None, *, watch: bool = False, interval: float = 30.0
) -> None:
    """Write `progress.png` and `progress.md` in the run's directory, and post them if a webhook is given; with
    `watch`, again after every new group, until interrupted."""
    store = Directory(directory)
    reported = -1
    while True:
        lines = iterations(store)
        if len(lines) != reported:
            reported = len(lines)
            curriculum = Curriculum(rows)
            if saved := store.read("curriculum.json"):
                curriculum.restore(json.loads(saved))
            text = summary(directory.name, lines, curriculum)
            image = chart(lines, rows, title=f"{directory.name}: climb through the curriculum") if lines else None
            store.write("progress.md", text + "\n")
            if image is not None:
                (directory / "progress.png").write_bytes(image)
            if webhook:
                await post(webhook, text, image)
            print(text, flush=True)
        if not watch:
            return
        await asyncio.sleep(interval)
