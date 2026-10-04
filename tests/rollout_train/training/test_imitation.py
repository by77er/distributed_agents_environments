"""Imitation: the guidance an episode's prompts carried is cut out of its segments, and what was sampled is kept."""

from pathlib import Path

from pydantic import JsonValue

from rollout.harness.blobs import FileBlobStore
from rollout_train import Checkpoints, FileLedger
from rollout_train.imitation import GUIDANCE, examples, imitate, without
from rollout_train.record import scope, table
from rollout_train.recorder import Segment, Span
from rollout_train.rollouts import Episode, Outcome, Record, Trajectory, stored
from rollout_train.rollouts.scheduler import EPISODES
from rollout_train.testing import plain_renderer
from tests.rollout_train.training.test_loop import Counting

WAY = "How to get there. Place the table."


def segment(text: str, sampled: str) -> Segment:
    tokens = [ord(character) for character in text + sampled]
    return Segment(tokens, [Span(len(text), len(tokens), 3, "r:0:1")], [-0.5] * len(sampled))


def test_a_segment_without_its_guidance_keeps_what_was_sampled_where_it_was() -> None:
    renderer = plain_renderer("plain")
    guided = segment(f"system: Goal: diamonds.\n\n{WAY}\n\nHow the game runs.\nuser: You see ore.\nassistant: ", "mine")
    cut = without(guided, [WAY], renderer)
    assert cut is not None
    text = renderer.decode(cut.tokens)
    assert text == "system: Goal: diamonds.\n\nHow the game runs.\nuser: You see ore.\nassistant: mine"
    (span,) = cut.spans
    assert renderer.decode(cut.tokens[span.start : span.end]) == "mine" and span.version == 3
    assert cut.logprobs == guided.logprobs
    assert without(guided, ["Not in it."], renderer) is None  # (guidance it did not carry: not an example of it)


async def test_the_solved_guided_episodes_are_examples_and_a_step_on_them_makes_a_version(tmp_path: Path) -> None:
    blobs, ledger = FileBlobStore(tmp_path / "blobs"), FileLedger(tmp_path / "ledger")
    fence = await ledger.take("runners/here")
    for number, (solved, told) in enumerate([(True, {"way": WAY}), (False, {"way": WAY}), (True, {})], start=1):
        trajectory = Trajectory([segment(f"system: Goal.\n\n{WAY}\nassistant: ", "craft")], {"default": 1.0})
        info: dict[str, JsonValue] = {"solved": solved, GUIDANCE: dict(told)}
        episode = Episode("train", 1, number, f"r{number}", {}, Outcome.COMPLETED, info=info,
                          trajectories={"ada": trajectory})  # fmt: skip
        record = (await stored(episode, [], blobs)).to_json()
        await ledger.append(table("train", EPISODES), f"1/{number}", record, fence)

    taught = await examples(ledger, "train", blobs, plain_renderer("plain"), kinds=["way"])
    assert taught.episodes == 1 and taught.left_out == 0  # solved, and told the way: the first only
    (example,) = taught.segments
    assert example.advantage == 1.0 and example.source == "train/1/1/ada/0"
    assert "".join(chr(token) for token in example.segment.tokens) == "system: Goal.\nassistant: craft"
    first = (await ledger.read(table("train", EPISODES)))["1/1"]
    assert Record.from_json(first).episode.info[GUIDANCE] == {"way": WAY}  # type: ignore[arg-type]

    checkpoints, trainer = Checkpoints(ledger, blobs), Counting()
    run = await ledger.take(scope("train"))
    checkpoint = await imitate(
        checkpoints,
        trainer,
        taught,
        fence=run,
        run="train",
        start=None,
        base="qwen",
        directory=tmp_path / "checkpoints",
    )
    assert (checkpoint.run, checkpoint.parents, checkpoint.depth, checkpoint.base) == ("train", (), 1, "qwen")
    assert checkpoint.metrics["imitated_episodes"] == 1.0 and trainer.batches == [[example]]
    again = await imitate(
        checkpoints, trainer, taught, fence=run, run="train", start=None, directory=tmp_path / "checkpoints"
    )
    assert again.parents == (checkpoint.id,) and again.depth == 2 and await checkpoints.head("train") == again
