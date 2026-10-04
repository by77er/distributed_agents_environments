"""What a turn costs to store, on the shape of a Minecraft episode: a system prompt and the action tools, a summary,
the recent turns in their remembered form, and the current observation in full with its map; about 4,900 prompt tokens
and 220 sampled per turn, in Qwen3.5's own tokens."""

import json
import lzma
import random
from array import array
from pathlib import Path

import pytest

from rollout.contracts import (
    FinishReason,
    Message,
    Reasoning,
    ReasoningScope,
    Role,
    SampleResult,
    Text,
    ToolCall,
    ToolResult,
    ToolResultBlock,
    Usage,
)
from rollout_train.gateway import TurnRecord, TurnStore, turns_table
from rollout_train.recorder.segments import segments_of
from tests.rollout_qwen.support import qwen_tokenizer
from tests.rollout_train.gateway.support import stores

prompts = pytest.importorskip("minecraft_team.prompts")
SAID = (
    "the ore is east of the corridor so I should walk there first then mine it with the stone pickaxe and pick up "
    "what drops before the creeper gets close ben said he is going north to the chest which leaves the furnace to me"
)
WORDS = SAID.split()
SYMBOLS = list("..........####,,,,wwvv?????~*") + ["#"] * 6


def system() -> str:
    return prompts.SYSTEM.format(
        opening=prompts.TEAM_OPENING.format(count="three", team="ada, ben, cy"),
        goal=prompts.PROGRESS_GOAL.format(early=""),
        way="\n\n" + " ".join(prompts.MAKING.values()),
        turns=prompts.TEAM_TURNS.format(count="three"),
        window="five",
        chat=prompts.CHAT.format(chat_lines=6),
        teamwork="",
        death="and what you carried lies where you died",
    )


def observation(chance: random.Random, turn: int, *, recalled: bool) -> str:
    x, z = chance.randrange(-200, 200), chance.randrange(-200, 200)
    lines = [
        f"You are ada, at ({x}, 64, {z}) (overworld, plains, day); health {chance.randrange(10, 21)}/20, food 18/20.",
        f"Inventory: {chance.randrange(1, 9)} oak_log, {chance.randrange(0, 20)} cobblestone, 1 stone_pickaxe.",
    ]
    if not recalled:
        lines.append("Map of what you have seen within 6 blocks: columns are x=-6 to x=6; each row starts with its z")
        for height in (66, 65, 64, 63, 62):
            lines.append(f"y={height}:")
            row = [chance.choice(SYMBOLS)]
            for _ in range(12):  # (runs of one block, as walls, floors and the unseen are)
                row.append(row[-1] if chance.random() < 0.75 else chance.choice(SYMBOLS))
            for depth in range(-6, 7):
                row = [each if chance.random() < 0.85 else chance.choice(SYMBOLS) for each in row]
                lines.append(f"{depth} " + " ".join(row))
        lines.append("Next to you, at head height and at foot height:")
        sides = ("north", "south", "east", "west")
        lines += [f"- {side}: ({x}, 65, {z}) empty, ({x}, 64, {z}) stone" for side in sides]
        lines.append("Team chat, oldest first:")
        lines += [f"- ben, {age} turns ago: {' '.join(chance.choices(WORDS, k=12))}" for age in range(6, 0, -1)]
    lines.append(f"Notable in sight: iron_ore at ({x - 3}, 63, {z + 1}), {chance.randrange(2, 9)}.1 away.")
    lines.append(f"Teammates in sight: ben at ({x + 4}, 64, {z - 2}); cy at ({x - 7}, 65, {z + 5}).")
    return "\n".join(lines)


def reply(chance: random.Random, number: int) -> Message:
    thought = " ".join(chance.choices(WORDS, k=165))
    call = ToolCall(call_id=f"call_{number}", name="mine", arguments={"x": chance.randrange(-9, 9), "y": 63, "z": 4})
    return Message(role=Role.ASSISTANT, content=[Reasoning(scope=ReasoningScope.PORTABLE, text=thought), call])


async def test_a_minecraft_turn_costs_a_few_kilobytes(tmp_path: Path) -> None:
    from rollout_qwen import qwen35

    renderer = qwen35(qwen_tokenizer())
    chance = random.Random(7)
    ledger, blobs = stores(tmp_path)
    store = TurnStore(ledger, blobs)
    fence = await ledger.take("runs/train/episodes/1/1")
    head = Message.system(system())
    summary: Message | None = None
    remembered: list[list[Message]] = []
    turns: list[TurnRecord] = []
    for number in range(60):
        current = Message.user(observation(chance, number, recalled=False))
        context = [head, *([summary] if summary else []), *(m for turn in remembered for m in turn), current]
        prompt = renderer.render(context, prompts.ACTIONS)
        if len(prompt) > 5_600:  # memory is full: the oldest turns go, into a summary
            summary = Message.user("What you remember: " + " ".join(chance.choices(WORDS, k=150)))
            remembered = remembered[-1:]
            context = [head, summary, *(m for turn in remembered for m in turn), current]
            prompt = renderer.render(context, prompts.ACTIONS)
        said = reply(chance, number)
        completion = renderer.encode(
            f"{said.content[0].text}\n</think>\n\n<tool_call>\n<function=mine>\n<parameter=x>\n3\n</parameter>\n"  # type: ignore[union-attr]
            "<parameter=y>\n63\n</parameter>\n<parameter=z>\n4\n</parameter>\n</function>\n</tool_call><|im_end|>"
        )
        logprobs = [float(array("f", [-chance.expovariate(3.0)])[0]) for _ in completion]
        result = SampleResult(
            message=said,
            finish_reason=FinishReason.TOOL_USE,
            usage=Usage(context_used=len(prompt) + len(completion), context_limit=8192, input_tokens=len(prompt)),
        )
        turn = TurnRecord(
            f"r_1:0:{number}", "train", "r_1", "policy", "policy", "kkkmnop", 12, array("i", prompt), completion,
            [True] * len(completion), logprobs, result, episode="1/1", attempt=1,
            timings={"started": 1.7e9 + number, "phases": [3.1, 0.4], "seconds": 3.6},
        )  # fmt: skip
        await store.record(turn, fence)
        turns.append(turn)
        recalled = Message.user(observation(chance, number, recalled=True))
        result_block = ToolResultBlock(call_id=f"call_{number}", result=ToolResult(content=[Text(text="Mined.")]))
        remembered.append([recalled, said, Message(role=Role.TOOL, content=[result_block])])

    index = await ledger.read(turns_table("train", "r_1"))
    blob_bytes = sum(int(entry["blob"]["size"]) for entry in index.values())  # type: ignore[index]
    ledger_bytes = sum(len(json.dumps(entry)) for entry in index.values())
    prompt_tokens = sum(len(turn.prompt) for turn in turns) / len(turns)
    sampled = sum(len(turn.completion) for turn in turns) / len(turns)
    whole = sum(
        len(lzma.compress(array("i", [*turn.prompt, *turn.completion]).tobytes() + array("f", turn.logprobs).tobytes()))
        for turn in turns
    )
    raw = sum(len(turn.prompt) * 4 + len(turn.completion) * 12 for turn in turns)
    per_turn = (blob_bytes + ledger_bytes) / len(turns)
    print(
        f"\n{len(turns)} turns, {prompt_tokens:.0f} prompt tokens and {sampled:.0f} sampled per turn: "
        f"{blob_bytes / len(turns):.0f} B of blob and {ledger_bytes / len(turns):.0f} B of ledger record per turn "
        f"({per_turn:.0f} B in all); each turn's tokens compressed whole would be {whole / len(turns):.0f} B, "
        f"and raw {raw / len(turns):.0f} B"
    )
    assert 4_000 < prompt_tokens < 5_200 and 180 < sampled < 300
    assert per_turn < 4_000
    back = await store.turns("train", "r_1")
    assert [turn.prompt for turn in back] == [turn.prompt for turn in turns]
    assert segments_of(back) == segments_of(turns)
