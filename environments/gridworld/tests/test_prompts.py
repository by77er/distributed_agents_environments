"""What agents read and call: observations from each agent's side, the system prompt, and replies read as actions."""

from pydantic import JsonValue

from gridworld.game import Action, Game
from gridworld.level import Door, DoorKind, Level, Plate, generate
from gridworld.prompts import CHAT_LINES, MESSAGE_CHARACTERS, TOOLS, observe, offset, parse, system_prompt
from rollout.contracts import Message, ToolCall
from rollout.testing import tool_call_reply

TWO_ROOMS = ("#######", "#..#..#", "#.....#", "#..#..#", "#######")
PLATES = (Plate(1, (1, 1), "a"), Plate(2, (1, 3), "a"), Plate(3, (5, 1)), Plate(4, (5, 3)))
LEVEL = Level(TWO_ROOMS, ((2, 1), (2, 3)), PLATES, (Door("a", (3, 2), DoorKind.PLATES),))


def test_the_first_observation() -> None:
    game = Game(LEVEL, ["Ada", "Ben"], 30, 0)
    assert observe(game, 0) == "\n".join(
        [
            "Turn 1 of 30. You are Ada.",
            "",
            "Map (north is up):",
            "# # # # # # #",
            "# 1 @ # . 3 #",
            "# . . + . . #",
            "# 2 B # . 4 #",
            "# # # # # # #",
            "@ you, B Ben, 1-4 plates, + a closed door, / an open door, # wall, . floor.",
            "",
            "Teammates (where they are from you):",
            "- Ben: 2S",
            "",
            "Plates, doors and levers (where they are from you):",
            "- plate 1: 1W. With plate 2, it opens door a for good. Not pressed.",
            "- plate 2: 2S 1W. With plate 1, it opens door a for good. Not pressed.",
            "- plate 3: 3E. A final plate. Not pressed.",
            "- plate 4: 2S 3E. A final plate. Not pressed.",
            "- door a: 1S 1E. Opens for good when plates 1 and 2 are pressed at once. Closed.",
            "",
            "Chat, oldest first:",
            "(Nothing has been said yet.)",
            "",
            "This is the first turn.",
        ]
    )


def test_each_agent_sees_itself_and_its_teammates_from_where_it_stands() -> None:
    game = Game(LEVEL, ["Ada", "Ben"], 30, 0)
    game.step([Action("move", direction="west", message="I take plate 1"), Action("move", direction="west")])
    ada, ben = observe(game, 0), observe(game, 1)
    assert "# @ . # . 3 #" in ada and "# B . # . 4 #" in ada
    assert "# A . # . 3 #" in ben and "# @ . # . 4 #" in ben
    assert "- Ben: 2S" in ada and "- Ada: 2N" in ben
    assert "- plate 1: here. With plate 2, it opens door a for good. Pressed by you." in ada
    assert "- plate 1: 2N. With plate 2, it opens door a for good. Pressed by Ada." in ben
    assert "- you, 1 turn ago: I take plate 1" in ada and "- Ada, 1 turn ago: I take plate 1" in ben
    assert "+ a closed door" in ada and "# . / . . #" not in ada
    last = "Last turn: You moved west. Plates 1 and 2 were pressed at once: door a is open for good."
    assert ada.endswith(last) and ben.endswith(last)
    assert "# . . / . . #" in ada, "the door shows open"
    game.step([Action("wait"), Action("say", message="go")])
    assert "- Ada, 2 turns ago: I take plate 1" in observe(game, 1)
    assert "- you, 1 turn ago: go" in observe(game, 1)


def test_chat_older_than_the_last_lines_is_out_of_sight() -> None:
    game = Game(LEVEL, ["Ada", "Ben"], 30, 0)
    for number in range(CHAT_LINES + 2):
        game.step([Action("say", message=f"line {number}"), Action("wait")])
    seen = observe(game, 1)
    assert "line 0" not in seen and "line 1\n" not in seen
    assert f"- Ada, 1 turn ago: line {CHAT_LINES + 1}" in seen
    assert f"- Ada, {CHAT_LINES} turns ago: line 2" in seen


def test_the_system_prompt_says_the_team_and_only_the_mechanics_the_level_has() -> None:
    plain = system_prompt(generate("open", 2, 0), ["Ada", "Ben"], 20)
    assert "in a team of two: Ada, Ben." in plain and "You have 20 turns" in plain
    assert "door" not in plain and "gate" not in plain and "lever" not in plain
    vault = system_prompt(generate("vault", 4, 0), ["Ada", "Ben", "Cy", "Dee"], 80)
    assert "plates are pressed at once" in vault and "holds it while others pass" in vault
    assert "A lever opens its door" in vault


def test_offsets_are_from_the_agent() -> None:
    assert offset((2, 2), (2, 2)) == "here"
    assert offset((2, 2), (5, 0)) == "2N 3E"
    assert offset((2, 2), (1, 4)) == "2S 1W"


def test_the_tools() -> None:
    assert [tool.name for tool in TOOLS] == ["move", "wait", "say"]


def call(name: str, **arguments: JsonValue) -> ToolCall:
    return ToolCall(call_id=name, name=name, arguments=arguments)


def test_a_reply_is_read_as_its_first_call() -> None:
    assert parse(Message.assistant("I would go north.")) == Action("none")
    assert parse(tool_call_reply(call("move", direction="North"))) == Action("move", direction="north")
    assert parse(tool_call_reply(call("move", direction="east", say="  on   my way "))) == Action(
        "move", direction="east", message="on my way"
    )
    assert parse(tool_call_reply(call("wait"), call("move", direction="west"))) == Action("wait", extra=True)
    assert parse(tool_call_reply(call("say", message="x" * 500))) == Action("say", message="x" * MESSAGE_CHARACTERS)


def test_a_call_that_is_no_action_says_why() -> None:
    assert parse(tool_call_reply(call("move", direction="up"))).problem == (
        "move needs a direction: north, south, east or west"
    )
    assert parse(tool_call_reply(call("say", message="   "))).problem == "say needs a message"
    jump = parse(tool_call_reply(call("jump", say="hi")))
    assert jump == Action("invalid", message="hi", problem="there is no tool jump")
