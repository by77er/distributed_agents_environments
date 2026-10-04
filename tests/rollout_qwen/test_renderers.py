"""How Qwen's renderers end and read a turn the model ended its own way: at the end of text, or inside its thinking."""

from collections.abc import Callable

import pytest

from rollout.contracts import Reasoning
from rollout_qwen import qwen3, qwen35
from rollout_train.recorder.renderers import Renderer, Tokenizer
from tests.rollout_qwen.support import qwen_tokenizer


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return qwen_tokenizer()


@pytest.mark.parametrize("family", [qwen3, qwen35])
def test_the_end_of_text_ends_a_turn_as_the_end_of_a_message_does(
    tokenizer: Tokenizer, family: Callable[[Tokenizer], Renderer]
) -> None:
    renderer = family(tokenizer)
    stops = renderer.stop_token_ids()
    assert stops[:1] == renderer.encode("<|im_end|>") and renderer.encode("<|endoftext|>")[0] in stops
    message = renderer.parse(renderer.encode("<think>\nfine\n</think>\n\nDone.<|endoftext|>"), [])
    assert message.text == "Done."


def test_thinking_the_model_opened_and_never_closed_is_all_thought(tokenizer: Tokenizer) -> None:
    renderer = qwen3(tokenizer)
    message = renderer.parse(renderer.encode("<think>\nI should run it, but then<|im_end|>"), [])
    (thought,) = message.content
    assert isinstance(thought, Reasoning) and thought.text == "I should run it, but then" and message.text == ""
