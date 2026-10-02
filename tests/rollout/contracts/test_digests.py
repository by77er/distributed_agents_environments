import hashlib

from rollout.contracts import (
    EMPTY_DIGEST,
    Message,
    RetryClass,
    ToolSpecification,
    arguments_digest,
    canonical_json,
    context_digests,
    message_digest,
    spec_hash,
)


def test_canonical_json_is_rfc8785() -> None:
    assert canonical_json({"b": 1.0, "a": [None, "é"]}) == '{"a":[null,"é"],"b":1}'.encode()


def test_arguments_digest_ignores_key_order_but_not_nulls() -> None:
    assert arguments_digest({"a": 1, "b": 2}) == arguments_digest({"b": 2, "a": 1})
    assert arguments_digest({"a": 1, "b": None}) != arguments_digest({"a": 1})
    assert arguments_digest({"a": 1}) == hashlib.sha256(b'{"a":1}').hexdigest()


def test_spec_hash_covers_only_model_visible_fields() -> None:
    plain = ToolSpecification(name="search", description="Search the web.")
    extended = ToolSpecification(name="search", description="Search the web.", retry_class=RetryClass.IDEMPOTENT)
    assert spec_hash(plain) == spec_hash(extended)
    assert spec_hash(plain) != spec_hash(ToolSpecification(name="search", description="Search."))


def test_message_digest_excludes_meta() -> None:
    message = Message.user("hi")
    assert message_digest(message) == message_digest(message.model_copy(update={"meta": {"trace": "x"}}))
    assert message_digest(message) != message_digest(Message.user("hello"))


def test_context_digest_chain() -> None:
    first, second, third = Message.system("s"), Message.user("u"), Message.assistant("a")
    chain = context_digests([first, second, third])
    assert len(chain) == 4
    assert chain[0] == EMPTY_DIGEST == hashlib.sha256(b"").hexdigest()
    link = bytes.fromhex(EMPTY_DIGEST) + bytes.fromhex(message_digest(first))
    assert chain[1] == hashlib.sha256(link).hexdigest()
    # A retained prefix is recognizable by its own chain value.
    assert context_digests([first, second])[2] == chain[2]
    assert context_digests([first, Message.user("other")])[2] != chain[2]
