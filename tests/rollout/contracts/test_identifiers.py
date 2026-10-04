import re

import pytest

from rollout.contracts import (
    EffectIdentity,
    SessionIdentity,
    effect_id,
    new_run_id,
    new_ulid,
    session_id,
)


def test_ulid_format_and_time_order() -> None:
    first = new_ulid()
    assert re.fullmatch(r"[0-7][0-9A-HJKMNP-TV-Z]{25}", first)
    assert new_ulid()[:10] >= first[:10]


def test_effect_id_round_trip() -> None:
    run_id = new_run_id()
    identifier = effect_id(run_id, 2, 17)
    assert identifier == f"{run_id}:2:17"
    assert EffectIdentity.parse(identifier) == EffectIdentity(run_id, 2, 17)


def test_effect_identity_rejects_invalid_parts() -> None:
    with pytest.raises(ValueError):
        EffectIdentity("r_a:b", 0, 0)
    with pytest.raises(ValueError):
        EffectIdentity("r_a", -1, 0)


def test_session_id_round_trip() -> None:
    assert session_id("r_x", "policy") == "r_x/policy"
    assert SessionIdentity.parse("u_y/user") == SessionIdentity("u_y", "user")
