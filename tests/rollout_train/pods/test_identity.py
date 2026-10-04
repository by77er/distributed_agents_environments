"""Pods' names and identities, and which pods may be reached: those whose newest beat is fresh and names the identity
named for the pod."""

import pytest
from pydantic import JsonValue

from rollout_train.pods import GATEWAY_IDENTITY, PodAddress, live, pod_identity
from rollout_train.presence import STALE, Beat


def beat(runner: str, *, age: float = 1.0, **pod: JsonValue) -> Beat:
    said: dict[str, JsonValue] = {"name": runner, "identity": pod_identity(runner), "address": "https://1.2.3.4:40001",
                                  "role": "inference", "ready": True, "serial": "1234"}  # fmt: skip
    return Beat(runner, 0.0, {"pod": {**said, **pod}}, age=age)


def test_a_pod_is_named_as_a_dns_label_is() -> None:
    assert pod_identity("inference-r1-0") == "spiffe://rollout/pod/inference-r1-0"
    assert GATEWAY_IDENTITY == "spiffe://rollout/gateway"
    for wrong in ("", "Upper", "-starts", "ends-", "a/b", "x" * 64, "dots.in.it"):
        with pytest.raises(ValueError):
            pod_identity(wrong)


def test_only_pods_alive_with_their_own_identity_are_reached() -> None:
    beats = [
        beat("inference-1"),
        beat("inference-2", ready=False, serial=None),
        beat("trainer-1", role="trainer"),
        beat("stale", age=STALE + 1),
        beat("impostor", identity=pod_identity("inference-1")),  # (says another pod's identity)
        beat("no-address", address=None),
        beat("renamed", name="inference-9"),  # (its beat is another runner's)
        Beat("runner", 0.0, {"kind": "runner"}),  # (not a pod)
    ]
    assert live(beats) == [
        PodAddress("inference-1", pod_identity("inference-1"), "https://1.2.3.4:40001", "inference", True, "1234"),
        PodAddress("inference-2", pod_identity("inference-2"), "https://1.2.3.4:40001", "inference", False),
        PodAddress("trainer-1", pod_identity("trainer-1"), "https://1.2.3.4:40001", "trainer", True, "1234"),
    ]
    assert [pod.name for pod in live(beats, "trainer")] == ["trainer-1"]
