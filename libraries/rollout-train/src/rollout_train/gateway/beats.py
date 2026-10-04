"""A gateway replica's heartbeat (`rollout_train.presence`): its host, where it listens, its machine, and each channel
it samples, with what it serves and how fast since the beat before, so the monitor shows the replicas alive."""

import socket
from pathlib import Path

from pydantic import JsonValue

from rollout_train.gateway.service import Gateway
from rollout_train.machine import measured

GATEWAY = "gateway"
"""What a gateway's heartbeat says it is (`about["kind"]`)."""


def name_of(listen: str) -> str:
    """A replica's name among the beats: `gateway/HOST/LISTEN`."""
    return f"{GATEWAY}/{socket.gethostname()}/{listen}"


def about(gateway: Gateway, listen: str, directory: Path | None = None) -> dict[str, JsonValue]:
    """What a replica says of itself in each beat (called in a thread: it measures): its host, where it listens, the
    machine (with the disk `directory` is on), and each channel: those of this process with what each serves, and the
    routed ones (`RUN/NAME`) with their servers; each with what passed through it since the beat before."""
    channels: list[JsonValue] = [
        {"channel": name, "adapter": channel.serving, "version": channel.version, **channel.take()}
        for name, channel in gateway.channels.items()
    ]
    routed = gateway.routes.channels() if gateway.routes is not None else {}
    for name, channel in routed.items():
        servers: list[JsonValue] = list(channel.servers())
        channels.append({"channel": name, **channel.take(), "servers": servers})
    return {
        "kind": GATEWAY,
        "host": socket.gethostname(),
        "listen": listen,
        "machine": measured(directory),
        "channels": channels,
    }
