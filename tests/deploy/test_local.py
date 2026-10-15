"""The services for one machine, read as files: Postgres and S3 in Docker Compose, whose credentials are well known,
and the K3s build's privileged BuildKit, are reached from this machine alone."""

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent


def test_the_local_services_are_published_on_this_machines_loopback_alone() -> None:
    compose: dict[str, Any] = yaml.safe_load((ROOT / "deploy" / "local" / "compose.yaml").read_text())
    published = [str(port) for service in compose["services"].values() for port in service.get("ports", [])]
    assert published and all(port.startswith("127.0.0.1:") for port in published)


def test_buildkit_listens_on_its_pods_loopback_alone() -> None:
    documents = list(yaml.safe_load_all((ROOT / "deploy" / "k3s" / "build.yaml").read_text()))
    built = {(each["kind"], each["metadata"]["name"]): each for each in documents if each}
    (container,) = built["StatefulSet", "buildkit"]["spec"]["template"]["spec"]["containers"]
    arguments: list[str] = container["args"]
    assert arguments[arguments.index("--addr") + 1].startswith(("tcp://127.0.0.1:", "unix://"))
