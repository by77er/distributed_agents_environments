"""The workspace's layers hold, read from the source with `ast` (nothing is imported, so no Ray, torch or GPU):

- a package imports exactly the workspace packages its project file declares (dependencies and extras);
- `rollout` imports no other workspace package;
- `rollout-train` imports `rollout` and no implementation or environment;
- an implementation imports the libraries and no environment;
- an environment imports `rollout` and nothing of `rollout-train` or an implementation.

An import counts wherever it is in a package's modules: at the top, inside a function, under `TYPE_CHECKING`. An edge
between two implementations or two environments is allowed only where `DEPENDENCIES` says why; `EXCEPTIONS` lists the
edges that break a rule today, each with the reason it stays until it is designed away."""

import ast
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LIBRARIES = frozenset({"rollout", "rollout-train"})

DEPENDENCIES = {
    ("rollout-lora", "rollout-objectives"): "the LoRA and full-weight trainers take the objective's step on the GPU",
    ("rollout-tinker", "rollout-objectives"): "the Tinker trainer takes the objective's plan, and its loss as its own",
    ("minecraft-horizons", "minecraft-team"): "its objectives are played in minecraft-team's worlds, with its agents",
}
"""Edges between two implementations or two environments, each declared in its project file, and why."""

EXCEPTIONS: dict[tuple[str, str], str] = {}
"""Edges that break a rule today, and why they stay until they are designed away."""


@dataclass(frozen=True)
class Project:
    name: str
    layer: str  # libraries, implementations or environments
    package: Path
    declared: frozenset[str]


def project_of(directory: Path) -> Project:
    table = tomllib.loads((directory / "pyproject.toml").read_text())
    requirements = [*table["project"].get("dependencies", [])]
    for extra in table["project"].get("optional-dependencies", {}).values():
        requirements += extra
    names = {match.group() for requirement in requirements if (match := re.match(r"[A-Za-z0-9._-]+", requirement))}
    (package,) = table["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    return Project(table["project"]["name"], directory.parent.name, directory / package, frozenset(names))


PROJECTS = sorted(
    (project_of(path.parent) for layer in ("libraries", "implementations", "environments")
     for path in ROOT.glob(f"{layer}/*/pyproject.toml")),
    key=lambda project: project.name,
)  # fmt: skip
NAMES = {project.package.name: project.name for project in PROJECTS}
"""Import name → distribution name, for every package of the workspace."""
WORKSPACE = frozenset(NAMES.values())
BY_NAME = {project.name: project for project in PROJECTS}


def imports(project: Project) -> dict[str, list[str]]:
    """The other workspace distributions a package's modules import, each with where (`path:line`)."""
    found: dict[str, list[str]] = {}
    for path in sorted(project.package.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                name = NAMES.get(module.split(".")[0])
                if name is not None and name != project.name:
                    found.setdefault(name, []).append(f"{path.relative_to(ROOT)}:{node.lineno}")
    return found


def allowed(project: Project) -> set[str]:
    """The workspace distributions a package may import, by its layer and the edges listed above."""
    if project.name == "rollout":
        by_layer: set[str] = set()
    elif project.layer == "implementations":
        by_layer = set(LIBRARIES)
    else:
        by_layer = {"rollout"}
    return by_layer | {to for start, to in DEPENDENCIES | EXCEPTIONS if start == project.name}


@pytest.mark.parametrize("project", PROJECTS, ids=lambda project: project.name)
def test_a_package_imports_exactly_the_workspace_packages_it_declares(project: Project) -> None:
    assert set(imports(project)) == project.declared & WORKSPACE


@pytest.mark.parametrize("project", PROJECTS, ids=lambda project: project.name)
def test_a_package_imports_only_what_its_layer_allows(project: Project) -> None:
    found = imports(project)
    assert {name: found[name] for name in set(found) - allowed(project)} == {}


def test_every_listed_edge_is_between_packages_that_have_it() -> None:
    for start, to in DEPENDENCIES | EXCEPTIONS:
        assert to in imports(BY_NAME[start]), f"{start} no longer imports {to}: take it off the list"
    for start, to in DEPENDENCIES:
        assert BY_NAME[start].layer == BY_NAME[to].layer != "libraries", (start, to)
