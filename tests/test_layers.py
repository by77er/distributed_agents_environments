"""The workspace's layers hold: each package imports only the workspace packages its project file declares, the
interface libraries depend on no implementation, and an environment needs the harness and nothing above it."""

import ast
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = sorted(
    path.parent
    for pattern in ("libraries/*", "implementations/*", "environments/*")
    for path in ROOT.glob(f"{pattern}/pyproject.toml")
)


def described(project: Path) -> tuple[str, Path, set[str]]:
    """A project's distribution name, its package's directory, and the workspace distributions it declares."""
    table = tomllib.loads((project / "pyproject.toml").read_text())
    declared = [*table["project"]["dependencies"]]
    for extra in table["project"].get("optional-dependencies", {}).values():
        declared += extra
    names = {requirement.split("[")[0].split(">")[0].split(";")[0].strip() for requirement in declared}
    (package,) = table["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    return table["project"]["name"], project / package, names


PACKAGES = {directory.name: name for name, directory, _ in map(described, PROJECTS)}
"""Import name → distribution name, for every package of the workspace."""


def imported(package: Path) -> set[str]:
    """The workspace distributions a package's modules import, anywhere in them."""
    found: set[str] = set()
    for path in package.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            modules += [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
            found |= {PACKAGES[module.split(".")[0]] for module in modules if module.split(".")[0] in PACKAGES}
    return found


@pytest.mark.parametrize("project", PROJECTS, ids=lambda project: project.name)
def test_a_package_imports_only_the_workspace_packages_it_declares(project: Path) -> None:
    name, package, declared = described(project)
    assert imported(package) - {name} <= declared


def test_the_interface_libraries_require_no_implementation() -> None:
    implementations = {described(project)[0] for project in PROJECTS if project.parent.name == "implementations"}
    for library in (project for project in PROJECTS if project.parent.name == "libraries"):
        table = tomllib.loads((library / "pyproject.toml").read_text())
        required = {requirement.split("[")[0].strip() for requirement in table["project"]["dependencies"]}
        assert not required & implementations, library.name
    assert described(ROOT / "libraries" / "rollout")[2] & set(PACKAGES.values()) == set()  # the harness stands alone


def test_an_environment_needs_the_harness_and_nothing_above_it() -> None:
    for environment in (project for project in PROJECTS if project.parent.name == "environments"):
        name, package, _ = described(environment)
        assert imported(package) - {name} == {"rollout"}, name
