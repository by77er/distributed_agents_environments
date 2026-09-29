"""Generate docs/guide/reference.md from the public API.

The reference lists every name in the `__all__` of the public modules. It is built from the source with `ast`, so
types and defaults appear as written, and attribute docstrings (fields, enum members) are included.

    uv run python scripts/generate_reference.py          # rewrite the reference
    uv run python scripts/generate_reference.py --check  # exit 1 if it is out of date
"""

import argparse
import ast
import importlib
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "src"
OUTPUT = ROOT / "docs" / "guide" / "reference.md"

PUBLIC_MODULES = [
    ("rollout.core.harness", "Writing tasks and agents."),
    ("rollout.core.contracts", "Types that cross layers: canonical content, identifiers, digests, effects, events."),
    ("rollout.core.local", "In-process implementations for the local profile."),
    ("rollout.core.testing", "Test doubles: a scripted model endpoint and helpers."),
    ("rollout.durable", "The durability layer: runs that survive crashes and restarts, on DBOS."),
    ("rollout.adapters.responses", "A model endpoint for the OpenAI Responses API, on an API key or a Codex login."),
]


@dataclass
class Member:
    name: str
    annotation: str = ""
    default: str = ""
    docstring: str = ""


@dataclass
class Definition:
    name: str
    kind: str  # class | function | constant | type alias
    path: Path
    node: ast.AST
    docstring: str = ""
    fields: list[Member] = field(default_factory=list[Member])
    methods: list[Member] = field(default_factory=list[Member])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if the reference is out of date")
    arguments = parser.parse_args()
    text = render()
    if arguments.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != text:
            print(f"{OUTPUT.relative_to(ROOT)} is out of date; run scripts/generate_reference.py", file=sys.stderr)
            return 1
        return 0
    OUTPUT.write_text(text)
    return 0


def render() -> str:
    lines = [
        "# API reference",
        "",
        "Generated from the source by `scripts/generate_reference.py`; do not edit by hand. Every public name,",
        "grouped by module, alphabetically. Types and defaults appear as written in the source. The",
        "[guide](README.md) explains how the pieces fit together.",
        "",
    ]
    sections: list[tuple[str, str, list[Definition]]] = []
    for module_name, summary in PUBLIC_MODULES:
        module = importlib.import_module(module_name)
        definitions = [find_definition(module_name, name) for name in sorted(module.__all__, key=str.lower)]
        sections.append((module_name, summary, definitions))
    lines += ["## Contents", ""]
    for module_name, summary, definitions in sections:
        names = ", ".join(f"[`{definition.name}`](#{anchor(definition.name)})" for definition in definitions)
        lines += [f"- **[`{module_name}`](#{anchor(module_name)})** — {summary} {names}"]
    lines.append("")
    for module_name, summary, definitions in sections:
        lines += [f"## `{module_name}`", "", summary, ""]
        for definition in definitions:
            lines += render_definition(definition)
    return "\n".join(lines).rstrip() + "\n"


def anchor(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum() or character in "-_")


def find_definition(module_name: str, name: str) -> Definition:
    """Find the top-level definition of `name` in the package that `module_name` belongs to."""
    module_file = Path(importlib.import_module(module_name).__file__ or "")
    is_package = module_file.name == "__init__.py"
    candidates = sorted(module_file.parent.rglob("*.py")) if is_package else [module_file]
    for path in candidates:
        body = ast.parse(path.read_text()).body
        # The last definition wins: for an overloaded function it is the implementation.
        matches = [index for index, node in enumerate(body) if defines(node, name)]
        if matches:
            index = matches[-1]
            following = body[index + 1] if index + 1 < len(body) else None
            return describe(name, body[index], path, following)
    raise LookupError(f"{name} from {module_name} not found in {module_file.parent}")


def defines(node: ast.stmt, name: str) -> bool:
    if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef | ast.TypeAlias):
        target = node.name.id if isinstance(node, ast.TypeAlias) else node.name
        return target == name
    if isinstance(node, ast.Assign):
        return any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    if isinstance(node, ast.AnnAssign):
        return isinstance(node.target, ast.Name) and node.target.id == name
    return False


def describe(name: str, node: ast.stmt, path: Path, following: ast.stmt | None) -> Definition:
    relative = path.relative_to(SOURCE)
    if isinstance(node, ast.ClassDef):
        definition = Definition(name, "class", relative, node, ast.get_docstring(node) or "")
        definition.fields, definition.methods = class_members(node)
        return definition
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return Definition(name, "function", relative, node, ast.get_docstring(node) or "")
    kind = "type alias" if isinstance(node, ast.TypeAlias) else "constant"
    return Definition(name, kind, relative, node, string_constant(following))


def string_constant(node: ast.stmt | None) -> str:
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
        return inspect.cleandoc(node.value.value)
    return ""


def class_members(node: ast.ClassDef) -> tuple[list[Member], list[Member]]:
    fields: list[Member] = []
    methods: list[Member] = []
    body = node.body
    for index, statement in enumerate(body):
        following = body[index + 1] if index + 1 < len(body) else None
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            if statement.target.id.startswith("_"):
                continue
            default = ast.unparse(statement.value) if statement.value is not None else ""
            fields.append(
                Member(statement.target.id, ast.unparse(statement.annotation), default, string_constant(following))
            )
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_") and target.id != "model_config":
                    fields.append(Member(target.id, "", ast.unparse(statement.value), string_constant(following)))
        elif isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
            if statement.name.startswith("_") and statement.name != "__init__":
                continue
            methods.append(Member(statement.name, signature(statement), "", ast.get_docstring(statement) or ""))
    return fields, methods


def signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    decorators = {ast.unparse(decorator) for decorator in node.decorator_list}
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    type_parameters = f"[{', '.join(ast.unparse(parameter) for parameter in node.type_params)}]"
    arguments = format_arguments(node.args)
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    text = f"{prefix} {node.name}{type_parameters if node.type_params else ''}({arguments}){returns}"
    if "property" in decorators:
        text = f"@property {text}"
    elif "classmethod" in decorators:
        text = f"@classmethod {text}"
    return text


def format_arguments(arguments: ast.arguments) -> str:
    """Like `ast.unparse`, with PEP 8 spacing around the defaults of annotated parameters."""

    def parameter(argument: ast.arg, default: ast.expr | None) -> str:
        text = argument.arg
        if argument.annotation is not None:
            text += f": {ast.unparse(argument.annotation)}"
        if default is not None:
            text += f" = {ast.unparse(default)}" if argument.annotation is not None else f"={ast.unparse(default)}"
        return text

    positional = arguments.posonlyargs + arguments.args
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(arguments.defaults))
    defaults += arguments.defaults
    parts = [parameter(argument, default) for argument, default in zip(positional, defaults, strict=True)]
    if arguments.posonlyargs:
        parts.insert(len(arguments.posonlyargs), "/")
    if arguments.vararg is not None:
        parts.append(f"*{parameter(arguments.vararg, None)}")
    elif arguments.kwonlyargs:
        parts.append("*")
    for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True):
        parts.append(parameter(argument, default))
    if arguments.kwarg is not None:
        parts.append(f"**{parameter(arguments.kwarg, None)}")
    return ", ".join(parts)


def render_definition(definition: Definition) -> list[str]:
    node = definition.node
    lines = [f"### `{definition.name}`", "", f"*{definition.kind}* · `src/{definition.path}`", ""]
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(base) for base in node.bases)
        lines += ["```python", f"class {node.name}({bases})" if bases else f"class {node.name}", "```", ""]
    elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        lines += ["```python", signature(node), "```", ""]
    elif isinstance(node, ast.TypeAlias | ast.Assign | ast.AnnAssign):
        lines += ["```python", ast.unparse(node), "```", ""]
    if definition.docstring:
        lines += [definition.docstring, ""]
    if definition.fields:
        if is_enum(node):
            lines += ["| Member | Value | Description |", "|---|---|---|"]
            for member in definition.fields:
                lines.append(f"| `{member.name}` | `{cell(member.default)}` | {cell(member.docstring)} |")
        else:
            has_constructor = any(method.name == "__init__" for method in definition.methods)
            lines += ["| Field | Type | Default | Description |", "|---|---|---|---|"]
            for member in definition.fields:
                missing = "see constructor" if has_constructor else "required"
                default = f"`{cell(member.default)}`" if member.default else missing
                annotation = f"`{cell(member.annotation)}`" if member.annotation else ""
                lines.append(f"| `{member.name}` | {annotation} | {default} | {cell(member.docstring)} |")
        lines.append("")
    if definition.methods:
        lines += ["**Methods**", ""]
        for member in definition.methods:
            description = f" — {member.docstring}" if member.docstring else ""
            lines.append(f"- `{member.annotation}`{indent(description)}")
        lines.append("")
    return lines


def is_enum(node: ast.AST) -> bool:
    return isinstance(node, ast.ClassDef) and any(ast.unparse(base) in ("Enum", "StrEnum") for base in node.bases)


def cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def indent(text: str) -> str:
    return text.replace("\n\n", "\n\n  ").replace("\n", "\n  ")


if __name__ == "__main__":
    sys.exit(main())
