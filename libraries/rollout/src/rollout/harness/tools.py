"""`@tool` methods: the action space a task declares (docs/guide/tools.md).

A tool's specification comes from its signature (type hints → JSON Schema, via pydantic) and its docstring. Tool
bodies are task code, not effects; exceptions they raise are returned to the model as error results.
"""

import asyncio
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, cast, get_type_hints, overload

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError, create_model
from pydantic_core import to_jsonable_python

from rollout.contracts import (
    Media,
    Message,
    RetryClass,
    Role,
    Text,
    ToolCall,
    ToolResult,
    ToolResultBlock,
    ToolSpecification,
)

_TOOL_ATTRIBUTE = "__rollout_tool__"


@dataclass(frozen=True)
class ToolOptions:
    name: str | None = None
    retry_class: RetryClass = RetryClass.PURE
    """`@tool` methods are task code; the effects they make carry their own retry classes."""
    timeout: timedelta | None = None


@overload
def tool[F: Callable[..., Any]](function: F, /) -> F: ...
@overload
def tool[F: Callable[..., Any]](
    *, name: str | None = None, retry_class: RetryClass = RetryClass.PURE, timeout: timedelta | None = None
) -> Callable[[F], F]: ...
def tool[F: Callable[..., Any]](
    function: F | None = None,
    /,
    *,
    name: str | None = None,
    retry_class: RetryClass = RetryClass.PURE,
    timeout: timedelta | None = None,
) -> F | Callable[[F], F]:
    """Declare a task method as a tool: `@tool` or `@tool(name=..., retry_class=..., timeout=...)`."""
    options = ToolOptions(name=name, retry_class=retry_class, timeout=timeout)

    def mark(target: F) -> F:
        setattr(target, _TOOL_ATTRIBUTE, options)
        return target

    return mark(function) if function is not None else mark


@dataclass(frozen=True)
class DeclaredTool:
    """A `@tool` method collected from a task class."""

    attribute: str
    specification: ToolSpecification
    arguments_model: type[BaseModel]
    timeout: timedelta | None
    wants_run: bool = False
    """The method takes a `run` parameter: the framework passes the run context, and the model never sees it."""


def collect_tools(cls: type) -> dict[str, DeclaredTool]:
    """The `@tool` methods of a class and its bases, keyed by tool name, in definition order."""
    tools: dict[str, DeclaredTool] = {}
    seen: set[str] = set()
    for klass in cls.__mro__:
        for attribute, value in vars(klass).items():
            if attribute in seen:
                continue
            seen.add(attribute)
            options: ToolOptions | None = getattr(value, _TOOL_ATTRIBUTE, None)
            if options is None:
                continue
            declared = _declare(attribute, cast(Callable[..., Any], value), options)
            if declared.specification.name in tools:
                raise TypeError(f"{cls.__name__} declares the tool {declared.specification.name!r} twice")
            tools[declared.specification.name] = declared
    return tools


def _declare(attribute: str, function: Callable[..., Any], options: ToolOptions) -> DeclaredTool:
    signature = inspect.signature(function)
    hints = get_type_hints(function, include_extras=True)
    fields: dict[str, Any] = {}
    wants_run = False
    for parameter in list(signature.parameters.values())[1:]:  # skip self
        if parameter.name == "run":
            wants_run = True
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TypeError(f"tool {function.__qualname__} cannot take *args or **kwargs")
        annotation = hints.get(parameter.name, Any)
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[parameter.name] = (annotation, default)
    name = options.name or function.__name__
    arguments_model = create_model(  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
        f"{name}_arguments", __config__=ConfigDict(extra="forbid"), **fields
    )
    schema = cast(dict[str, JsonValue], arguments_model.model_json_schema())
    schema.pop("title", None)
    specification = ToolSpecification(
        name=name,
        description=inspect.getdoc(function) or "",
        input_schema=schema,
        retry_class=options.retry_class,
    )
    return DeclaredTool(attribute, specification, arguments_model, options.timeout, wants_run)


async def execute_tool(
    owner: object, declared: DeclaredTool, arguments: Mapping[str, JsonValue], run: object = None
) -> ToolResult:
    """Validate the arguments, run the tool body and normalize what it returns. Failures become error results."""
    try:
        validated = declared.arguments_model.model_validate(arguments)
    except ValidationError as error:
        return error_result(f"invalid arguments for {declared.specification.name}: {error}")
    method: Callable[..., Any] = getattr(owner, declared.attribute)
    keyword_arguments = {name: getattr(validated, name) for name in type(validated).model_fields}
    if declared.wants_run:
        keyword_arguments["run"] = run
    try:
        async with asyncio.timeout(declared.timeout.total_seconds() if declared.timeout else None):
            value = method(**keyword_arguments)
            if inspect.isawaitable(value):
                value = await value
    except TimeoutError:
        return error_result(f"{declared.specification.name} timed out after {declared.timeout}")
    except Exception as error:
        return error_result(f"{type(error).__name__}: {error}")
    return normalize_result(value)


def normalize_result(value: object) -> ToolResult:
    """`ToolResult` as is; `str` and blocks as content; anything else as structured JSON with a text rendering."""
    if isinstance(value, ToolResult):
        return value
    if isinstance(value, str):
        return ToolResult(content=[Text(text=value)])
    if isinstance(value, Text | Media):
        return ToolResult(content=[value])
    if isinstance(value, Sequence) and value and all(isinstance(item, Text | Media) for item in value):  # pyright: ignore[reportUnknownVariableType]
        return ToolResult(content=cast(Sequence[Text | Media], value))
    structured = cast(JsonValue, to_jsonable_python(value, fallback=str))
    return ToolResult(content=[Text(text=json.dumps(structured, ensure_ascii=False))], structured=structured)


def error_result(text: str) -> ToolResult:
    return ToolResult(content=[Text(text=text)], is_error=True)


def tool_message(calls: Sequence[ToolCall], results: Sequence[ToolResult]) -> Message:
    """One TOOL message answering every call, in call order."""
    blocks = [ToolResultBlock(call_id=call.call_id, result=result) for call, result in zip(calls, results, strict=True)]
    return Message(role=Role.TOOL, content=blocks)
