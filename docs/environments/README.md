# Environments

Status: **Working** (2026-10-02) · Code: `rollout.harness.environments`, `rollout_computers`

An environment is a computer that task code creates and acts on: it runs shell commands and reads and writes files.
Task code sees `run.environments` and `Environment` handles. Every operation on them is an effect, performed by an
`EnvironmentService`: the backend a runner is given. `run.environments` is `None` when the runner has no backend.

```python
from rollout.harness import EnvironmentSpecification, RunContext, Task
from rollout_computers import NamespaceEnvironments
from rollout_computers.tools import ComputerTools

runner = LocalRunner(providers=providers, environments=NamespaceEnvironments(Path("state/environments")))


class Build(ComputerTools, Task):                      # the agent gets shell and file tools on the environment
    async def setup(self, run: RunContext) -> None:
        specification = EnvironmentSpecification(image="alpine", setup=["apk add git"])
        self.environment_id = (await run.environments.create(specification)).environment_id
```

## What task code sees

`rollout.harness` exports these types.

| `EnvironmentSpecification` field | Default | Meaning |
|---|---|---|
| `image: str` | `"alpine"` | a base image the backend knows |
| `setup: Sequence[str]` | `()` | shell commands run once at creation, in order |
| `labels: Mapping[str, str]` | `{}` | free-form labels, stored with the environment |

| `run.environments` (`Environments`) | Effect | Does |
|---|---|---|
| `await create(specification: EnvironmentSpecification \| None = None) -> Environment` | `environment.lifecycle` | creates an environment the run owns |
| `attach(environment_id: str) -> Environment` | none | a handle to an existing environment |
| `owned: list[str]` | | the ids of the environments this run created |
| `await release_all() -> None` | none | destroys every owned environment; runners call it when a run ends |

| `Environment` | Effect | Does |
|---|---|---|
| `environment_id: str` | | the id, to keep on `self` or pass to `attach` |
| `await execute(command: str, *, timeout_seconds: float = 120.0, cwd: str \| None = None) -> ExecutionResult` | `environment.call`, guarded | runs a shell command |
| `await put(path: str, data: bytes \| str) -> None` | `environment.call` | writes a file, creating parent directories; the event records the content's SHA-256 and size |
| `await get(path: str) -> bytes` | `environment.call` | reads a file; the event records its size |
| `await destroy() -> None` | `environment.lifecycle` | destroys the environment |

| `ExecutionResult` field | Meaning |
|---|---|
| `exit_code: int \| None` | `None` when the command timed out |
| `output: str` | standard output and standard error, interleaved |
| `truncated: bool` | `output` is only the end of the output |
| `timed_out: bool` | the command ran past `timeout_seconds` and was killed |
| `full_output_path: str \| None` | when truncated: where the full output was saved |

An environment's id is `e_` followed by the first 24 hex digits of the SHA-256 of the creating effect's `effect_id`.
A replayed or retried creation therefore finds the same environment instead of making another.

`execute` is guarded: under the durable runner a command that a crash interrupted is not run again, and the call
raises `OutcomeUnknown` ([durability](../durability/README.md#effects)).

Both runners destroy the environments a run created when the run ends. An environment reached with `attach` is not
owned and is left alone.

## Backends

A backend implements `EnvironmentService`. Every method must be safe to repeat with the same arguments, except
`execute`.

```python
class EnvironmentService(Protocol):
    async def create(self, environment_id: str, specification: EnvironmentSpecification) -> None: ...
    async def execute(
        self, environment_id: str, command: str, *, timeout_seconds: float, cwd: str | None, effect_id: str = ""
    ) -> ExecutionResult: ...
    async def put(self, environment_id: str, path: str, data: bytes) -> None: ...
    async def get(self, environment_id: str, path: str) -> bytes: ...
    async def destroy(self, environment_id: str) -> None: ...
```

`rollout_computers` has two.

| | `NamespaceEnvironments(directory, images=None)` | `LocalEnvironments(directory, *, variables=None)` |
|---|---|---|
| An environment is | its own copy of an image's root filesystem, in `directory/{id}/rootfs` | a workspace directory on the host, `directory/{id}/workspace` |
| Images | `alpine` or `alpine:{version}` | only `host`; any other image raises `ValueError` |
| A command runs | through `unshare` in new user, mount and PID namespaces, chrooted, as root inside (the host user outside), with a clean set of environment variables | with the host's `/bin/sh` as the runner's user, with the runner's environment variables (or `variables`) plus `WORKSPACE` |
| Working directory | `/workspace`, or `cwd` | the workspace, or `cwd` under it |
| Paths in `put` and `get` | relative: under `/workspace`; absolute: inside the root filesystem; a path that leaves it raises `ValueError` | relative: under the workspace; absolute or `~`: the host's own |
| Files visible | only the environment's; `/tmp` is fresh for each command | everything the runner's user can read and change |
| Network | the host's | the host's |
| After a command | its whole process tree is gone | its process group is killed; a process that started its own session survives |
| Full output of a truncated command | `/var/tmp/output-{hash}.log` in the environment | `directory/{id}/outputs/output-{hash}.log` |
| Isolation | of files and processes; not a security boundary against hostile code | none |

Common to both:

- An environment is a directory, so it survives restarts of the runner. No process runs between commands.
- Creation is complete once the file `ready` exists; creating an existing environment does nothing. `setup` commands
  have 600 seconds each, and one that fails raises `RuntimeError`.
- A command that passes its timeout is killed with its process group.
- Output longer than 2000 lines or 50 KiB keeps its end, in whole lines where possible. The full output is saved under
  a name derived from the `effect_id`, the same on every attempt.

`ImageStore(directory)` resolves `alpine` and `alpine:3.24.2` to Alpine's mini root filesystem for x86_64. It
downloads the tarball once, verifies it against its published SHA-256 and caches it. `alpine` without a version means
the latest stable release at first use, which the store then pins. `NamespaceEnvironments` keeps its store in
`directory/images` unless given one.

## Computer tools

`rollout_computers.tools.ComputerTools` is a mixin for a `Task` that gives its agent a computer. Set
`environment_id`, usually in `setup`. The methods are `@tool`s ([tools](../guide/tools.md)) over the environment, so
they work under any runner.

| Tool | Does |
|---|---|
| `shell(command: str, timeout_seconds: float = 120)` | runs a command; the timeout is kept between 1 and 1800 seconds. The result starts with `exit N` or a timeout note, and is an error for a non-zero exit or a timeout. An `OutcomeUnknown` becomes an error result saying the command may or may not have run |
| `read_file(path: str, offset: int = 1, limit: int \| None = None)` | reads text from line `offset`: at most 2000 lines or 50 KiB, with a note on how to continue. Binary files are refused |
| `write_file(path: str, content: str)` | writes a text file, replacing it if it exists |
| `edit_file(path: str, edits: list[Replacement])` | replaces exact text. Each `old_text` must occur exactly once in the original file, and edits must not overlap. Line endings and a byte-order mark are kept |
| `read_image(path: str)` | shows an image to the model as a `Media` block. PNG, JPEG, GIF and WebP go as they are; other formats become PNG. Images over 2000 pixels a side are scaled down, and the result stays under 4.5 MB. It needs `run.blobs` ([content](../guide/content.md)) and Pillow |

Writes and edits to the same file run one at a time, even when the model calls several tools at once.
