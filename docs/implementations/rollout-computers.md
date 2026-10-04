# Computers

Code: `rollout_computers`

An environment is a computer that task code creates and acts on: it runs shell commands and reads and writes files.
Task code sees `run.environments` and `Environment` handles, which the harness defines. Every operation on them is an
effect, performed by an [`EnvironmentService`](../guide/reference.md#environmentservice): the backend a runner is
given. This package has two backends, and the tools that give an agent a computer. `run.environments` is `None` when
the runner has no backend.

```python
from rollout.harness import EnvironmentSpecification, RunContext, Task
from rollout.local import LocalRunner
from rollout_computers import NamespaceEnvironments
from rollout_computers.tools import ComputerTools

runner = LocalRunner(providers=providers, environments=NamespaceEnvironments(Path("state/environments")))


class Build(ComputerTools, Task):                      # the agent gets shell and file tools on the environment
    async def setup(self, run: RunContext) -> None:
        specification = EnvironmentSpecification(image="alpine", setup=["apk add git"])
        self.environment_id = (await run.environments.create(specification)).environment_id
```

## What task code sees

The types are in the reference: [`EnvironmentSpecification`](../guide/reference.md#environmentspecification),
[`Environments`](../guide/reference.md#environments), [`Environment`](../guide/reference.md#rolloutharnessenvironment) and
[`ExecutionResult`](../guide/reference.md#executionresult).

| Operation | Effect | Does |
|---|---|---|
| `run.environments.create(specification)` | `environment.lifecycle` | Creates an environment the run owns |
| `run.environments.attach(environment_id)` | none | A handle to an existing environment, which the run does not own |
| `environment.execute(command)` | `environment.call`, guarded | Runs a shell command |
| `environment.put(path, data)` | `environment.call` | Writes a file, creating parent directories; the event records the content's SHA-256 and size |
| `environment.get(path)` | `environment.call` | Reads a file; the event records its size |
| `environment.destroy()` | `environment.lifecycle` | Destroys the environment |

An environment's id is `e_` followed by the first 24 hex digits of the SHA-256 of the creating effect's `effect_id`.
A replayed or retried creation therefore finds the same environment instead of making another.

`execute` is guarded: under the durable runner a command that a crash interrupted is not run again, and the call
raises `OutcomeUnknown` ([durable runner](rollout-durable/README.md#effects)).

Both runners destroy the environments a run created when the run ends (`release_all`, which is not an effect). An
environment reached with `attach` is left alone.

## Backends

A backend implements `EnvironmentService`. Every method must be safe to repeat with the same arguments, except
`execute`.

| | `NamespaceEnvironments(directory, images=None)` | `LocalEnvironments(directory, *, variables=None)` |
|---|---|---|
| An environment is | Its own copy of an image's root filesystem, in `directory/{id}/rootfs` | A workspace directory on the host, `directory/{id}/workspace` |
| Images | `alpine` or `alpine:{version}` | Only `host`; any other image raises `ValueError` |
| A command runs | Through `unshare` in new user, mount and PID namespaces, chrooted, as root inside (the host user outside), with a clean set of environment variables | With the host's `/bin/sh` as the runner's user, with the runner's environment variables (or `variables`) plus `WORKSPACE` |
| Working directory | `/workspace`, or `cwd` | The workspace, or `cwd` under it |
| Paths in `put` and `get` | Relative: under `/workspace`; absolute: inside the root filesystem; a path that leaves it raises `ValueError` | Relative: under the workspace; absolute or `~`: the host's own |
| Files visible | Only the environment's; `/tmp` is fresh for each command | Everything the runner's user can read and change |
| Network | The host's | The host's |
| Full output of a truncated command | `/var/tmp/output-{hash}.log` in the environment | `directory/{id}/outputs/output-{hash}.log` |
| Isolation | Of files and processes; not a security boundary against hostile code | None |

Common to both:

- An environment is a directory, so it survives restarts of the runner. No process runs between commands.
- Creation is complete once the file `ready` exists; creating an existing environment does nothing. A `setup` command
  that fails raises `RuntimeError`.
- Both run commands through one function, `rollout_computers.processes.run_command`. The command runs in a process
  group of its own, with its output written to a file. The group is killed when the command ends, when it passes its
  timeout, and when the caller is cancelled: nothing in the group keeps running, whichever way the command ends.
  Under the namespace backend the whole process tree ends with its PID namespace. Under the local backend a process
  that starts a session of its own leaves the group and survives.
- Output longer than `MAX_OUTPUT_LINES` lines or `MAX_OUTPUT_BYTES` bytes keeps its end, in whole lines where
  possible. The full output is saved under a name derived from the `effect_id`, the same on every attempt.

`ImageStore(directory)` resolves `alpine` and `alpine:3.24.2` to Alpine's mini root filesystem for x86_64. It
downloads the tarball once, verifies it against its published SHA-256 and caches it. `alpine` without a version means
the latest stable release at first use, which the store then pins. `NamespaceEnvironments` keeps its store in
`directory/images` unless given one.

## Computer tools

`rollout_computers.tools.ComputerTools` is a mixin for a `Task` that gives its agent a computer. Set
`environment_id`, usually in `setup`. The methods are `@tool`s ([tools](../guide/tools.md)) over the environment, so
they work under any runner. Their limits are constants of `rollout_computers.tools`.

| Tool | Does |
|---|---|
| `shell(command, timeout_seconds)` | Runs a command; the timeout is kept between one second and `MAX_TIMEOUT_SECONDS`. The result starts with `exit N` or a timeout note, and is an error for a non-zero exit or a timeout. An `OutcomeUnknown` becomes an error result saying the command may or may not have run |
| `read_file(path, offset, limit)` | Reads text from line `offset`: at most `MAX_READ_LINES` lines or `MAX_READ_BYTES` bytes, with a note on how to continue. Binary files are refused |
| `write_file(path, content)` | Writes a text file, creating parent directories and replacing it if it exists |
| `edit_file(path, edits)` | Replaces exact text. Each `old_text` must occur exactly once in the original file, edits must not overlap, and edits that change nothing are refused. Line endings and a byte-order mark are kept |
| `read_image(path)` | Shows an image to the model as a `Media` block. PNG, JPEG, GIF and WebP go as they are; other formats become PNG. Images over `MAX_IMAGE_SIDE` pixels a side are scaled down, and the result stays under `MAX_IMAGE_BYTES`. It needs `run.blobs` ([content](../guide/content.md#media-and-blobs)) |

Writes and edits to the same file run one at a time, even when the model calls several tools at once.
