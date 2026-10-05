# Import an environment from git

For environment authors who keep an environment in a git repository of its own: what the repository holds, what it may
depend on, and what a cluster checks when it imports it.

**Read first:** [checking an environment](testing.md#checking-an-environment). **Next:** [Train a
model](../train/README.md).

An environment in a git repository can be imported into a cluster from the monitor (**Import from git** on its
Environments page, [importing from git](../libraries/rollout-train/monitor.md#importing-from-git)), with nothing
redeployed: the platform fetches its source at a commit, keeps it in the blob store, checks it in the Python
environment it will run in, and records a version the cluster then offers. This page says what such a repository
holds, what its project may depend on, and what the import checks (`rollout_train.publishing`).

## What the repository holds

An environment is imported from a Python project: a directory with a `pyproject.toml`, at the repository's root or in
a subdirectory the import names. The project declares its environment as an entry point in the group
`rollout.environments`:

```toml title="pyproject.toml"
[project]
name = "say-the-word"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = ["rollout"]

[project.entry-points."rollout.environments"]
words = "words:environment"
```

- The entry point is `module:name`: `words:environment` is the object `environment` in the module `words`, an
  [environment](perspectives.md#building-an-environment) (`rollout.environment.Environment`: its program, rows,
  starts, eval data, description and version).
- Its name in the group (`words`) is the environment's name: the version it is imported as is `words@VERSION`. A project
  that declares no entry point is imported by the one the import names, and named after the project.
- A project that declares one environment is imported by it without saying more. One that declares several is
  imported one at a time, the import naming which, by its name in the group or by `module:name`.
- The module is imported from the project's directory, or from its `src/` for a project that keeps its packages there
  (`src/words/__init__.py`). The project is not installed: every file it needs is in its directory.

What is imported is the project's directory as it is at the commit, packed into one zip, stored uncompressed with fixed
times and modes, so that the same files are the same bytes. It leaves out `.git`, `.venv`, `__pycache__` and the
caches of pytest, mypy and ruff, compiled bytecode, and symbolic links; it holds 256 MiB at most. Data an environment
reads goes in the project, read relative to its module (`Path(__file__).parent`).

## Dependencies on the platform

The project's code runs beside the platform's: the run's training loop and runners import it in the same process. So
the platform's packages are the platform's own, and an environment depends on them by name, without a version
pinned or a URL: `rollout` for the environment's parts, and `rollout-train` where it uses that library (a trainer of its
own, say). An import leaves out every dependency the platform's Python holds already: the distribution installed, at a
version the requirement allows, with what the extras it asks for add. Its `requires-python` must allow the platform's
Python (3.13). Only `[project].dependencies` is read: `[tool.uv.sources]` and dependency groups are not.

The dependencies left are installed by Ray, with uv, into a copy of the platform's virtual environment: the run's
process and every Ray worker of its job start in it. Ray builds that copy on a node the first time a job there asks
for that list of dependencies, and keeps it while its Ray cluster lives; versions whose lists are the same share it.
A dependency the platform holds at another version is installed over the platform's in that copy, so the
environment's version is the one its code sees. A project whose dependencies the platform holds runs in the
platform's own Python, with nothing built, and its jobs start at once.

## What the import checks

An import is refused, with the reason the monitor shows under its form, where:

- **Fetching:** the URL does not clone at the ref (a branch, a tag or a commit; none: the default branch). Git asks for
  no credentials: the URL is one the platform can read.
- **The project:** the subdirectory has no `pyproject.toml`, it has no `[project]` with a name, its `requires-python`
  leaves out the platform's Python, or its files hold more than 256 MiB.
- **The entry point:** none is named and the project declares none, or several and none is named; it is no
  `module:name`; its module is neither in the project's directory nor in its `src/`.
- **The dependencies:** one is no requirement.
- **Its Python:** Ray could not build its runtime environment (uv could not install the dependencies left).
- **Loading:** the entry point does not import in its runtime environment.
- **The checks:** one of the checks `rollout env check` runs without a model fails: its rows build (keys and titles
  unique, `counts_for` naming rows it has), its description and version say something, one seed draws one start and its
  eval data is the same each time, training draws no eval start; and one
  [episode](../libraries/rollout-train/episodes.md) played with a scripted model ends, with a reward in the range its
  description gives and a result that says what it says it does ([checking an
  environment](testing.md#checking-an-environment)). The episode is not played where the environment's program imports a
  tool set or declares a sandbox, which an import cannot serve: that finding passes, flagged.

The checks run in a Ray job on the cluster, in the version's runtime environment, so an environment that imports in
the job is one a run can play. Before pushing, the same checks run in a checkout of the platform, with the project's
directory (or its `src/`) on the path:

```bash
PYTHONPATH=path/to/say-the-word uv run rollout env check words:environment
```

## Environment versions

A version's id is the SHA-256 of the project's zip: the same files are the same version, from whatever URL or ref they
were fetched, and importing them again returns the version recorded, without checking it again. A changed file is a new
version. The [ledger](../libraries/rollout-train/checkpoints.md#the-ledger) keeps each version beside it
(`rollout_train.published`): its name, source URL, the ref asked for and the commit it was, its subdirectory, entry
point, the blob, its runtime environment, what it says of itself (its version, description, rows, eval data, curriculum
and the kinds of sandbox its program declares, `sandboxes`, as its check found them) and the check's findings, and when
it was imported.

A cluster offers every version the ledger keeps, by `NAME@VERSION`, beside the environments its config names
(`GET /api/offers`), and the New run form lists them; a run on one is asked for as on any environment, and is refused
where the version declares a kind of sandbox the cluster has no pool of. A run on one is a Ray job in the version's
runtime environment ([launching runs](../libraries/rollout-train/launching.md#the-job)), and its start records the version (`published`: its name, id, source, ref, commit,
subdirectory and entry point).
