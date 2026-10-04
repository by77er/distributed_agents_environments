#!/usr/bin/env bash
# Installs workspace packages, with their dependencies exactly as uv.lock pins them, into /opt/rollout/venv on a Python
# 3.13 of its own (the image's own Python, if it has one, is left as it is).
#
#   install.sh "rollout-train[http]" rollout-s3 -- libraries/rollout libraries/rollout-train ...
#
# Before `--`: the packages whose locked dependencies are installed, each with its extras. After it: the workspace
# packages themselves, by directory. Run from the repository's root, in the image being built.
set -euo pipefail

export UV_PYTHON_INSTALL_DIR=/opt/rollout/python
uv python install 3.13
uv venv --python 3.13 /opt/rollout/venv

requirements=()
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do
    spec=$1
    package=${spec%%\[*}
    extras=()
    if [ "$spec" != "$package" ]; then
        IFS=, read -r -a listed <<<"${spec#*\[}"
        for extra in "${listed[@]}"; do
            extras+=(--extra "${extra%]}")
        done
    fi
    file=/tmp/requirements-$package.txt
    uv export --frozen --no-dev --no-emit-workspace --package "$package" "${extras[@]}" -o "$file" >/dev/null
    requirements+=(-r "$file")
    shift
done
shift # (the --)

uv pip install --python /opt/rollout/venv/bin/python "${requirements[@]}"
uv pip install --python /opt/rollout/venv/bin/python --no-deps "$@"
rm -f /tmp/requirements-*.txt
