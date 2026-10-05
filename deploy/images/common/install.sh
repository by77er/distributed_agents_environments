#!/usr/bin/env bash
# Installs workspace packages into /opt/rollout/venv: a virtual environment on the image's own Python that sees the
# image's packages, so that the image's PyTorch and CUDA libraries are the only ones. Run while a pod's image is built,
# with uv on the PATH:
#
#   install.sh export "rollout-train[http]" rollout-s3 ...
#       (from the repository's root) the packages' locked dependencies, as uv.lock pins them, on stdout; less Ray and
#       uv (what Ray installs environments with), which the pods do not run
#   install.sh dependencies LOCKED
#       the virtual environment, and in it each of LOCKED's packages at uv.lock's version where the image does not
#       have that version; except for the image's PyTorch and the CUDA libraries and Triton built with it (OWNED),
#       which are the image's, and whose torch must be uv.lock's release
#   install.sh packages DIRECTORY...
#       the workspace packages themselves (no dependencies), then a check that every requirement of what is in the
#       virtual environment is met there or by the image
#
# A process on the virtual environment's Python thus runs uv.lock's versions of everything but PyTorch's own build
# (`+cu130`); the image's own processes (vLLM's) see none of it.
set -euo pipefail

VENV=/opt/rollout/venv
PYTHON=${PYTHON:-/usr/bin/python3}
OWNED='torch|triton|nvidia-.*|cuda-.*'
PRUNED="ray uv"
export UV_NO_CACHE=1 UV_COMPILE_BYTECODE=1
unset UV_OVERRIDE # (a vLLM image's, for its own installs)

case "$1" in
export)
    shift
    options=()
    for spec in "$@"; do
        package=${spec%%\[*}
        options+=(--package "$package")
        if [ "$spec" != "$package" ]; then
            IFS=, read -r -a extras <<<"${spec#*\[}"
            for extra in "${extras[@]}"; do
                options+=(--extra "${extra%]}")
            done
        fi
    done
    for each in $PRUNED; do
        options+=(--prune "$each")
    done
    uv export --frozen --no-dev --no-emit-workspace --no-hashes --no-annotate --no-header "${options[@]}"
    ;;
dependencies)
    uv venv --system-site-packages --python "$PYTHON" "$VENV"
    "$PYTHON" - "$2" "$OWNED" >"$VENV/added.txt" <<'EOF'
import importlib.metadata
import re
import sys


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


installed = {canonical(each.metadata["Name"]): each.version for each in importlib.metadata.distributions()}
owned = re.compile(sys.argv[2])
for line in open(sys.argv[1]):
    name, _, version = line.partition(";")[0].strip().partition("==")
    if not version:
        continue
    name = canonical(name)
    have = installed.get(name)
    release = have and have.split("+")[0]  # (a local label, `+cu130`: the same release, built for its CUDA)
    if release == version:
        continue
    if have and owned.fullmatch(name):
        if name == "torch":
            sys.exit(f"the image's torch is {have}, and uv.lock's {version}")
        print(f"the image's {name} {have} (uv.lock: {version})", file=sys.stderr)
        continue
    print(line, end="")
EOF
    uv pip install --python "$VENV/bin/python" --no-deps -r "$VENV/added.txt"
    ;;
packages)
    shift
    uv pip install --python "$VENV/bin/python" --no-deps "$@"
    canonical() { tr '[:upper:]_.' '[:lower:]--'; }
    ours=$(uv pip list --python "$VENV/bin/python" --format freeze | sed 's/==.*//' | canonical)
    checked=$("$VENV/bin/python" -m pip check 2>&1 || true) # (pip sees the virtual environment and the image)
    unmet=0
    while read -r name rest; do
        if ! grep -qxF "$(canonical <<<"$name")" <<<"$ours"; then
            continue # (the image's own: what its processes see is the image's alone)
        fi
        for each in $PRUNED; do
            if [[ $rest == *"requires $each,"* ]]; then
                continue 2
            fi
        done
        echo "$name $rest" >&2
        unmet=1
    done <<<"$checked"
    exit "$unmet"
    ;;
*)
    echo "install.sh export|dependencies|packages ..." >&2
    exit 2
    ;;
esac
