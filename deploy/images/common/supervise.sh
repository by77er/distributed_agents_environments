#!/usr/bin/env bash
# What the pods' entrypoints share: start each of a pod's processes, and end them all when any one ends, so that the
# container exits and RunPod starts it again with nothing half running; or, for a process the others do not need
# (`restarting`), start it again alone whenever it ends. Sourced by an entrypoint.

# As many open files as the container may have: a machine's default soft limit can be too few for Envoy, vLLM and
# their connections.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

children=()

start() { # start NAME COMMAND...: run COMMAND in the background, its output prefixed with NAME
    local name=$1
    shift
    "$@" > >(sed -u "s/^/[$name] /") 2>&1 &
    children+=("$!")
}

restarting() { # restarting COMMAND...: run COMMAND, and again 5 seconds after each time it ends, until this is ended
    local child=
    trap '[ -n "$child" ] && kill -TERM "$child" 2>/dev/null && wait "$child"; exit 0' TERM
    while true; do
        "$@" &
        child=$!
        local status=0
        wait "$child" || status=$?
        child=
        echo "it ended (status $status): starting it again in 5 s" >&2
        sleep 5 &
        wait $! || true
    done
}

supervise() { # wait for the first process to end, then end the rest; exit with its status
    trap 'kill -TERM "${children[@]}" 2>/dev/null' TERM INT
    local status=0
    wait -n "${children[@]}" || status=$?
    echo "a process of the pod ended (status $status): ending the others" >&2
    kill -TERM "${children[@]}" 2>/dev/null || true
    wait || true
    exit "$status"
}
