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

# restarting COMMAND...: run COMMAND, and again after each time it ends, until this is ended: 5 seconds after, doubling
# up to 5 minutes while it keeps ending within a minute of its start. After 10 such quick ends running it is no longer
# started, and this waits (it never ends by itself, so the pod's other processes go on).
restarting() {
    local child= status wait=5 quick=0 began
    trap '[ -n "$child" ] && kill -TERM "$child" 2>/dev/null && wait "$child"; exit 0' TERM
    while true; do
        began=$SECONDS
        "$@" &
        child=$!
        status=0
        wait "$child" || status=$?
        child=
        if ((SECONDS - began < ${RESTART_QUICK:-60})); then
            quick=$((quick + 1))
        else
            quick=1 wait=5
        fi
        if ((quick >= ${RESTART_LIMIT:-10})); then
            echo "it ended (status $status) $quick times running within a minute of its start: not started again" >&2
            sleep infinity &
            child=$!
            wait "$child"
            exit 0
        fi
        echo "it ended (status $status): starting it again in $wait s" >&2
        sleep "$wait" &
        child=$!
        wait "$child" || true
        child=
        wait=$((wait * 2 > 300 ? 300 : wait * 2))
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
