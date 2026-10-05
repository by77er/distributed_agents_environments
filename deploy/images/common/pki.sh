#!/usr/bin/env bash
# A pod's certificate, from the cluster's step-ca (deploy/images/README.md):
#
#   pki.sh bootstrap   the first certificate, with the one-time token in STEP_TOKEN (or the one on the volume, while it
#                      is still valid), then published for Envoy
#   pki.sh renew       renew it at about two thirds of its life, over mutual TLS with the certificate it has, and
#                      publish each new one (runs until killed)
#   pki.sh publish     make what step wrote the certificate Envoy serves (what `renew` runs after each renewal)
#
# step writes the certificate, its key and the root under $ROLLOUT_CERTS/live. Publishing copies them into a directory
# of their own and swaps the symbolic link $ROLLOUT_CERTS/current to it in one rename, which Envoy watches for: it
# reads the new pair whole, with no restart and no connection dropped. /certs is the directory Envoy's configuration
# names; it points at $ROLLOUT_CERTS, which is on the pod's volume so that a pod started again renews what it has.
set -euo pipefail

CERTS=${ROLLOUT_CERTS:-/workspace/certs}
LIVE=$CERTS/live
PYTHON=/opt/rollout/venv/bin/python
export STEPPATH=$CERTS/.step

need() {
    for name in "$@"; do
        if [ -z "${!name:-}" ]; then
            echo "pki.sh: $name is not set" >&2
            exit 1
        fi
    done
}

bootstrap() {
    need STEP_CA_URL STEP_FINGERPRINT ROLLOUT_POD_NAME
    local identity="spiffe://rollout/pod/$ROLLOUT_POD_NAME"
    mkdir -p "$LIVE" "$STEPPATH"
    chmod 700 "$CERTS" "$LIVE"
    if [ "$CERTS" != /certs ]; then
        ln -sfn "$CERTS" /certs
    fi
    if [ ! -s "$LIVE/ca.crt" ]; then
        step ca root "$LIVE/ca.crt" --ca-url "$STEP_CA_URL" --fingerprint "$STEP_FINGERPRINT" --force
    fi
    if [ -s "$LIVE/tls.crt" ] && step certificate verify "$LIVE/tls.crt" --roots "$LIVE/ca.crt" >/dev/null 2>&1; then
        echo "pki.sh: the certificate on the volume is still valid: it is renewed, not issued again"
    else
        need STEP_TOKEN
        # The key is made here and never leaves the pod; the token names this identity, and step-ca takes it once.
        step ca certificate "$identity" "$LIVE/tls.crt" "$LIVE/tls.key" --token "$STEP_TOKEN" \
            --ca-url "$STEP_CA_URL" --root "$LIVE/ca.crt" --kty EC --curve P-256 --force
    fi
    chmod 600 "$LIVE/tls.key"
    publish
}

renew() {
    need STEP_CA_URL
    # --daemon renews at about two thirds of the certificate's life (with jitter), for as long as step-ca renews it:
    # a certificate its starter had revoked is not renewed, and lapses.
    exec step ca renew --daemon --ca-url "$STEP_CA_URL" --root "$LIVE/ca.crt" \
        --exec "$0 publish" "$LIVE/tls.crt" "$LIVE/tls.key"
}

publish() {
    local version
    version=$CERTS/.v$(date +%s%N)
    mkdir -m 700 "$version"
    cp "$LIVE/tls.crt" "$LIVE/tls.key" "$LIVE/ca.crt" "$version/"
    # (the serial the pod's beats say, so that its starter can have this certificate revoked; empty if unread)
    step certificate inspect "$LIVE/tls.crt" --format json |
        "$PYTHON" -c 'import json, sys; print(json.load(sys.stdin)["serial_number"])' >"$version/serial" ||
        : >"$version/serial"
    ln -sfn "$(basename "$version")" "$CERTS/.current.next"
    mv -T "$CERTS/.current.next" "$CERTS/current"
    # (the one before stays, for a reader that began on it; those before it are deleted)
    find "$CERTS" -maxdepth 1 -name '.v*' -type d | sort | head -n -2 | xargs -r rm -rf
    echo "pki.sh: serving certificate $(cat "$version/serial")"
}

case "${1:-}" in
bootstrap) bootstrap ;;
renew) renew ;;
publish) publish ;;
*)
    echo "usage: pki.sh bootstrap | renew | publish" >&2
    exit 2
    ;;
esac
