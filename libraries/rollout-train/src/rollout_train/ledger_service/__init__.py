"""The ledger over HTTP: the ledger service (`app`), the client every role can use by URL (`HttpLedger`), and the tokens
that say what each may do (`scopes`).

The service serves the ledger the platform already has (a database, in a deployment) and the stores beside it. The
platform's roles hold the platform's token and may do everything; a pod holds a token of its own (`pod_token`), with
which it reads its run's serving records and checkpoints and writes its own beats, nothing else.
"""

from rollout_train.ledger_service.client import HttpLedger, LedgerUnreachable
from rollout_train.ledger_service.scopes import PLATFORM, Forbidden, Scope, pod_token, scope_of
from rollout_train.ledger_service.service import app
from rollout_train.ledger_service.wire import Conflict

__all__ = [
    "PLATFORM",
    "Conflict",
    "Forbidden",
    "HttpLedger",
    "LedgerUnreachable",
    "Scope",
    "app",
    "pod_token",
    "scope_of",
]
