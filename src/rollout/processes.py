"""Child processes that end with the process that started them."""

import contextlib
import ctypes
import signal


def end_with_parent() -> None:
    """In a child process (as it starts, or as a `preexec_fn`): have the kernel end it when its parent dies (Linux).
    A driver that is killed would otherwise leave its trainer holding the GPU, or its servers their memory."""
    with contextlib.suppress(OSError, AttributeError):
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
