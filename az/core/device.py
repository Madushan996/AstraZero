"""Pick a torch device that actually works.

`torch.cuda.is_available()` is not sufficient. It returns True for a GPU whose compute
capability the installed PyTorch has no compiled kernels for, and the failure only
surfaces when a kernel runs -- as `cudaErrorNoKernelImageForDevice`, deep inside
self-play, after the work has already started.

Kaggle hits this routinely: it hands out Tesla P100s (sm_60) while its own PyTorch build
supports sm_70 and above. A whole session died this way, reporting RUNNING throughout.

So probe with a real operation before trusting the GPU, and fall back rather than fail.
"""

from __future__ import annotations

import torch


def select_device(preference: str = "cuda", verbose: bool = True) -> torch.device:
    """Return a usable device, falling back to CPU if the GPU cannot run a kernel."""
    if preference == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        # A matmul forces real kernel launch and compilation; allocation alone passes
        # on an incompatible card.
        probe = torch.zeros((8, 8), device="cuda")
        (probe @ probe).sum().item()
        return torch.device("cuda")
    except Exception as error:
        if verbose:
            name = "unknown GPU"
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:
                pass
            print(
                f"[device] {name} is unusable with this PyTorch build "
                f"({type(error).__name__}); using CPU instead. Self-play is bound by "
                f"Python tree search, so this costs perhaps 20-30%, not everything.",
                flush=True,
            )
        return torch.device("cpu")


def describe(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    try:
        return f"cuda ({torch.cuda.get_device_name(0)})"
    except Exception:
        return "cuda"
