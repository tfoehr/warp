# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind Warp to the ROCm runtime shipped in the Python environment.

When the ROCm SDK is installed as Python packages (``rocm[libraries]``, which
PyTorch ROCm wheels depend on), the HIP runtime libraries must be loaded from
those packages before ``warp.so`` is opened. Otherwise the dynamic loader may
resolve ``libamdhip64.so.7`` from a system ROCm install, and Warp and PyTorch
would end up sharing a HIP runtime that does not match the one PyTorch was
built against.
"""

import os

_preloaded = False


def _disable_hip_mempool_vmheap() -> None:
    """Make HIP stream-ordered allocations use the classic memory pool instead of the VM heap.

    With the VM-heap pool (the default in ROCm 7.x), memory that was freed back to the pool and
    handed out again can be zero-filled by a later host-to-device or device-to-host copy of
    another allocation from the same freed range (observed on ROCm 7.13, gfx1151, e.g. after
    freeing a block larger than 32 MiB): live arrays silently lose their contents. HIP reads
    the setting when its runtime initializes, so it must be in the environment before the first
    HIP call in the process. An explicit user setting is kept.
    """
    if "+rocm" not in _warp_version():
        return
    os.environ.setdefault("DEBUG_HIP_MEM_POOL_VMHEAP", "0")


def _warp_version() -> str:
    try:
        from warp.config import version  # noqa: PLC0415
    except ImportError:
        return ""
    return version


def preload_rocm_sdk_libraries() -> None:
    """Preload the HIP runtime from the ``rocm_sdk`` package, if present.

    Set ``WARP_ROCM_SDK_PRELOAD=0`` to skip this and use the system loader paths.
    """
    global _preloaded
    if _preloaded:
        return
    _preloaded = True

    _disable_hip_mempool_vmheap()

    if os.environ.get("WARP_ROCM_SDK_PRELOAD", "1") == "0":
        return

    try:
        import rocm_sdk  # noqa: PLC0415
    except ImportError:
        return

    try:
        rocm_sdk.initialize_process(preload_shortnames=["amd_comgr", "amdhip64", "hiprtc"])
    except Exception:
        # Partial ROCm SDK installs (e.g. without the libraries package) fall back to the system loader.
        pass
