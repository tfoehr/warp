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


def preload_rocm_sdk_libraries() -> None:
    """Preload the HIP runtime from the ``rocm_sdk`` package, if present.

    Set ``WARP_ROCM_SDK_PRELOAD=0`` to skip this and use the system loader paths.
    """
    global _preloaded
    if _preloaded or os.environ.get("WARP_ROCM_SDK_PRELOAD", "1") == "0":
        return
    _preloaded = True

    try:
        import rocm_sdk  # noqa: PLC0415
    except ImportError:
        return

    try:
        rocm_sdk.initialize_process(preload_shortnames=["amd_comgr", "amdhip64", "hiprtc"])
    except Exception:
        # Partial ROCm SDK installs (e.g. without the libraries package) fall back to the system loader.
        pass
