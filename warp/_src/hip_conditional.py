# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-side helpers for the HIP conditional graph emulation in :mod:`warp._src.context`."""

import warp

# Sequence numbers wrap within 30 bits so that ``(seq << 1) | flag`` fits a positive int32.
SEQUENCE_MASK = (1 << 30) - 1


@warp.kernel
def publish_condition(condition: warp.array(dtype=warp.int32), slot: warp.array(dtype=warp.int32)):
    """Publish ``condition[0] != 0`` to a pinned host ``slot`` together with a new sequence number.

    Value and sequence number are written with a single 32-bit store, so the host never observes
    a new sequence number paired with a stale value.
    """
    seq = ((slot[0] >> 1) + 1) & SEQUENCE_MASK
    flag = 0
    if condition[0] != 0:
        flag = 1
    slot[0] = (seq << 1) | flag
