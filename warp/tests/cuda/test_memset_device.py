# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for device memsets (``wp.array.zero_`` and the native ``memset`` helper).

On HIP devices Warp sets memory with its own single-dispatch kernel instead of
``hipMemsetAsync``, which splits ranges that are not 16-byte multiples into up to three
fill kernels. The kernel stores whole 16-byte chunks as vectors and the partial chunks at
both ends byte by byte, so the tests cover every head/tail alignment, sizes around the
chunk size, and that no byte outside the range changes, eagerly and in a captured graph.
"""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

GUARD = 0x5A
SIZES = (0, 1, 2, 3, 4, 5, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 100, 1000, 4099)


def _check_memsets(test, device, capture):
    buf_len = 4200 + 64
    expected = np.full(buf_len, GUARD, dtype=np.uint8)
    buf = wp.array(expected, dtype=wp.uint8, device=device)

    cases = []
    for offset in range(17):
        for size in SIZES:
            value = (offset * 37 + size) % 251 + 1
            cases.append((offset, size, value))

    for offset, size, value in cases:
        expected[:] = GUARD
        buf.assign(expected)
        if capture:
            with wp.ScopedCapture(device=device, force_module_load=False) as cap:
                device.memset(buf.ptr + offset, value, size)
            wp.capture_launch(cap.graph)
        else:
            device.memset(buf.ptr + offset, value, size)
        expected[offset : offset + size] = value
        result = buf.numpy()
        if not np.array_equal(result, expected):
            bad = np.nonzero(result != expected)[0]
            test.fail(f"memset(offset={offset}, size={size}, value={value}) wrong at bytes {bad[:8]}")


def test_memset_alignments(test, device):
    _check_memsets(test, device, capture=False)


def test_memset_alignments_captured(test, device):
    _check_memsets(test, device, capture=True)


def test_zero_large(test, device):
    # larger than the kernel's grid-stride threshold (4096 blocks * 256 threads * 16 bytes)
    n = 20 * 1024 * 1024 + 3
    a = wp.full(n + 8, 7, dtype=wp.uint8, device=device)
    view = wp.array(ptr=a.ptr + 3, shape=(n,), dtype=wp.uint8, device=device)
    view.zero_()
    result = a.numpy()
    test.assertTrue(np.all(result[:3] == 7))
    test.assertTrue(np.all(result[3 : 3 + n] == 0))
    test.assertTrue(np.all(result[3 + n :] == 7))


def test_zero_float_arrays(test, device):
    for n in (1, 3, 5, 17, 1001):
        a = wp.full(n, 1.5, dtype=wp.float32, device=device)
        a.zero_()
        assert_np_equal(a.numpy(), np.zeros(n, dtype=np.float32))


cuda_devices = get_cuda_test_devices()


class TestMemsetDevice(unittest.TestCase):
    pass


add_function_test(TestMemsetDevice, "test_memset_alignments", test_memset_alignments, devices=cuda_devices)
add_function_test(
    TestMemsetDevice, "test_memset_alignments_captured", test_memset_alignments_captured, devices=cuda_devices
)
add_function_test(TestMemsetDevice, "test_zero_large", test_zero_large, devices=cuda_devices)
add_function_test(TestMemsetDevice, "test_zero_float_arrays", test_zero_float_arrays, devices=cuda_devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
