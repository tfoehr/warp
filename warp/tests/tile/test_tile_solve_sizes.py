# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tile Cholesky and triangular-solve tests across tile sizes and block sizes.

Without libmathdx, single right-hand-side solves on HIP devices run on the first wavefront with
the rows spread over its lanes (``wave_triangular_solve_vec`` in ``tile_solve.h``). These tests
cover sizes below, at and above multiples of the wavefront size, blocks smaller than a
wavefront, both precisions, vector and ``(n, 1)`` right-hand sides, transposed views, and in-place
solves, against NumPy.
"""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

wp.set_module_options({"enable_mathdx_solver": False})

SIZES = {wp.float32: (5, 33, 64, 100), wp.float64: (5, 33, 64)}
BLOCK_DIMS = (16, 64)

_kernels = {}


def _get_kernels(n, dtype):
    key = (n, dtype)
    if key in _kernels:
        return _kernels[key]

    @wp.kernel(enable_backward=False, module="unique")
    def chol_kernel(A: wp.array2d[dtype], y: wp.array1d[dtype], L_out: wp.array2d[dtype], x: wp.array1d[dtype]):
        # Cholesky factorization and solve: A x = y
        a = wp.tile_load(A, shape=(n, n), storage="shared")
        wp.tile_cholesky_inplace(a)
        wp.tile_store(L_out, a)
        rhs = wp.tile_load(y, shape=(n,), storage="shared")
        wp.tile_cholesky_solve_inplace(a, rhs)
        wp.tile_store(x, rhs)

    @wp.kernel(enable_backward=False, module="unique")
    def lower_kernel(L: wp.array2d[dtype], y: wp.array1d[dtype], x: wp.array1d[dtype]):
        # L x = y
        lt = wp.tile_load(L, shape=(n, n), storage="shared")
        rhs = wp.tile_load(y, shape=(n,), storage="shared")
        wp.tile_store(x, wp.tile_lower_solve(lt, rhs))

    @wp.kernel(enable_backward=False, module="unique")
    def upper_kernel(L: wp.array2d[dtype], y: wp.array1d[dtype], x: wp.array1d[dtype]):
        # L^T x = y through a transposed view
        lt = wp.tile_load(L, shape=(n, n), storage="shared")
        rhs = wp.tile_load(y, shape=(n,), storage="shared")
        wp.tile_upper_solve_inplace(wp.tile_transpose(lt), rhs)
        wp.tile_store(x, rhs)

    @wp.kernel(enable_backward=False, module="unique")
    def inplace_2d_kernel(L: wp.array2d[dtype], y: wp.array2d[dtype]):
        # in place, (n, 1) right-hand side: L L^T x = y
        lt = wp.tile_load(L, shape=(n, n), storage="shared")
        r2 = wp.tile_load(y, shape=(n, 1), storage="shared")
        wp.tile_lower_solve_inplace(lt, r2)
        wp.tile_upper_solve_inplace(wp.tile_transpose(lt), r2)
        wp.tile_store(y, r2)

    _kernels[key] = (chol_kernel, lower_kernel, upper_kernel, inplace_2d_kernel)
    return _kernels[key]


def _run(test, device, n, block_dim, dtype):
    np_dtype = np.float32 if dtype == wp.float32 else np.float64
    rng = np.random.default_rng(1000 + n)
    B = rng.standard_normal((n, n))
    A_np = B @ B.T / n + 2.0 * np.eye(n)
    L_np = np.linalg.cholesky(A_np)
    y_np = rng.standard_normal(n)

    def arr(a):
        return wp.array(a.astype(np_dtype), dtype=dtype, device=device)

    A, L, y = arr(A_np), arr(L_np), arr(y_np)
    y2 = arr(y_np.reshape(n, 1))
    L_out = wp.zeros((n, n), dtype=dtype, device=device)
    x_chol = wp.zeros(n, dtype=dtype, device=device)
    x_lower = wp.zeros(n, dtype=dtype, device=device)
    x_upper = wp.zeros(n, dtype=dtype, device=device)

    chol_kernel, lower_kernel, upper_kernel, inplace_2d_kernel = _get_kernels(n, dtype)
    launch = {"dim": 1, "block_dim": block_dim, "device": device}
    wp.launch_tiled(chol_kernel, inputs=[A, y, L_out, x_chol], **launch)
    wp.launch_tiled(lower_kernel, inputs=[L, y, x_lower], **launch)
    wp.launch_tiled(upper_kernel, inputs=[L, y, x_upper], **launch)
    wp.launch_tiled(inplace_2d_kernel, inputs=[L, y2], **launch)

    tol = 2e-4 if dtype == wp.float32 else 1e-10
    x_ref = np.linalg.solve(A_np, y_np)
    np.testing.assert_allclose(L_out.numpy(), L_np, rtol=tol, atol=tol)
    np.testing.assert_allclose(x_chol.numpy(), x_ref, rtol=tol, atol=tol)
    np.testing.assert_allclose(x_lower.numpy(), np.linalg.solve(L_np, y_np), rtol=tol, atol=tol)
    np.testing.assert_allclose(x_upper.numpy(), np.linalg.solve(L_np.T, y_np), rtol=tol, atol=tol)
    np.testing.assert_allclose(y2.numpy()[:, 0], x_ref, rtol=tol, atol=tol)


def test_solve_sizes_float32(test, device):
    for n in SIZES[wp.float32]:
        for block_dim in BLOCK_DIMS:
            with test.subTest(n=n, block_dim=block_dim):
                _run(test, device, n, block_dim, wp.float32)


def test_solve_sizes_float64(test, device):
    for n in SIZES[wp.float64]:
        for block_dim in BLOCK_DIMS:
            with test.subTest(n=n, block_dim=block_dim):
                _run(test, device, n, block_dim, wp.float64)


class TestTileSolveSizes(unittest.TestCase):
    pass


devices = get_test_devices()

add_function_test(TestTileSolveSizes, "test_solve_sizes_float32", test_solve_sizes_float32, devices=devices)
add_function_test(TestTileSolveSizes, "test_solve_sizes_float64", test_solve_sizes_float64, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
