# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the HIP/ROCm emulation of conditional graph nodes.

HIP has no conditional graph nodes, so ``wp.capture_if()``/``wp.capture_while()`` calls made during
graph capture are emulated with host-driven segmented graphs (see
``warp.config.hip_conditional_graph_emulation``). These tests compare emulated graph replays against
eager (uncaptured) execution of the same Python code.

Run the per-iteration overhead microbenchmark with::

    python -m warp.tests.cuda.test_hip_conditional_emulation --bench [iterations]
"""

import os
import sys
import tempfile
import time
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def accumulate_kernel(src: wp.array(dtype=wp.int32), dst: wp.array(dtype=wp.int32)):
    dst[0] = dst[0] + src[0]


@wp.kernel
def add_kernel(a: wp.array[wp.int32], value: int):
    tid = wp.tid()
    a[tid] = a[tid] + value


@wp.kernel
def scale_and_check_kernel(a: wp.array[wp.int32], cond: wp.array[wp.int32], limit: int):
    tid = wp.tid()
    a[tid] = a[tid] * 2
    if tid == 0 and a[0] > limit:
        cond[0] = 0


@wp.kernel
def set_divisible_kernel(a: wp.array[wp.int32], flag: wp.array[wp.int32], divisor: int):
    if a[0] % divisor == 0:
        flag[0] = 1
    else:
        flag[0] = 0


@wp.kernel
def set_kernel(c: wp.array[wp.int32], value: int):
    c[0] = value


@wp.kernel
def decrement_kernel(c: wp.array[wp.int32]):
    c[0] = c[0] - 1


def launch_add(a, value):
    wp.launch(add_kernel, dim=a.size, inputs=[a, value])


def nested_program(a, w, flag, inner):
    """A while loop containing an if/else and an inner while loop, all data dependent."""

    def inner_body():
        launch_add(a, 5)
        wp.launch(decrement_kernel, dim=1, inputs=[inner])

    def outer_body():
        wp.launch(scale_and_check_kernel, dim=a.size, inputs=[a, w, 1000])
        wp.launch(set_divisible_kernel, dim=1, inputs=[a, flag, 3])
        wp.capture_if(flag, on_true=lambda: launch_add(a, 1), on_false=lambda: launch_add(a, -1))
        wp.launch(set_kernel, dim=1, inputs=[inner, 2])
        wp.capture_while(inner, while_body=inner_body)

    wp.capture_while(w, while_body=outer_body)


def nested_program_reference(values):
    """NumPy reference for :func:`nested_program` starting from ``w = 1``."""
    a = np.array(values, dtype=np.int64)
    while True:
        a = a * 2
        done = a[0] > 1000
        a = a + (1 if a[0] % 3 == 0 else -1)
        a = a + 10
        if done:
            return a


class _ConfigOverride:
    def __init__(self, **settings):
        self.settings = settings
        self.saved = {}

    def __enter__(self):
        for name, value in self.settings.items():
            self.saved[name] = getattr(wp.config, name)
            setattr(wp.config, name, value)

    def __exit__(self, *exc):
        for name, value in self.saved.items():
            setattr(wp.config, name, value)


def test_reports_supported(test, device):
    with _ConfigOverride(hip_conditional_graph_emulation=True):
        test.assertTrue(wp.is_conditional_graph_supported())
    with _ConfigOverride(hip_conditional_graph_emulation=False):
        test.assertFalse(wp.is_conditional_graph_supported())


def test_plain_graph_unchanged(test, device):
    with wp.ScopedDevice(device):
        a = wp.zeros(4, dtype=wp.int32)
        wp.load_module(device=device)
        with wp.ScopedCapture(force_module_load=False) as capture:
            launch_add(a, 3)

        test.assertIsNone(capture.graph._program)
        test.assertTrue(capture.graph.graph)
        wp.capture_launch(capture.graph)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(a.numpy(), [6, 6, 6, 6])


def test_if_else(test, device):
    with wp.ScopedDevice(device):
        a = wp.zeros(4, dtype=wp.int32)
        cond = wp.zeros(1, dtype=wp.int32)
        wp.load_module(device=device)

        with wp.ScopedCapture(force_module_load=False) as capture:
            launch_add(a, 1)
            wp.capture_if(cond, on_true=lambda: launch_add(a, 10), on_false=lambda: launch_add(a, 100))
            launch_add(a, 1000)

        test.assertIsNotNone(capture.graph._program)
        test.assertIsNone(capture.graph.graph)

        for value, expected in ((0, 1101), (1, 1011), (7, 1011), (-1, 1011)):
            a.zero_()
            cond.fill_(value)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(a.numpy(), [expected] * 4)


def test_if_only_and_else_only(test, device):
    with wp.ScopedDevice(device):
        a = wp.zeros(2, dtype=wp.int32)
        cond = wp.zeros(1, dtype=wp.int32)
        wp.load_module(device=device)

        with wp.ScopedCapture(force_module_load=False) as if_capture:
            wp.capture_if(cond, on_true=lambda: launch_add(a, 1))
        with wp.ScopedCapture(force_module_load=False) as else_capture:
            wp.capture_if(cond, on_false=lambda: launch_add(a, 2))

        for value in (0, 1):
            a.zero_()
            cond.fill_(value)
            wp.capture_launch(if_capture.graph)
            wp.capture_launch(else_capture.graph)
            np.testing.assert_array_equal(a.numpy(), [1 if value else 2] * 2)


def test_while(test, device):
    with wp.ScopedDevice(device):
        a = wp.zeros(3, dtype=wp.int32)
        count = wp.zeros(1, dtype=wp.int32)
        wp.load_module(device=device)

        def body():
            launch_add(a, 1)
            wp.launch(decrement_kernel, dim=1, inputs=[count])

        with wp.ScopedCapture(force_module_load=False) as capture:
            wp.capture_while(count, while_body=body)

        # repeated replays of the same graph, including zero iterations
        for iterations in (5, 0, 1, 17, 0):
            a.zero_()
            count.fill_(iterations)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(a.numpy(), [iterations] * 3)
            test.assertEqual(count.numpy()[0], 0)


def test_nested_matches_eager(test, device):
    with wp.ScopedDevice(device):
        initial = [1, 2, 3, 4]
        a = wp.array(initial, dtype=wp.int32)
        w = wp.ones(1, dtype=wp.int32)
        flag = wp.zeros(1, dtype=wp.int32)
        inner = wp.zeros(1, dtype=wp.int32)
        wp.load_module(device=device)

        # eager execution (host-driven conditionals without capture)
        nested_program(a, w, flag, inner)
        eager = a.numpy()
        np.testing.assert_array_equal(eager, nested_program_reference(initial))

        with wp.ScopedCapture(force_module_load=False) as capture:
            nested_program(a, w, flag, inner)

        for _ in range(3):
            a.assign(initial)
            w.fill_(1)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(a.numpy(), eager)


def test_if_inside_while_body_graph(test, device):
    """A captured program (with conditionals) used as a while body is spliced into the loop."""
    with wp.ScopedDevice(device):
        a = wp.zeros(1, dtype=wp.int32)
        flag = wp.zeros(1, dtype=wp.int32)
        count = wp.zeros(1, dtype=wp.int32)
        wp.load_module(device=device)

        with wp.ScopedCapture(force_module_load=False) as body_capture:
            wp.launch(set_divisible_kernel, dim=1, inputs=[count, flag, 2])
            wp.capture_if(flag, on_true=lambda: launch_add(a, 1), on_false=lambda: launch_add(a, 100))
            wp.launch(decrement_kernel, dim=1, inputs=[count])
        test.assertIsNotNone(body_capture.graph._program)

        with wp.ScopedCapture(force_module_load=False) as capture:
            wp.capture_while(count, while_body=body_capture.graph)

        # counts 6..1: three even (+1) and three odd (+100) values
        count.fill_(6)
        wp.capture_launch(capture.graph)
        test.assertEqual(a.numpy()[0], 303)


def test_plain_graph_bodies(test, device):
    with wp.ScopedDevice(device):
        a = wp.zeros(2, dtype=wp.int32)
        cond = wp.zeros(1, dtype=wp.int32)
        count = wp.zeros(1, dtype=wp.int32)
        wp.load_module(device=device)

        with wp.ScopedCapture(force_module_load=False) as add_one:
            launch_add(a, 1)
        with wp.ScopedCapture(force_module_load=False) as add_ten:
            launch_add(a, 10)
        with wp.ScopedCapture(force_module_load=False) as loop_body:
            launch_add(a, 1000)
            wp.launch(decrement_kernel, dim=1, inputs=[count])

        with wp.ScopedCapture(force_module_load=False) as capture:
            wp.capture_if(cond, on_true=add_one.graph, on_false=add_ten.graph)
            wp.capture_while(count, while_body=loop_body.graph)

        for value in (0, 1):
            a.zero_()
            cond.fill_(value)
            count.fill_(2)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(a.numpy(), [2000 + (1 if value else 10)] * 2)

        # the body graphs remain usable on their own
        a.zero_()
        wp.capture_launch(add_one.graph)
        np.testing.assert_array_equal(a.numpy(), [1, 1])


def test_graph_in_graph_splice(test, device):
    with wp.ScopedDevice(device):
        a = wp.zeros(1, dtype=wp.int32)
        count = wp.zeros(1, dtype=wp.int32)
        wp.load_module(device=device)

        def body():
            launch_add(a, 2)
            wp.launch(decrement_kernel, dim=1, inputs=[count])

        with wp.ScopedCapture(force_module_load=False) as inner:
            wp.capture_while(count, while_body=body)
        with wp.ScopedCapture(force_module_load=False) as plain:
            launch_add(a, 1000)

        with wp.ScopedCapture(force_module_load=False) as outer:
            wp.launch(set_kernel, dim=1, inputs=[count, 3])
            wp.capture_launch(inner.graph)
            wp.capture_launch(plain.graph)
            wp.launch(set_kernel, dim=1, inputs=[count, 1])
            wp.capture_launch(inner.graph)

        program = outer.graph._program
        test.assertIsNotNone(program)
        test.assertIn(inner.graph, outer.graph._program_refs)
        test.assertIn(plain.graph, outer.graph._program_refs)

        # the outer graph keeps the spliced work alive on its own
        del inner, plain
        for _ in range(2):
            a.zero_()
            wp.capture_launch(outer.graph)
            test.assertEqual(a.numpy()[0], 3 * 2 + 1000 + 2)


def test_kwargs_forwarded(test, device):
    with wp.ScopedDevice(device):
        a = wp.zeros(1, dtype=wp.int32)
        cond = wp.ones(1, dtype=wp.int32)
        count = wp.zeros(1, dtype=wp.int32)
        wp.load_module(device=device)
        seen = []

        # ``retry`` is forwarded to the body like any other keyword (Newton's kamino solver relies on this)
        def if_body(arr, value, retry=False):
            seen.append(("if", retry))
            launch_add(arr, value)

        def while_body(arr, counter, value):
            seen.append(("while", value))
            launch_add(arr, value)
            wp.launch(decrement_kernel, dim=1, inputs=[counter])

        with wp.ScopedCapture(force_module_load=False) as capture:
            wp.capture_if(cond, on_true=if_body, arr=a, value=7, retry=True)
            wp.capture_while(count, while_body=while_body, arr=a, counter=count, value=3)

        # bodies are traced exactly once during capture
        test.assertEqual(seen, [("if", True), ("while", 3)])

        count.fill_(4)
        wp.capture_launch(capture.graph)
        test.assertEqual(a.numpy()[0], 7 + 4 * 3)
        test.assertEqual(len(seen), 2)


def test_body_error_restores_capture(test, device):
    with wp.ScopedDevice(device):
        a = wp.zeros(1, dtype=wp.int32)
        cond = wp.ones(1, dtype=wp.int32)
        wp.load_module(device=device)

        def bad_body():
            launch_add(a, 1)
            raise ValueError("boom")

        for make_call in (
            lambda: wp.capture_if(cond, on_true=bad_body),
            lambda: wp.capture_while(cond, while_body=bad_body),
            lambda: wp.capture_if(cond, on_true=lambda: wp.capture_while(cond, while_body=bad_body)),
            lambda: wp.capture_if(cond, on_true=42),
        ):
            with test.assertRaises((ValueError, TypeError)):
                with wp.ScopedCapture(force_module_load=False):
                    launch_add(a, 1)
                    make_call()
            test.assertFalse(device.captures)
            test.assertFalse(device.stream.is_capturing)

        # capture still works afterwards
        with wp.ScopedCapture(force_module_load=False) as capture:
            wp.capture_if(cond, on_true=lambda: launch_add(a, 5))
        a.zero_()
        wp.capture_launch(capture.graph)
        test.assertEqual(a.numpy()[0], 5)


def test_alloc_in_body(test, device):
    """Allocations in emulated bodies are graph-retained device memory and stay valid across replays."""
    with wp.ScopedDevice(device):
        total = wp.zeros(1, dtype=wp.int32)
        count = wp.zeros(1, dtype=wp.int32)
        wp.load_module(device=device)

        def body():
            tmp = wp.full(1, value=2, dtype=wp.int32)  # allocated inside the loop body
            wp.launch(accumulate_kernel, dim=1, inputs=[tmp, total])
            wp.launch(decrement_kernel, dim=1, inputs=[count])

        with wp.ScopedCapture(force_module_load=False) as capture:
            wp.capture_while(count, while_body=body)

        for iterations in (4, 7):
            total.zero_()
            count.fill_(iterations)
            wp.capture_launch(capture.graph)
            test.assertEqual(total.numpy()[0], 2 * iterations)


def test_alloc_in_body_across_captures(test, device):
    """Bodies may allocate, and conditions may be allocated during capture.

    Capture allocations on HIP are graph-retained device memory, so arrays allocated in one
    segment stay valid in later segments and across replays, also after other graphs are captured.
    """
    with wp.ScopedDevice(device):
        wp.load_module(device=device)
        total = wp.zeros(1, dtype=wp.int32)

        def body(cond):
            tmp = wp.zeros(1, dtype=wp.int32)
            wp.launch(add_kernel, dim=1, inputs=[tmp, 2])
            wp.launch(accumulate_kernel, dim=1, inputs=[tmp, total])
            wp.launch(decrement_kernel, dim=1, inputs=[cond])

        with wp.ScopedCapture(force_module_load=False) as capture:
            cond = wp.full(1, value=3, dtype=wp.int32)
            wp.capture_while(cond, while_body=body, cond=cond)

        for replay in range(3):
            # capturing other graphs must not invalidate the allocations of this one
            with wp.ScopedCapture(force_module_load=False):
                wp.zeros(4, dtype=wp.int32)
            wp.capture_launch(capture.graph)
            test.assertEqual(total.numpy()[0], 6 * (replay + 1))


def test_wait_modes(test, device):
    for mode in ("spin", "sync"):
        with test.subTest(mode=mode):
            _check_wait_mode(test, device, mode)


def _check_wait_mode(test, device, mode):
    with wp.ScopedDevice(device), _ConfigOverride(hip_conditional_graph_wait=mode):
        a = wp.zeros(1, dtype=wp.int32)
        count = wp.zeros(1, dtype=wp.int32)
        wp.load_module(device=device)

        def body():
            launch_add(a, 1)
            wp.launch(decrement_kernel, dim=1, inputs=[count])

        with wp.ScopedCapture(force_module_load=False) as capture:
            wp.capture_while(count, while_body=body)

        for iterations in (0, 3, 50):
            a.zero_()
            count.fill_(iterations)
            wp.capture_launch(capture.graph)
            test.assertEqual(a.numpy()[0], iterations)


def test_emulation_disabled(test, device):
    with wp.ScopedDevice(device), _ConfigOverride(hip_conditional_graph_emulation=False):
        a = wp.zeros(1, dtype=wp.int32)
        cond = wp.ones(1, dtype=wp.int32)
        wp.load_module(device=device)

        # without capture, conditionals still run eagerly
        wp.capture_if(cond, on_true=lambda: launch_add(a, 1))
        test.assertEqual(a.numpy()[0], 1)

        with test.assertRaises(RuntimeError):
            with wp.ScopedCapture(force_module_load=False):
                wp.capture_if(cond, on_true=lambda: launch_add(a, 1))


def test_other_stream(test, device):
    with wp.ScopedDevice(device):
        stream = wp.Stream(device)
        a = wp.zeros(1, dtype=wp.int32)
        count = wp.zeros(1, dtype=wp.int32)
        wp.load_module(device=device)

        def body():
            launch_add(a, 1)
            wp.launch(decrement_kernel, dim=1, inputs=[count])

        with wp.ScopedStream(stream):
            with wp.ScopedCapture(force_module_load=False) as capture:
                wp.capture_while(count, while_body=body)

        # replay on a different stream than the capture stream
        count.fill_(4)
        wp.synchronize_device(device)
        wp.capture_launch(capture.graph)
        test.assertEqual(a.numpy()[0], 4)

        a.zero_()
        count.fill_(2)
        wp.synchronize_device(device)
        wp.capture_launch(capture.graph, stream=stream)
        wp.synchronize_stream(stream)
        test.assertEqual(a.numpy()[0], 2)


def test_debug_dot_print(test, device):
    with wp.ScopedDevice(device):
        a = wp.zeros(1, dtype=wp.int32)
        cond = wp.ones(1, dtype=wp.int32)
        wp.load_module(device=device)

        with wp.ScopedCapture(force_module_load=False) as capture:
            launch_add(a, 1)
            wp.capture_if(cond, on_true=lambda: launch_add(a, 2), on_false=lambda: launch_add(a, 3))
            launch_add(a, 4)

        with tempfile.TemporaryDirectory() as temp_dir:
            wp.capture_debug_dot_print(capture.graph, os.path.join(temp_dir, "program.dot"))
            files = sorted(os.listdir(temp_dir))
            test.assertGreaterEqual(len(files), 3)
            test.assertTrue(all(f.startswith("program.seg") and f.endswith(".dot") for f in files))


def test_capture_save_rejected(test, device):
    with wp.ScopedDevice(device):
        cond = wp.ones(1, dtype=wp.int32)
        a = wp.zeros(1, dtype=wp.int32)
        wp.load_module(device=device)
        with wp.ScopedCapture(force_module_load=False) as capture:
            wp.capture_if(cond, on_true=lambda: launch_add(a, 1))
        with tempfile.TemporaryDirectory() as temp_dir, test.assertRaises(RuntimeError):
            wp.capture_save(capture.graph, os.path.join(temp_dir, "graph"))


# ---------------------------------------------------------------------------------------------
# microbenchmark
# ---------------------------------------------------------------------------------------------


def _bench_body(a, count):
    wp.launch(add_kernel, dim=1, inputs=[a, 1])
    wp.launch(decrement_kernel, dim=1, inputs=[count])


def _time_call(fn, device, repeats):
    best = float("inf")
    for _ in range(repeats):
        wp.synchronize_device(device)
        start = time.perf_counter()
        fn()
        wp.synchronize_device(device)
        best = min(best, time.perf_counter() - start)
    return best


def benchmark_conditional_overhead(device=None, iterations=1000, repeats=5, wait_modes=("sync", "spin")):
    """Measure the host overhead of emulated ``capture_while`` iterations on a HIP device.

    Each loop iteration launches two tiny kernels. The per-iteration overhead of the emulation is
    reported relative to a single graph containing the same number of unrolled iterations.

    Returns:
        A dict with microseconds per iteration for each variant.
    """
    device = wp.get_device(device)
    results = {}
    with wp.ScopedDevice(device):
        a = wp.zeros(1, dtype=wp.int32)
        count = wp.zeros(1, dtype=wp.int32)
        wp.load_module(device=device)

        with wp.ScopedCapture(force_module_load=False) as unrolled:
            for _ in range(iterations):
                _bench_body(a, count)
        wp.capture_launch(unrolled.graph)  # instantiate outside the timed region
        t_unrolled = _time_call(lambda: wp.capture_launch(unrolled.graph), device, repeats)
        results["unrolled_graph_us_per_iter"] = t_unrolled / iterations * 1e6

        with wp.ScopedCapture(force_module_load=False) as looped:
            wp.capture_while(count, while_body=_bench_body, a=a, count=count)

        def run_looped():
            count.fill_(iterations)
            wp.capture_launch(looped.graph)

        saved_wait = wp.config.hip_conditional_graph_wait
        try:
            for mode in wait_modes:
                wp.config.hip_conditional_graph_wait = mode
                run_looped()  # warm up (segment instantiation, spin buffers)
                t_loop = _time_call(run_looped, device, repeats)
                results[f"emulated_while_{mode}_us_per_iter"] = t_loop / iterations * 1e6
                results[f"emulated_while_{mode}_overhead_us_per_iter"] = (t_loop - t_unrolled) / iterations * 1e6
        finally:
            wp.config.hip_conditional_graph_wait = saved_wait

        def run_eager():
            count.fill_(iterations)
            wp.capture_while(count, while_body=_bench_body, a=a, count=count)

        run_eager()
        t_eager = _time_call(run_eager, device, repeats)
        results["eager_while_us_per_iter"] = t_eager / iterations * 1e6

    return results


def _launch_budget(device):
    from warp._src import context  # noqa: PLC0415

    return context._hip_launch_budgets.get(device.stream.cuda_stream)


def test_launch_budget_bounds(test, device):
    # graphs smaller than, close to and larger than the in-flight node budget, replayed back to
    # back: the nodes in flight must stay within the budget and every replay must take effect
    limit = 64
    with wp.ScopedDevice(device), _ConfigOverride(hip_graph_max_inflight_nodes=limit):
        a = wp.zeros(4, dtype=wp.int32)
        wp.load_module(device=device)
        graphs = []
        for nodes in (1, 5, 40, 100):
            with wp.ScopedCapture(force_module_load=False) as capture:
                for _ in range(nodes):
                    launch_add(a, 1)
            graphs.append((nodes, capture.graph))

        wp.synchronize_device(device)
        expected = 0
        for i in range(300):
            nodes, graph = graphs[(i * 7) % len(graphs)]
            wp.capture_launch(graph)
            expected += nodes
            budget = _launch_budget(device)
            test.assertLessEqual(budget.total, max(limit, nodes))
            test.assertEqual(budget.total, sum(n for _, n in budget.events) + budget.uncovered)
        np.testing.assert_array_equal(a.numpy(), [expected] * 4)


def test_launch_budget_drained_by_condition(test, device):
    # reading a published condition means the stream has drained: the budget restarts from zero
    with wp.ScopedDevice(device), _ConfigOverride(hip_graph_max_inflight_nodes=16):
        a = wp.zeros(3, dtype=wp.int32)
        count = wp.zeros(1, dtype=wp.int32)
        wp.load_module(device=device)

        def body():
            for _ in range(6):
                launch_add(a, 1)
            wp.launch(decrement_kernel, dim=1, inputs=[count])

        with wp.ScopedCapture(force_module_load=False) as capture:
            wp.capture_while(count, while_body=body)

        for iterations in (0, 1, 9, 40):
            a.zero_()
            count.fill_(iterations)
            wp.capture_launch(capture.graph)
            # the program ends with the read of the final (false) condition
            test.assertEqual(_launch_budget(device).total, 0)
            np.testing.assert_array_equal(a.numpy(), [6 * iterations] * 3)


def test_benchmark_smoke(test, device):
    results = benchmark_conditional_overhead(device, iterations=20, repeats=1)
    for key, value in results.items():
        test.assertTrue(np.isfinite(value), key)
    if os.environ.get("WARP_HIP_CONDITIONAL_BENCH"):
        print({key: round(value, 2) for key, value in results.items()})


hip_devices = [d for d in get_cuda_test_devices() if d.is_hip]


class TestHipConditionalEmulation(unittest.TestCase):
    pass


for _name, _func in (
    ("test_reports_supported", test_reports_supported),
    ("test_plain_graph_unchanged", test_plain_graph_unchanged),
    ("test_if_else", test_if_else),
    ("test_if_only_and_else_only", test_if_only_and_else_only),
    ("test_while", test_while),
    ("test_nested_matches_eager", test_nested_matches_eager),
    ("test_if_inside_while_body_graph", test_if_inside_while_body_graph),
    ("test_plain_graph_bodies", test_plain_graph_bodies),
    ("test_graph_in_graph_splice", test_graph_in_graph_splice),
    ("test_kwargs_forwarded", test_kwargs_forwarded),
    ("test_body_error_restores_capture", test_body_error_restores_capture),
    ("test_alloc_in_body", test_alloc_in_body),
    ("test_alloc_in_body_across_captures", test_alloc_in_body_across_captures),
    ("test_wait_modes", test_wait_modes),
    ("test_emulation_disabled", test_emulation_disabled),
    ("test_other_stream", test_other_stream),
    ("test_debug_dot_print", test_debug_dot_print),
    ("test_capture_save_rejected", test_capture_save_rejected),
    ("test_launch_budget_bounds", test_launch_budget_bounds),
    ("test_launch_budget_drained_by_condition", test_launch_budget_drained_by_condition),
    ("test_benchmark_smoke", test_benchmark_smoke),
):
    add_function_test(TestHipConditionalEmulation, _name, _func, devices=hip_devices)


if __name__ == "__main__":
    if "--bench" in sys.argv:
        args = [arg for arg in sys.argv[1:] if arg != "--bench"]
        bench_iterations = int(args[0]) if args else 1000
        wp.init()
        for hip_device in hip_devices:
            print(hip_device, benchmark_conditional_overhead(hip_device, iterations=bench_iterations))
    else:
        wp.clear_kernel_cache()
        unittest.main(verbosity=2)
