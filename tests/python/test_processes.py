"""medrs in threads and forked worker processes (as used by PyTorch's DataLoader)."""

import multiprocessing
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import medrs

from conftest import ramp


def _child(queue, path):
    image = medrs.load(path)
    queue.put(float(image.z_normalize().resample_to_shape((32, 32, 32)).to_numpy().sum()))


@pytest.mark.skipif(sys.platform != "linux", reason="fork start method")
# Forking a multi-threaded parent is the point of this test.
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:os.fork:RuntimeWarning")  # JAX may be imported by other tests
def test_forked_children_can_run_parallel_work(tmp_path):
    image = medrs.NiftiImage(ramp((80, 80, 80)))
    image.z_normalize()  # the parent's thread pool is now running
    path = tmp_path / "vol.nii"
    image.save(path)
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    child = context.Process(target=_child, args=(queue, path))
    child.start()
    child.join(60)
    if child.is_alive():
        child.kill()
        pytest.fail("forked child hung")
    assert child.exitcode == 0
    assert np.isfinite(queue.get(timeout=5))


def test_thread_count_is_configurable():
    before = medrs.num_threads()
    medrs.set_num_threads(2)
    assert medrs.num_threads() == 2
    medrs.NiftiImage(ramp((70, 70, 70))).z_normalize()
    medrs.set_num_threads(0)
    assert medrs.num_threads() == before


def test_concurrent_threads_match_serial_results(tmp_path):
    paths = []
    for i in range(4):
        paths.append(tmp_path / f"vol{i}.nii.gz")
        medrs.NiftiImage(ramp((40, 48, 36)) * (i + 1)).save(paths[-1])
    shared = medrs.load(paths[0])

    def work(task):
        image = shared if task % 5 == 4 else medrs.load(paths[task % 4])
        return image.reorient("LPS").z_normalize().resample_to_shape((24, 24, 24)).to_numpy()

    tasks = range(40)
    expected = {task: work(task) for task in range(5)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(work, tasks))
    for task, result in zip(tasks, results, strict=True):
        key = 4 if task % 5 == 4 else task % 4
        np.testing.assert_array_equal(result, expected[key])


@pytest.mark.skipif(not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="free-threaded Python")
def test_import_keeps_the_gil_disabled():
    assert not sys._is_gil_enabled()
