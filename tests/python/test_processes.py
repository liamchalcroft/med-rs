"""medrs in forked worker processes (as used by PyTorch's DataLoader)."""

import multiprocessing
import sys

import numpy as np
import pytest

import medrs

from conftest import ramp


def _child(queue, path):
    image = medrs.load(path)
    queue.put(float(image.z_normalize().resample_to_shape((32, 32, 32)).to_numpy().sum()))


@pytest.mark.skipif(sys.platform != "linux", reason="fork start method")
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
