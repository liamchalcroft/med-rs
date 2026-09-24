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


def _jvol_sum(path):
    return float(medrs.load(path).to_numpy().sum())


def _jvol_files(tmp_path):
    """Write chunked .jvol files, so encoding and decoding use the thread pool."""
    paths = []
    for i, quality in enumerate([80, None, 60, None]):
        path = tmp_path / f"vol{i}.jvol"
        image = medrs.NiftiImage(ramp((64, 64, 48)) + i)
        image.save(path, quality=quality, chunk_shape=(32, 32, 16))
        paths.append(path)
    return paths


@pytest.mark.skipif(sys.platform != "linux", reason="fork start method")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:os.fork:RuntimeWarning")
def test_forked_pool_loads_jvol_after_the_parent_did(tmp_path):
    paths = _jvol_files(tmp_path)
    expected = [_jvol_sum(p) for p in paths]  # the parent's thread pool is now running
    with multiprocessing.get_context("fork").Pool(2) as pool:
        # A deadlocked worker raises TimeoutError here instead of hanging.
        sums = pool.map_async(_jvol_sum, paths * 2).get(timeout=60)
    assert sums == expected * 2


@pytest.mark.skipif(sys.platform != "linux", reason="fork start method")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:os.fork:RuntimeWarning")
def test_dataloader_workers_load_jvol_after_the_parent_did(tmp_path):
    torch = pytest.importorskip("torch")
    paths = _jvol_files(tmp_path)
    expected = [_jvol_sum(p) for p in paths]

    class Volumes(torch.utils.data.Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            return medrs.load(paths[i]).to_torch().double().sum()

    loader = torch.utils.data.DataLoader(
        Volumes(), batch_size=None, num_workers=2, multiprocessing_context="fork", timeout=60
    )
    for _ in range(2):  # a second epoch forks new workers
        sums = [float(s) for s in loader]
        assert sums == pytest.approx(expected, rel=1e-6)


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
