import numpy as np

import medrs
from medrs.cli import main

from conftest import OBLIQUE, ramp


def test_info_and_convert(tmp_path, capsys):
    data = ramp((6, 5, 4), np.int16)
    src = tmp_path / "scan.nii.gz"
    medrs.NiftiImage(data, OBLIQUE).save(src)

    assert main(["info", str(src)]) == 0
    out = capsys.readouterr().out
    assert "(6, 5, 4)" in out
    assert "int16" in out

    assert main(["convert", str(src), "--to", "jvol", "-o", str(tmp_path / "out")]) == 0
    np.testing.assert_array_equal(medrs.load(tmp_path / "out" / "scan.jvol").to_numpy(), data)
    assert main(["convert", str(src), "--to", "mgzip", "--suffix", "_mgz"]) == 0
    np.testing.assert_array_equal(medrs.load(tmp_path / "scan_mgz.nii.gz").to_numpy(), data)

    assert main(["convert", str(src), "--to", "nii.gz"]) == 1
    assert "overwrite" in capsys.readouterr().err
