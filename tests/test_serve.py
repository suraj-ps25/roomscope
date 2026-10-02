import zipfile
from pathlib import Path

from roomscope.serve import capture_root, safe_relative, unpack


def test_upload_paths_stay_inside_the_job():
    assert safe_relative("living/c00a/depth/000001.png") == Path("living/c00a/depth/000001.png")
    assert safe_relative("clip.mov") == Path("clip.mov")
    for escape in ("../evil.txt", "a/../../evil.txt", "/etc/passwd", "C:/evil.txt", "", "./"):
        assert safe_relative(escape) is None


def test_a_zipped_export_becomes_its_folder(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    with zipfile.ZipFile(capture / "export.zip", "w") as bundle:
        bundle.writestr("c00a170fe1/odometry.csv", "timestamp\n")
        bundle.writestr("__MACOSX/c00a170fe1/._odometry.csv", "")
        bundle.writestr("../evil.txt", "x")
    unpack(capture)
    assert (capture / "c00a170fe1" / "odometry.csv").exists()
    assert not (capture / "export.zip").exists()
    assert not (capture / "__MACOSX").exists() and not (tmp_path / "evil.txt").exists()
    assert capture_root(capture) == capture / "c00a170fe1"
