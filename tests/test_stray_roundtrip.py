from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from roomscope.geometry.cloud import frame_points
from roomscope.io.detect import detect_tier, resolve_capture
from roomscope.io.stray import read_stray
from roomscope.sim.scene import build_mesh, load_scene
from roomscope.sim.writers import write_lidar_tier

SCENE = Path(__file__).resolve().parents[1] / "benchmark" / "sim" / "flat_a.yaml"


@pytest.fixture(scope="module")
def capture(tmp_path_factory):
    out = tmp_path_factory.mktemp("stray")
    write_lidar_tier(SCENE, out, route=["hallway"], fps=1.0, rgb_size=(320, 240), drift_level=0.0)
    return out


def test_reads_every_frame(capture):
    bundle = read_stray(capture)
    assert bundle.tier == "lidar"
    assert len(bundle.frames) == len(list((capture / "depth").glob("*.png")))
    assert bundle.frames[0].depth_size == (256, 192)
    assert bundle.frames[0].image_size == (320, 240)


def test_points_land_on_true_surfaces(capture):
    """With zero drift, back-projected depth must sit on the scene geometry. A wrong axis
    flip or intrinsics scale puts points centimetres to metres off the walls."""
    bundle = read_stray(capture)
    mesh = build_mesh(load_scene(SCENE))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(mesh.vertices.astype(np.float32)),
                        o3d.core.Tensor(mesh.triangles.astype(np.uint32)))
    true_poses = np.load(capture / "ground_truth_poses.npy")
    for frame in bundle.frames[::3]:
        assert np.allclose(frame.pose, true_poses[frame.index], atol=1e-4)
        cloud = frame_points(frame, frame.pose, min_confidence=2, stride=4)
        distance = scene.compute_distance(o3d.core.Tensor(cloud.points.astype(np.float32))).numpy()
        assert np.median(distance) < 0.01, f"frame {frame.index}: median {np.median(distance):.3f} m off-surface"


def test_rgb_decodes(capture):
    bundle = read_stray(capture)
    bundle.warm_rgb(bundle.frames[:3])
    rgb = bundle.frames[1].rgb()
    assert rgb.shape == (240, 320, 3)
    assert rgb.mean() > 20


def test_recording_inside_a_wrapper_folder(capture, tmp_path):
    wrapper = tmp_path / "living_room"
    wrapper.mkdir()
    (wrapper / "c00a170fe1").symlink_to(capture, target_is_directory=True)
    assert resolve_capture(wrapper) == wrapper / "c00a170fe1"
    assert detect_tier(resolve_capture(wrapper)) == "lidar"
