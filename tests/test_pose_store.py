# ============================
# FILE: tests/test_pose_store.py
# ============================
import time
from vgr_sdk.core.pose_store import PoseStore


def test_pose_store_crud(tmp_path):
    poses_path = tmp_path / "recorded_poses.yaml"

    # fresh store
    store = PoseStore(str(poses_path))
    assert store.names() == []

    # add a pose
    home_joints = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    pe = store.set("home", home_joints, note="initial")
    assert pe.name == "home"
    assert store.exists("home")
    assert store.get("home").joints == home_joints

    # save and reload
    store.save()
    t_saved = poses_path.stat().st_mtime
    store2 = PoseStore(str(poses_path))
    assert store2.exists("home")
    assert store2.get("home").joints == home_joints

    # rename
    ok = store2.rename("home", "home2")
    assert ok
    assert not store2.exists("home")
    assert store2.exists("home2")
    store2.save()

    # delete
    ok = store2.delete("home2")
    assert ok
    assert not store2.exists("home2")
    store2.save()

    # file actually updated
    assert poses_path.exists()
    assert poses_path.stat().st_mtime >= t_saved
