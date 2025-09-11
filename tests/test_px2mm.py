# =========================
# FILE: tests/test_px2mm.py
# =========================
from math import isclose
from vgr_sdk.core.transform import GeomConfig, px_to_world_xy, angle_img_to_world, detection_img_to_world


def test_px_to_world_with_inverts_and_offsets():
    cfg = GeomConfig(
        proc_width=640, proc_height=360,
        mm_per_px_x=0.5, mm_per_px_y=1.0,
        origin_px=(100.0, 50.0), origin_mm=(1000.0, 500.0),
        invert_x=False, invert_y=True,
        theta_sign=-1, yaw_offset_deg=10.0,
        plane_z_mm=0.0,
    )

    # origin maps to origin
    X0, Y0 = px_to_world_xy(100.0, 50.0, cfg)
    assert isclose(X0, 1000.0, rel_tol=0, abs_tol=1e-6)
    assert isclose(Y0, 500.0,  rel_tol=0, abs_tol=1e-6)

    # +10px X, +20px Y -> X increases by 10*0.5, Y *decreases* by 20*1.0 due to invert_y
    X1, Y1 = px_to_world_xy(110.0, 70.0, cfg)
    assert isclose(X1, 1005.0, rel_tol=0, abs_tol=1e-6)
    assert isclose(Y1, 480.0, rel_tol=0, abs_tol=1e-6)

    # angle: theta_world = theta_sign * theta_img + yaw_offset
    a = angle_img_to_world(20.0, cfg)
    assert isclose(a, -10.0, rel_tol=0, abs_tol=1e-6)


def test_detection_img_to_world_basic():
    cfg = GeomConfig(
        proc_width=640, proc_height=360,
        mm_per_px_x=0.5, mm_per_px_y=1.0,
        origin_px=(100.0, 50.0), origin_mm=(1000.0, 500.0),
        invert_x=False, invert_y=True,
        theta_sign=-1, yaw_offset_deg=10.0,
        plane_z_mm=0.0,
    )

    det = {
        "name": "Obj1",
        "template_size": [96, 96],
        "pose": {"x": 110.0, "y": 70.0, "theta_deg": 20.0, "x_scale": 1.0, "y_scale": 1.0},
        "center": [110.0, 70.0],
        "quad": [[100.0, 60.0], [120.0, 60.0], [120.0, 80.0], [100.0, 80.0]],
        "score": 0.6, "inliers": 20,
    }

    dw = detection_img_to_world(det, cfg)

    # center -> world
    cw = dw["center_world_mm"]
    assert len(cw) == 3
    assert isclose(cw[0], 1005.0, abs_tol=1e-6)  # X
    assert isclose(cw[1], 480.0, abs_tol=1e-6)   # Y
    assert isclose(cw[2], 0.0,   abs_tol=1e-6)   # Z fixed plane

    # pose_world
    pw = dw["pose_world"]
    assert isclose(pw["x_mm"], 1005.0, abs_tol=1e-6)
    assert isclose(pw["y_mm"], 480.0,  abs_tol=1e-6)
    assert isclose(pw["z_mm"], 0.0,    abs_tol=1e-6)
    assert isclose(pw["yaw_deg"], -10.0, abs_tol=1e-6)

    # quad -> world has 4 points with Z
    qw = dw["quad_world_mm"]
    assert len(qw) == 4 and all(len(p) == 3 for p in qw)
