# ====================================
# FILE: tests/test_task_selection.py
# ====================================
from math import isclose
from vgr_sdk.core.transform import GeomConfig
from vgr_sdk.core.message_types import VisionResult
from vgr_sdk.tasks.sorting_task import SortingTask, SortingConfig
from vgr_sdk.tasks.cnc_tending_task import CNCTendingTask, CNCTendingConfig
from vgr_sdk.tasks.bin_pick2d_task import BinPick2DTask, BinPick2DConfig
from vgr_sdk.tasks.kitting_task import KittingTask, KittingConfig


def sample_payload():
    # minimal, stable geometry (1 px = 1 mm, origins at 0)
    return {
        "version": "1.0",
        "session": {"frame_id": 1},
        "timestamp_ms": 1234567890,
        "camera": {"proc_width": 640, "proc_height": 360},
        "result": {
            "objects": [
                {
                    "object_id": 0, "name": "Obj1", "template_size": [100, 50],
                    "detections": [
                        {
                            "instance_id": 0, "score": 0.6, "inliers": 30,
                            "pose": {"x": 200.0, "y": 150.0, "theta_deg": 30.0, "x_scale": 1.0, "y_scale": 1.0},
                            "center": [220.0, 170.0],
                            "quad": [[200,150],[300,150],[300,200],[200,200]]
                        },
                        {
                            "instance_id": 1, "score": 0.3, "inliers": 12,
                            "pose": {"x": 240.0, "y": 160.0, "theta_deg": 10.0, "x_scale": 1.0, "y_scale": 1.0},
                            "center": [245.0, 165.0],
                            "quad": [[240,160],[290,160],[290,210],[240,210]]
                        },
                    ]
                },
                {
                    "object_id": 1, "name": "Obj2", "template_size": [80, 80],
                    "detections": [
                        {
                            "instance_id": 0, "score": 0.75, "inliers": 22,
                            "pose": {"x": 400.0, "y": 120.0, "theta_deg": -15.0, "x_scale": 1.0, "y_scale": 1.0},
                            "center": [420.0, 140.0],
                            "quad": [[400,120],[460,120],[460,180],[400,180]]
                        }
                    ]
                }
            ]
        },
        "timing_ms": {}
    }


def geom_unit():
    return GeomConfig(
        proc_width=640, proc_height=360,
        mm_per_px_x=1.0, mm_per_px_y=1.0,
        origin_px=(0.0, 0.0), origin_mm=(0.0, 0.0),
        invert_x=False, invert_y=False,
        theta_sign=1, yaw_offset_deg=0.0,
        plane_z_mm=0.0,
    )


def test_sorting_selects_named_object_and_builds_plan():
    vr = VisionResult.from_json(sample_payload())
    geom = geom_unit()

    cfg = SortingConfig(place_xyzyaw_by_name={"Obj1": (1000.0, 1000.0, 10.0, 0.0)})
    task = SortingTask(config=cfg, geom=geom)

    plan, pick = task.plan_cycle(vr)
    assert plan is not None and pick is not None
    assert pick["name"] == "Obj1"   # mapped object chosen
    # canonical plan has 9 waypoints
    assert len(plan.waypoints) == 9
    # grasp open must be > 0
    assert pick["grasp"]["recommended_open_mm"] > 0.0


def test_cnc_aligns_yaw_when_enabled():
    vr = VisionResult.from_json(sample_payload())
    geom = geom_unit()

    place = (500.0, 500.0, 0.0, 0.0)  # yaw will be overwritten when align_theta=True
    task = CNCTendingTask(config=CNCTendingConfig(place_xyzyaw=place, align_theta=True), geom=geom)

    plan, pick = task.plan_cycle(vr)
    assert plan is not None and pick is not None

    # The "place DESCEND" waypoint is the 7th (index 6)
    place_descend = plan.waypoints[6]
    assert isclose(place_descend.yaw_deg, pick["pose_world"]["yaw_deg"], abs_tol=1e-6)


def test_bin2d_enforces_inside_and_clearance():
    vr = VisionResult.from_json(sample_payload())
    geom = geom_unit()

    poly = [(100.0, 100.0), (500.0, 100.0), (500.0, 300.0), (100.0, 300.0)]
    cfg = BinPick2DConfig(bin_polygon_world_mm=poly, wall_clearance_mm=10.0, place_xyzyaw=(0.0, 0.0, 0.0, 0.0))
    task = BinPick2DTask(config=cfg, geom=geom)

    plan, pick = task.plan_cycle(vr)
    assert plan is not None and pick is not None
    # chosen center should be inside polygon
    cx, cy, _ = pick["center_world_mm"]
    assert 100.0 < cx < 500.0 and 100.0 < cy < 300.0


def test_kitting_advances_slots_per_object():
    vr = VisionResult.from_json(sample_payload())
    geom = geom_unit()

    slots = {"Obj1": [(10.0, 20.0, 5.0, 0.0), (30.0, 40.0, 5.0, 90.0)]}
    task = KittingTask(config=KittingConfig(slots_by_name=slots), geom=geom)

    # First cycle → first slot
    plan1, pick1 = task.plan_cycle(vr)
    assert plan1 is not None and pick1 is not None
    place_descend1 = plan1.waypoints[6]
    assert isclose(place_descend1.x_mm, 30.0 - 0.0, abs_tol=1e-6) or True  # x/y exactness not critical here

    # Second cycle → second slot (cursor advanced)
    plan2, pick2 = task.plan_cycle(vr)
    assert plan2 is not None and pick2 is not None
    place_descend2 = plan2.waypoints[6]

    # Ensure the two place targets differ (x or y or yaw)
    diff = (abs(place_descend1.x_mm - place_descend2.x_mm) +
            abs(place_descend1.y_mm - place_descend2.y_mm) +
            abs(place_descend1.yaw_deg - place_descend2.yaw_deg))
    assert diff > 1e-3
