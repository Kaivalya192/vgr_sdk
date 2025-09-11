## config/

* **vision\_geom.yaml**

  * Purpose: pixel→mm & yaw mapping for a fixed Z plane.
  * Used by: `nodes/vgr_manager.py` → `GeomConfig`.
  * Tweak: `mm_per_px_*`, `origin_*`, `invert_*`, `theta_sign`, `yaw_offset_deg`, `plane_z_mm`.

* **robot.yaml**

  * Purpose: robot/gripper presets (TCP offset, speeds, stroke).
  * Used by: your executor (not in repo code paths yet).
  * Tweak: `gripper.stroke_*`, `heights_mm.*`, `speed_presets.*`.

* **topics.yaml**

  * Purpose: ROS topic names (convenience).
  * Used by: your launch/param wiring (not mandatory).

* **tasks/sorting.yaml**

  * Purpose: object `name` → drop `place_xyzyaw`.
  * Used by: `SortingTask` via `VGRManagerNode._build_task`.

* **tasks/cnc\_tending.yaml**

  * Purpose: fixture `place_xyzyaw`, yaw/scale gates.
  * Used by: `CNCTendingTask`.

* **tasks/bin\_picking\_2d.yaml**

  * Purpose: bin polygon (world mm), wall clearance, drop pose.
  * Used by: `BinPick2DTask`.

* **tasks/kitting.yaml**

  * Purpose: per-object slot lists (XYZYaw), ordered fill.
  * Used by: `KittingTask`.

---

## schemas/

* **vision\_result.schema.json**

  * Purpose: optional JSON Schema for incoming Vision payloads.
  * Used by: `core/vision_ingest.py` (if `~schema_path` set).
  * Tweak: extend when your Vision JSON format evolves.

---

## scripts/ (ROS entrypoints)

* **vision\_bridge\_node.py**

  * Role: UDP → `/vgr/vision_result` (`std_msgs/String` JSON).
  * Uses: `nodes/vision_bridge.VisionBridgeNode`.

* **vgr\_manager\_node.py**

  * Role: `/vgr/vision_result` → task plan `/vgr/plan` (JSON).
  * Uses: `nodes/vgr_manager.VGRManagerNode` (loads task + geom).

* **pose\_recorder\_node.py**

  * Role: save current `/joint_states` to `poses/recorded_poses.yaml` on trigger topic.
  * Uses: `core/pose_store.PoseStore`.

* **gripper\_node.py**

  * Role: gripper control topics (`~open_mm`, `~close_mm`, `~close_force_n`, `~cmd`) + `~status`.
  * Backend: in-file `SimBackend` (dummy). Swap with your driver via param `~driver_module`.

* **trigger\_client.py**

  * Role: send `{"cmd":"TRIGGER"}` via UDP/HTTP to your Vision app.

---

## src/vgr\_sdk/core/

* **message\_types.py**

  * Classes: `VisionResult`, `ObjectDetections`, `Detection`, `Pose`.
  * Role: tolerant JSON→Python model; `VisionResult.from_json()`, `flat_detections()`.

* **transform.py**

  * Class: `GeomConfig`.
  * Fns: `px_to_world_xy`, `extent_px_to_mm`, `angle_img_to_world`, `quad_px_to_world`, `detection_img_to_world`.
  * Role: image-space → world (mm/deg) using `vision_geom.yaml`.

* **filters.py**

  * Fns: `gate_detections`, `sort_by_quality`, `dedupe_by_center_px/mm`.
  * Role: quality gates + duplicate suppression.

* **gripper.py**

  * Fns: `quad_minor_major_mm`, `recommend_opening_mm`, `grasp_axis_deg_from_quad_world`, `symmetric_grasp_points_world`.
  * Role: compute jaw opening, grasp axis, grasp points from geometry.

* **planner.py**

  * Classes: `Waypoint`, `Plan`, `PickPlaceParams`.
  * Fn: `build_pick_place_plan(pick_xyzyaw, place_xyzyaw, ...)`.
  * Role: creates tagged Cartesian waypoint sequences (`APPROACH`, `DESCEND`, `GRASP`, …).

* **pose\_store.py**

  * Classes: `PoseStore`, `PoseEntry`.
  * Role: load/save/CRUD joint-space poses (YAML).

* **vision\_ingest.py**

  * Classes: `VisionUDPIngest`, `VisionJSONValidator`.
  * Role: UDP receive + (optional) schema validate + JSON parse (lenient).

* **robot\_io.py**

  * Protocols: `RobotInterface`, `GripperInterface`.
  * Dummies: `DummyRobot`, `DummyGripper` (for tests).
  * Role: reference interfaces for your real robot/gripper drivers.

* **utils.py**

  * Fns: `clamp`, `deg2rad`, `rad2deg`, `ang_norm_deg`, `polygon_area`, `poly_centroid`, `pca_axis`.
  * Class: `SimpleTimer`.

---

## src/vgr\_sdk/nodes/

* **vision\_bridge.py**

  * Class: `VisionBridgeNode` (paramized UDP bind/port/topic).
  * Role: internal logic for `scripts/vision_bridge_node.py`.

* **vgr\_manager.py**

  * Class: `VGRManagerNode`.
  * Role: loads `GeomConfig` + Task from YAML, subscribes results, publishes plan.

---

## src/vgr\_sdk/tasks/

* **base\_task.py**

  * Classes: `BaseTask`, `GripperConfig`, `QualityGates`.
  * Role: common pipeline: flatten → world-transform → enrich (grasp hints) → plan builder.

* **sorting\_task.py**

  * Class: `SortingTask`, `SortingConfig`.
  * Role: choose detection by object `name`→drop mapping; build plan.

* **cnc\_tending\_task.py**

  * Class: `CNCTendingTask`, `CNCTendingConfig`.
  * Role: yaw/scale gating; optional align place yaw to detected yaw; build plan.

* **bin\_pick2d\_task.py**

  * Class: `BinPick2DTask`, `BinPick2DConfig`.
  * Role: keep detections inside polygon with wall clearance; build plan.

* **kitting\_task.py**

  * Class: `KittingTask`, `KittingConfig`.
  * Role: per-object slot cursors; place in next slot; build plan.

---

## tests/

* **test\_pose\_store.py**

  * Verifies: set/save/load/rename/delete for `PoseStore` (atomic save).

* **test\_px2mm.py**

  * Verifies: px→mm transform, invert flags, angle mapping, detection→world conversion.

* **test\_task\_selection.py**

  * Verifies: task selection & plan generation across Sorting/CNC/Bin2D/Kitting with a sample payload.

---

## Where you **swap dummies** for real hardware

* **Gripper**: `scripts/gripper_node.py`

  * Replace `SimBackend` with your module via param:
    `_driver_module:=your_pkg.your_backend` (must expose `Backend` with `open_mm/close_mm/close_force/get_status`).

* **Robot execution**: add your **executor node** (not included here)

  * Subscribe `/vgr/plan`, send Cartesian moves to your controller, act on tags `GRASP/RELEASE` (publish to `/vgr_gripper/...`).
  * Use `planner.PickPlaceParams` & `config/robot.yaml` for heights/speeds; use `pose_store` for joint waypoints if needed.
