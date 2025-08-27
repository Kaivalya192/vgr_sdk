---

# Core concepts (used by every task)

### vision\_geom.yaml — “how pixels turn into robot millimeters”

```yaml
proc_width: 640          # processed frame width from the Vision app
proc_height: 360         # processed frame height
mm_per_px_x: 0.25        # X scale: mm per pixel
mm_per_px_y: 0.25        # Y scale: mm per pixel
origin_px: [0.0, 0.0]    # pixel that maps to world origin (X0,Y0)
origin_mm: [0.0, 0.0]    # that origin’s world coordinates in mm
invert_x: false          # flip X if your robot frame is mirrored
invert_y: false          # flip Y if your robot frame is mirrored
theta_sign: 1            # +1 if image θ equals world yaw, -1 if mirrored
yaw_offset_deg: 0.0      # add a constant offset (deg) if needed
plane_z_mm: 0.0          # fixed Z height of the pick plane
```

* **mm\_per\_px\_x/y**: Measure a known object in pixels (from your Vision overlay) and in mm. `mm_per_px = known_mm / measured_px`.
* **origin\_px → origin\_mm**: Decide where “(0,0)” is on the table. Put a tape cross there, read its pixel position from the Vision overlay; set `origin_px` to that, and `origin_mm` to the robot’s world XY at that cross.
* **invert\_x/y**: If your world X increases to the left on screen, set that axis’s `invert_*: true`.
* **theta\_sign / yaw\_offset\_deg**: If rotations look mirrored or offset, flip sign and/or add an offset (e.g., +90°) until the published yaw matches your robot world yaw convention.

### Quality gates & deduplication (avoid junk and duplicates)

These defaults live in code (`BaseTask.QualityGates`):

* `min_inliers = 10` → minimum RANSAC inliers to accept a detection.
* `min_score = 0.25` → ratio of inliers to matches.
* `min_center_dist_px = 40` → minimum center-to-center distance (in pixels) to consider two detections different.

> If you want to tighten/loosen: raise/lower these in `BaseTask.QualityGates`. (Advanced: we can expose them via ROS params later.)

### What the manager publishes

* Topic: `/vgr/plan` (JSON)

  * `plan.waypoints[]`: tagged Cartesian waypoints (mm/deg) with tags like `APPROACH`, `DESCEND`, `GRASP`, `RELEASE`, etc.
  * `pick`: the chosen detection **enriched** with:

    * `pose_world` `{x_mm,y_mm,z_mm,yaw_deg}`
    * `center_world_mm` `[X,Y,Z]`
    * `quad_world_mm` `[[X,Y,Z]×4]` when available
    * `grasp` `{minor_extent_mm, major_extent_mm, recommended_open_mm, axis_deg, points_world}`

Your robot-side executor consumes `plan` and triggers gripper actions at `GRASP/RELEASE`.

---

# 1) Sorting task

### Config — `config/tasks/sorting.yaml`

```yaml
map:                     # object name → drop pose (world XYZYaw)
  Obj1: [200.0, 300.0, 20.0, 0.0]
  Obj2: [250.0, 300.0, 20.0, 90.0]
default: [220.0, 300.0, 20.0, 0.0]   # (optional) fallback if name not mapped
```

**What it does**

* From all detections, pick the highest-quality detection **whose `name` exists** in `map`.
* Build a standard pick→place plan to the mapped pose (or `default` if provided).

**How to tune**

* Change destination poses by editing `map`.
* Use `default` to always place “unknown” objects somewhere safe (or omit it to skip).

**Run (2 terminals)**

```bash
# (A) Vision bridge: UDP → /vgr/vision_result
rosrun vgr_sdk vision_bridge_node.py _port:=40001 _topic:=/vgr/vision_result

# (B) VGR manager: Sorting
rosrun vgr_sdk vgr_manager_node.py \
  _vision_geom:=config/vision_geom.yaml \
  _task_config:=config/tasks/sorting.yaml \
  _task:=sorting \
  _sub_result_topic:=/vgr/vision_result \
  _pub_plan_topic:=/vgr/plan
```

**Observe**

```bash
rostopic echo /vgr/plan   # see waypoints and grasp recommendation
```

---

# 2) CNC tending task

### Config — `config/tasks/cnc_tending.yaml`

```yaml
place_xyzyaw: [400.0, 250.0, 0.0, 0.0]   # machine/fixture insert pose
align_theta: true                         # align place yaw to detected yaw
theta_tolerance_deg: 7.5                  # reject if |Δyaw| > this (if align_theta=false, ignored)
scale_tolerance: 0.25                     # accept if x_scale and y_scale within ±25% of nominal
```

**What it does**

* Filters detections by **orientation** and **scale drift**, then selects best one.
* If `align_theta: true`, the placement **yaw** is set to detected yaw (good for fitted insertion).
* Otherwise, always insert with the configured `place_xyzyaw[3]`.

**How to tune**

* If parts are slightly rotated on the tray, keep `align_theta: true`.
* Tighten `theta_tolerance_deg` if insertion is sensitive.
* Tighten `scale_tolerance` if camera zoom/scaling should be very stable.

**Run**

```bash
rosrun vgr_sdk vgr_manager_node.py \
  _vision_geom:=config/vision_geom.yaml \
  _task_config:=config/tasks/cnc_tending.yaml \
  _task:=cnc \
  _sub_result_topic:=/vgr/vision_result \
  _pub_plan_topic:=/vgr/plan
```

---

# 3) 2D bin picking (top layer)

### Config — `config/tasks/bin_picking_2d.yaml`

```yaml
bin_polygon_world_mm:
  - [100.0, 100.0]
  - [300.0, 100.0]
  - [300.0, 300.0]
  - [100.0, 300.0]
wall_clearance_mm: 20.0         # min distance from detection center to any wall
place_xyzyaw: [350.0, 300.0, 20.0, 0.0]
```

**What it does**

* Keeps detections **inside** the polygon and at least `wall_clearance_mm` away from edges.
* Selects the best one and creates a pick→place plan.

**How to tune**

* Define polygon vertices **in world mm** (counterclockwise recommended).
* Increase `wall_clearance_mm` to avoid edge grasps or jaw collisions.

**Run**

```bash
rosrun vgr_sdk vgr_manager_node.py \
  _vision_geom:=config/vision_geom.yaml \
  _task_config:=config/tasks/bin_picking_2d.yaml \
  _task:=bin2d \
  _sub_result_topic:=/vgr/vision_result \
  _pub_plan_topic:=/vgr/plan
```

---

# 4) Kitting task

### Config — `config/tasks/kitting.yaml`

```yaml
slots_by_name:
  ObjA:
    - [100.0, 200.0, 10.0, 0.0]
    - [120.0, 200.0, 10.0, 0.0]
  ObjB:
    - [200.0, 200.0, 10.0, 90.0]
allow_fallback: false   # if true, reuse last slot when list is exhausted (demo mode)
```

**What it does**

* For each object **name**, it keeps a cursor to the **next slot**.
* Each time it sees that object, it places into the next slot in order.

**How to tune**

* Order the slots as you want them filled.
* Turn on `allow_fallback: true` for demos (keeps placing even after slots end).

**Run**

```bash
rosrun vgr_sdk vgr_manager_node.py \
  _vision_geom:=config/vision_geom.yaml \
  _task_config:=config/tasks/kitting.yaml \
  _task:=kitting \
  _sub_result_topic:=/vgr/vision_result \
  _pub_plan_topic:=/vgr/plan
```

---

# Recording reusable robot poses (joint-space waypoints)

Use this when you want to build job recipes and save poses for later (executor level).

### Node — `scripts/pose_recorder_node.py`

* Subscribes to `/joint_states`.
* Saves the **current joints** when you publish a name to `/vgr/pose_recorder/save`.

**Run**

```bash
rosrun vgr_sdk pose_recorder_node.py _poses_path:=poses/recorded_poses.yaml
```

**Save a pose**

```bash
# Move robot to the desired place; then:
rostopic pub -1 /vgr/pose_recorder/save std_msgs/String "home"
```

`poses/recorded_poses.yaml` will be updated atomically.

> Note: The current tasks use **cartesian place poses** from their YAML. Your executor can combine those with **recorded joint waypoints** (e.g., pre-place safe approach) when actually generating motion.

---

# Gripper usage

### Node — `scripts/gripper_node.py`

* Topics:

  * `~open_mm` (`Float64`): target opening (mm)
  * `~close_mm` (`Float64`): target closed width (mm)
  * `~close_force_n` (`Float64`): force close
  * `~cmd` (`String` JSON): `{"cmd":"open","open_mm":30}`, `{"cmd":"close","open_mm":20}`, or `{"cmd":"status"}`
  * `~status` (`String` JSON): periodic status (`opening_mm`, `object_detected`, etc)

**Run (sim backend)**

```bash
rosrun vgr_sdk gripper_node.py _stroke_min_mm:=5.0 _stroke_max_mm:=80.0
```

**Command examples**

```bash
# Open to 35 mm
rostopic pub -1 /vgr_gripper/open_mm std_msgs/Float64 35.0

# Close to 20 mm
rostopic pub -1 /vgr_gripper/close_mm std_msgs/Float64 20.0

# JSON command (alternative)
rostopic pub -1 /vgr_gripper/cmd std_msgs/String '{"cmd":"open","open_mm":30}'
```

> The plan’s `pick.grasp.recommended_open_mm` tells you what to send at `GRASP`. Your executor should publish that to `/vgr_gripper/close_mm` when it reaches the `GRASP` waypoint, and open at `RELEASE`.

---

# Example end-to-end (Sorting)

1. Fill `config/vision_geom.yaml` (px→mm, origin, Z).
2. Edit `config/tasks/sorting.yaml` (drop poses).
3. Start nodes:

```bash
rosrun vgr_sdk vision_bridge_node.py _port:=40001 _topic:=/vgr/vision_result
rosrun vgr_sdk vgr_manager_node.py _vision_geom:=config/vision_geom.yaml _task_config:=config/tasks/sorting.yaml _task:=sorting
rosrun vgr_sdk gripper_node.py
```

4. Verify `/vgr/plan`:

```bash
rostopic echo /vgr/plan
```

5. Your executor (your controller PC) subscribes `/vgr/plan`, drives robot waypoints linearly, and at:

   * `GRASP`: `close_mm` to the plan’s `pick.grasp.recommended_open_mm`
   * `RELEASE`: `open_mm` to a safe value

---

# Example end-to-end (CNC tending)

* Use `align_theta: true` to make the placement yaw equal to the part’s detected yaw.
* Tighten `theta_tolerance_deg` if insertion is tight; manager will skip parts outside tolerance.

Commands are the same as above; just choose `_task:=cnc` and the CNC config file.

---

# Example trigger workflow

Your Vision app already supports “Training” vs “Trigger” mode. In **Trigger** mode it sends one JSON **on demand**.

1. Start bridge + manager (any task).
2. When you want a snapshot, send a trigger packet to the Vision app:

```bash
# If you add a UDP listener on the Vision side (e.g., port 40011):
rosrun vgr_sdk trigger_client.py --mode udp --ip 127.0.0.1 --port 40011 --job-id demo1
```

3. The bridge republishes the single result; manager emits a single `/vgr/plan`.

---

# Where to tinker logic

* **Gating & dedupe**: `src/vgr_sdk/tasks/base_task.py` → `QualityGates`.
  Increase `min_center_dist_px` to reduce double picks of the same object.
* **Grasp sizing**: `src/vgr_sdk/core/gripper.py`

  * `recommend_opening_mm(minor + clearance → clamp to stroke)`
* **Waypoint shaping**: `src/vgr_sdk/core/planner.py`

  * `PickPlaceParams` (approach height, lift, speeds).
  * Path composition (add/remove waypoints, switch JOINT vs LINEAR).
* **World transform**: `src/vgr_sdk/core/transform.py`

  * Mapping from image `pose` → `pose_world` uses `vision_geom.yaml`.

---
