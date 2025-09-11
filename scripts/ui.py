#!/usr/bin/env python3
"""
Desktop Client UI (PyQt5) for RPi Vision Server

- Connects to ws://HOST:PORT from your headless server.
- Receives:
    << { "type":"hello" }
    << { "type":"frame", "jpeg_b64":..., "w":..., "h":..., ... }
    << { "type":"detections", "payload":{...}, "overlay_jpeg_b64": "..." }
    << { "type":"ack", "cmd":"...", "ok":true/false, "error":null|str }
    << { "type":"pong", "ts_ms":... }

- Sends:
    >> { "type":"set_mode", "mode":"training"|"trigger" }
    >> { "type":"trigger" }
    >> { "type":"set_proc_width", "width":640 }
    >> { "type":"set_publish_every", "n": 1 }
    >> { "type":"set_params", "params": { ... camera params ... } }
    >> { "type":"set_view", "flip_h":..., "flip_v":..., "rot_quadrant":... }
    >> { "type":"add_template_rect", "slot":i, "name":..., "rect":[x,y,w,h], "max_instances":N }
    >> { "type":"add_template_poly", "slot":i, "name":..., "points":[[x,y],...], "max_instances":N }
    >> { "type":"clear_template", "slot":i }
    >> { "type":"set_slot_state", "slot":i, "enabled":true, "max_instances":N }
"""

import sys, time, base64, json
from typing import Optional, Dict, List, Tuple
from PyQt5 import QtCore, QtGui, QtWidgets, QtWebSockets, QtNetwork

import numpy as np
import cv2  # only used to convert to RGB888 if needed (safe to remove if you prefer QImage only)

# ---- Bring in your existing video ROI helper (keeps aspect, selection tools)
from dexsdk.ui.video_label import VideoLabel


# ---------- Collapsible tray (same as your old UI) ----------
class CollapsibleTray(QtWidgets.QWidget):
    toggled = QtCore.pyqtSignal(bool)
    def __init__(self, title: str = "Camera Controls", parent=None):
        super().__init__(parent)
        self._body = QtWidgets.QWidget()
        self._body.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self._toggle = QtWidgets.QToolButton(text=title, checkable=True, checked=False)
        self._toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(QtCore.Qt.RightArrow)
        self._toggle.clicked.connect(self._on_toggle)

        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self._toggle)
        top.addStretch(1)

        self._wrap = QtWidgets.QWidget()
        wrap_l = QtWidgets.QVBoxLayout(self._wrap); wrap_l.setContentsMargins(0, 0, 0, 0)
        wrap_l.addWidget(self._body)

        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(top); lay.addWidget(self._wrap)
        self.set_collapsed(True)

    def set_body_layout(self, layout: QtWidgets.QLayout):
        self._body.setLayout(layout)

    def set_collapsed(self, collapsed: bool):
        self._wrap.setVisible(not collapsed)
        self._toggle.setChecked(not collapsed)
        self._toggle.setArrowType(QtCore.Qt.DownArrow if not collapsed else QtCore.Qt.RightArrow)
        self.toggled.emit(not collapsed)


    def _on_toggle(self):
        self.set_collapsed(self._wrap.isVisible())


# ---------- Camera Panel (client) : emits params to send to server ----------
class CameraPanel(QtWidgets.QWidget):
    paramsChanged = QtCore.pyqtSignal(dict)  # emits partial { ... } to merge under "params"
    viewChanged = QtCore.pyqtSignal(dict)    # emits {"flip_h":..,"flip_v":..,"rot_quadrant":..}

    def __init__(self, parent=None):
        super().__init__(parent)
        self._debounce = QtCore.QTimer(self)
        self._debounce.setInterval(150)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._emit_params)

        # Ranges (we don't have PiCam here; use safe fallbacks)
        exp_lo, exp_hi = 100, 33000
        gain_lo, gain_hi = 1.0, 16.0

        grid = QtWidgets.QGridLayout(self)
        grid.setHorizontalSpacing(10); grid.setVerticalSpacing(6)
        r = 0

        # AE / Exposure / Gain
        self.chk_ae = QtWidgets.QCheckBox("Auto Exposure"); self.chk_ae.setChecked(False)
        self.chk_ae.toggled.connect(self._on_any_change)
        grid.addWidget(self.chk_ae, r, 0, 1, 2); r += 1

        self.sp_exposure = QtWidgets.QSpinBox(); self.sp_exposure.setRange(exp_lo, exp_hi)
        self.sp_exposure.setSuffix(" µs"); self.sp_exposure.setSingleStep(100); self.sp_exposure.setValue(6000)
        self.sp_exposure.valueChanged.connect(self._on_any_change)
        grid.addWidget(QtWidgets.QLabel("Exposure"), r, 0); grid.addWidget(self.sp_exposure, r, 1); r += 1

        self.dsb_gain = QtWidgets.QDoubleSpinBox(); self.dsb_gain.setRange(gain_lo, gain_hi)
        self.dsb_gain.setDecimals(2); self.dsb_gain.setSingleStep(0.05); self.dsb_gain.setValue(2.0)
        self.dsb_gain.valueChanged.connect(self._on_any_change)
        grid.addWidget(QtWidgets.QLabel("Analogue Gain"), r, 0); grid.addWidget(self.dsb_gain, r, 1); r += 1

        # FPS
        self.dsb_fps = QtWidgets.QDoubleSpinBox(); self.dsb_fps.setRange(1.0, 120.0)
        self.dsb_fps.setDecimals(1); self.dsb_fps.setValue(30.0)
        self.dsb_fps.valueChanged.connect(self._on_any_change)
        grid.addWidget(QtWidgets.QLabel("Framerate (FPS)"), r, 0); grid.addWidget(self.dsb_fps, r, 1); r += 1

        # AWB
        self.cmb_awb = QtWidgets.QComboBox(); self.cmb_awb.addItems(["auto","tungsten","fluorescent","indoor","daylight","cloudy"])
        self.cmb_awb.currentTextChanged.connect(self._on_any_change)
        grid.addWidget(QtWidgets.QLabel("AWB Mode"), r, 0); grid.addWidget(self.cmb_awb, r, 1); r += 1

        self.dsb_awb_r = QtWidgets.QDoubleSpinBox(); self.dsb_awb_r.setRange(0.1, 8.0); self.dsb_awb_r.setSingleStep(0.05); self.dsb_awb_r.setValue(2.0)
        self.dsb_awb_b = QtWidgets.QDoubleSpinBox(); self.dsb_awb_b.setRange(0.1, 8.0); self.dsb_awb_b.setSingleStep(0.05); self.dsb_awb_b.setValue(2.0)
        wb_row = QtWidgets.QHBoxLayout(); wb_row.addWidget(QtWidgets.QLabel("AWB Gains R/B")); wb_row.addWidget(self.dsb_awb_r); wb_row.addWidget(self.dsb_awb_b)
        for w in (self.dsb_awb_r, self.dsb_awb_b):
            w.valueChanged.connect(self._on_any_change)
        grid.addWidget(QtWidgets.QLabel(""), r, 0); grid.addLayout(wb_row, r, 1); r += 1

        # Focus
        self.cmb_af = QtWidgets.QComboBox(); self.cmb_af.addItems(["auto","continuous","manual"])
        self.cmb_af.currentTextChanged.connect(self._on_any_change)
        grid.addWidget(QtWidgets.QLabel("AF Mode"), r, 0); grid.addWidget(self.cmb_af, r, 1); r += 1

        self.dsb_dioptre = QtWidgets.QDoubleSpinBox(); self.dsb_dioptre.setRange(0.0, 10.0)
        self.dsb_dioptre.setDecimals(2); self.dsb_dioptre.setSingleStep(0.05); self.dsb_dioptre.setValue(0.0)
        self.dsb_dioptre.valueChanged.connect(self._on_any_change)
        grid.addWidget(QtWidgets.QLabel("Lens (dioptres)"), r, 0); grid.addWidget(self.dsb_dioptre, r, 1); r += 1

        # Image tuning
        self.dsb_brightness = QtWidgets.QDoubleSpinBox(); self.dsb_brightness.setRange(-1.0, 1.0); self.dsb_brightness.setValue(0.0)
        self.dsb_contrast   = QtWidgets.QDoubleSpinBox(); self.dsb_contrast.setRange(0.0, 2.0);  self.dsb_contrast.setValue(1.0)
        self.dsb_saturation = QtWidgets.QDoubleSpinBox(); self.dsb_saturation.setRange(0.0, 2.0);self.dsb_saturation.setValue(1.0)
        self.dsb_sharpness  = QtWidgets.QDoubleSpinBox(); self.dsb_sharpness.setRange(0.0, 2.0); self.dsb_sharpness.setValue(1.0)
        self.cmb_denoise = QtWidgets.QComboBox(); self.cmb_denoise.addItems(["off","fast","high_quality"]); self.cmb_denoise.setCurrentText("fast")

        for w in (self.dsb_brightness, self.dsb_contrast, self.dsb_saturation, self.dsb_sharpness):
            w.valueChanged.connect(self._on_any_change)
        self.cmb_denoise.currentTextChanged.connect(self._on_any_change)
        grid.addWidget(QtWidgets.QLabel("Brightness"), r, 0); grid.addWidget(self.dsb_brightness, r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("Contrast"),   r, 0); grid.addWidget(self.dsb_contrast,   r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("Saturation"), r, 0); grid.addWidget(self.dsb_saturation, r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("Sharpness"),  r, 0); grid.addWidget(self.dsb_sharpness,  r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("Denoise"),    r, 0); grid.addWidget(self.cmb_denoise,    r, 1); r += 1

        # View transforms (server-side)
        self.chk_flip_h = QtWidgets.QCheckBox("Flip H")
        self.chk_flip_v = QtWidgets.QCheckBox("Flip V")
        self.btn_rotate = QtWidgets.QPushButton("Rotate 90°")
        rowv = QtWidgets.QHBoxLayout(); rowv.addWidget(self.chk_flip_h); rowv.addWidget(self.chk_flip_v); rowv.addWidget(self.btn_rotate); rowv.addStretch(1)
        grid.addWidget(QtWidgets.QLabel("Pipeline view"), r, 0); grid.addLayout(rowv, r, 1); r += 1

        self._rot_quadrant = 0
        self.chk_flip_h.toggled.connect(self._emit_view)
        self.chk_flip_v.toggled.connect(self._emit_view)
        self.btn_rotate.clicked.connect(self._on_rotate)

        # HDR & Zoom
        self.dsb_zoom = QtWidgets.QDoubleSpinBox(); self.dsb_zoom.setRange(1.0, 5.0); self.dsb_zoom.setDecimals(2); self.dsb_zoom.setSingleStep(0.1); self.dsb_zoom.setValue(1.0)
        self.dsb_zoom.valueChanged.connect(self._on_any_change)
        self.cmb_hdr = QtWidgets.QComboBox(); self.cmb_hdr.addItems(["off","single","multi","night","unmerged"])
        self.cmb_hdr.currentTextChanged.connect(self._on_any_change)
        grid.addWidget(QtWidgets.QLabel("Digital Zoom (×)"), r, 0); grid.addWidget(self.dsb_zoom, r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("Pi5 HDR mode"), r, 0); grid.addWidget(self.cmb_hdr, r, 1); r += 1

        # Reset
        self.btn_reset = QtWidgets.QPushButton("Reset tuning")
        self.btn_reset.clicked.connect(self._reset_defaults)
        grid.addWidget(self.btn_reset, r, 0, 1, 2); r += 1

        self._reset_defaults()

    def _on_any_change(self, *a):
        # Debounce to batch changes into one set_params
        self._debounce.start()

    def _emit_params(self):
        p = {
            "auto_exposure": bool(self.chk_ae.isChecked()),
            "exposure_us": int(self.sp_exposure.value()),
            "gain": float(self.dsb_gain.value()),
            "fps": float(self.dsb_fps.value()),
            "awb_mode": self.cmb_awb.currentText(),
            "awb_rb": [float(self.dsb_awb_r.value()), float(self.dsb_awb_b.value())],
            "focus_mode": self.cmb_af.currentText(),
            "dioptre": float(self.dsb_dioptre.value()),
            "brightness": float(self.dsb_brightness.value()),
            "contrast": float(self.dsb_contrast.value()),
            "saturation": float(self.dsb_saturation.value()),
            "sharpness": float(self.dsb_sharpness.value()),
            "denoise": self.cmb_denoise.currentText(),
            "zoom": float(self.dsb_zoom.value()),
            "hdr": self.cmb_hdr.currentText(),
        }
        self.paramsChanged.emit(p)

    def _on_rotate(self):
        self._rot_quadrant = (self._rot_quadrant + 1) % 4
        self._emit_view()

    def _emit_view(self):
        self.viewChanged.emit({
            "flip_h": bool(self.chk_flip_h.isChecked()),
            "flip_v": bool(self.chk_flip_v.isChecked()),
            "rot_quadrant": int(self._rot_quadrant)
        })

    def _reset_defaults(self):
        self.blockSignals(True)
        self.chk_ae.setChecked(False)
        self.sp_exposure.setValue(6000)
        self.dsb_gain.setValue(2.0)
        self.dsb_fps.setValue(30.0)
        self.cmb_awb.setCurrentText("auto")
        self.dsb_awb_r.setValue(2.0); self.dsb_awb_b.setValue(2.0)
        self.cmb_af.setCurrentText("manual")
        self.dsb_dioptre.setValue(0.0)
        self.dsb_brightness.setValue(0.0); self.dsb_contrast.setValue(1.0)
        self.dsb_saturation.setValue(1.0); self.dsb_sharpness.setValue(1.0)
        self.cmb_denoise.setCurrentText("fast")
        self.dsb_zoom.setValue(1.0)
        self.cmb_hdr.setCurrentText("off")
        self.chk_flip_h.setChecked(False); self.chk_flip_v.setChecked(False)
        self._rot_quadrant = 0
        self.blockSignals(False)
        # emit both
        self._emit_params()
        self._emit_view()


# ---------- Main Window ----------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RPi Vision Client (dexsdk)")
        self.resize(1480, 980)

        # --- State ---
        self.session_id = "-"
        self.last_frame_wh = (0, 0)
        self._overlay_until = 0.0
        self._last_overlay_img: Optional[QtGui.QImage] = None
        self._fps_acc = 0.0; self._fps_n = 0
        self._last_framets = time.perf_counter()

        # --- WebSocket ---
        self.ws = QtWebSockets.QWebSocket()
        self.ws.error.connect(self._on_ws_error)
        self.ws.connected.connect(self._on_ws_connected)
        self.ws.disconnected.connect(self._on_ws_disconnected)
        self.ws.textMessageReceived.connect(self._on_ws_text)
        self.ws.binaryMessageReceived.connect(self._on_ws_bin)

        # ---- UI scaffold ----
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central); root.setContentsMargins(8,8,8,8); root.setSpacing(8)

        # Left: topbar + video + camera tray (scrollable)
        left = QtWidgets.QVBoxLayout(); left.setSpacing(8)

        # Topbar: connect + capture controls
        topbar = QtWidgets.QHBoxLayout()
        self.ed_host = QtWidgets.QLineEdit("ws://127.0.0.1:8765")
        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_disconnect = QtWidgets.QPushButton("Disconnect"); self.btn_disconnect.setEnabled(False)

        self.btn_capture = QtWidgets.QPushButton("Capture Rect ROI → Active")
        self.btn_capture_poly = QtWidgets.QPushButton("Capture Poly ROI → Active")
        self.btn_clear   = QtWidgets.QPushButton("Clear Active")

        for w in (self.ed_host, self.btn_connect, self.btn_disconnect, self.btn_capture, self.btn_capture_poly, self.btn_clear):
            topbar.addWidget(w); topbar.addSpacing(6)
        topbar.addStretch(1)
        left.addLayout(topbar)

        # Video label
        self.video = VideoLabel()
        self.video.setMinimumSize(1024, 576)
        self.video.setStyleSheet("background:#111;")

        # Collapsible tray with camera controls
        self.cam_panel = CameraPanel()
        self.tray = CollapsibleTray("Camera Controls")
        body = QtWidgets.QVBoxLayout(); body.setContentsMargins(8,4,8,4)

        self.cam_scroll = QtWidgets.QScrollArea()
        self.cam_scroll.setWidgetResizable(True); self.cam_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.cam_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.cam_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.cam_panel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.cam_scroll.setWidget(self.cam_panel)
        body.addWidget(self.cam_scroll)
        self.tray.set_body_layout(body)

        # Vertical splitter (video vs controls)
        self.left_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.left_split.addWidget(self.video)
        self.left_split.addWidget(self.tray)
        self.left_split.setCollapsible(0, False); self.left_split.setCollapsible(1, True)
        self.left_split.setStretchFactor(0, 3); self.left_split.setStretchFactor(1, 1)
        left.addWidget(self.left_split, 1)
        root.addLayout(left, 2)

        # Right panel: Mode, Processing, Templates, Detections
        right = QtWidgets.QVBoxLayout(); right.setSpacing(10)

        # Mode
        mode_box = QtWidgets.QGroupBox("Mode")
        mode_h = QtWidgets.QHBoxLayout(mode_box)
        self.rad_training = QtWidgets.QRadioButton("Training"); self.rad_trigger = QtWidgets.QRadioButton("Trigger")
        self.rad_training.setChecked(True)
        self.btn_trigger = QtWidgets.QPushButton("TRIGGER Detect")
        mode_h.addWidget(self.rad_training); mode_h.addWidget(self.rad_trigger); mode_h.addStretch(1); mode_h.addWidget(self.btn_trigger)
        right.addWidget(mode_box)

        # Processing (server-side)
        proc_box = QtWidgets.QGroupBox("Processing")
        proc_form = QtWidgets.QFormLayout(proc_box)
        self.cmb_proc_w = QtWidgets.QComboBox()
        for w in [320, 480, 640, 800, 960, 1280]:
            self.cmb_proc_w.addItem(str(w))
        self.cmb_proc_w.setCurrentText("640")
        self.sp_every = QtWidgets.QSpinBox(); self.sp_every.setRange(1, 5); self.sp_every.setValue(1)
        proc_form.addRow("Process width", self.cmb_proc_w)
        proc_form.addRow("Detect every Nth frame", self.sp_every)
        right.addWidget(proc_box)

        # Templates
        tmpl_box = QtWidgets.QGroupBox("Templates (max 5)")
        tmpl_v = QtWidgets.QVBoxLayout(tmpl_box)
        row1 = QtWidgets.QHBoxLayout()
        self.cmb_slot = QtWidgets.QComboBox(); [self.cmb_slot.addItem(f"Slot {i+1}") for i in range(5)]
        self.ed_name = QtWidgets.QLineEdit("Obj1")
        self.sp_maxinst = QtWidgets.QSpinBox(); self.sp_maxinst.setRange(1, 10); self.sp_maxinst.setValue(3)
        self.chk_enable = QtWidgets.QCheckBox("Enabled"); self.chk_enable.setChecked(True)
        row1.addWidget(QtWidgets.QLabel("Active:")); row1.addWidget(self.cmb_slot)
        row1.addWidget(QtWidgets.QLabel("Name:")); row1.addWidget(self.ed_name)
        row1.addWidget(QtWidgets.QLabel("Max Inst:")); row1.addWidget(self.sp_maxinst)
        row1.addWidget(self.chk_enable)
        tmpl_v.addLayout(row1)
        right.addWidget(tmpl_box)

        # Detections table
        tbl_box = QtWidgets.QGroupBox("Detections")
        vtbl = QtWidgets.QVBoxLayout(tbl_box)
        self.tbl = QtWidgets.QTableWidget(0, 8)
        self.tbl.setHorizontalHeaderLabels(["Obj","Inst","x","y","θ(deg)","score","inliers","center"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        vtbl.addWidget(self.tbl)
        right.addWidget(tbl_box, 1)

        root.addLayout(right, 1)

        # Status bar
        self.status = QtWidgets.QStatusBar(); self.setStatusBar(self.status)
        self.lbl_info = QtWidgets.QLabel("Disconnected")
        self.lbl_fps = QtWidgets.QLabel("FPS: —")
        self.status.addWidget(self.lbl_info); self.status.addPermanentWidget(self.lbl_fps)

        # ---- Signals ----
        self.btn_connect.clicked.connect(self._connect_ws)
        self.btn_disconnect.clicked.connect(self._disconnect_ws)

        self.btn_capture.clicked.connect(lambda: self.video.enable_rect_selection(True))
        self.btn_capture_poly.clicked.connect(lambda: self.video.enable_polygon_selection(True))
        self.btn_clear.clicked.connect(self._on_clear_active)

        self.video.roiSelected.connect(self._on_rect_selected)
        self.video.polygonSelected.connect(self._on_poly_selected)

        self.rad_training.toggled.connect(self._on_mode_radio)
        self.rad_trigger.toggled.connect(self._on_mode_radio)
        self.btn_trigger.clicked.connect(self._send_trigger)

        self.cmb_proc_w.currentTextChanged.connect(self._on_proc_w_changed)
        self.sp_every.valueChanged.connect(self._on_every_changed)

        self.cmb_slot.currentIndexChanged.connect(self._on_slot_changed)
        self.sp_maxinst.valueChanged.connect(self._on_slot_state)
        self.chk_enable.toggled.connect(self._on_slot_state)

        self.cam_panel.paramsChanged.connect(self._on_params_changed)
        self.cam_panel.viewChanged.connect(self._on_view_changed)

        # Ping timer
        self.ping_timer = QtCore.QTimer(self); self.ping_timer.setInterval(10000)
        self.ping_timer.timeout.connect(lambda: self._send({"type":"ping"}))

        # Start disconnected UI
        self._set_connected(False)

    # -------------- WebSocket helpers --------------
    def _connect_ws(self):
        url = self.ed_host.text().strip()
        self.ws.open(QtCore.QUrl(url))

    def _disconnect_ws(self):
        self.ws.close()

    def _set_connected(self, ok: bool):
        self.btn_connect.setEnabled(not ok)
        self.btn_disconnect.setEnabled(ok)
        self.lbl_info.setText(f"{'Connected' if ok else 'Disconnected'} — Session {self.session_id}")
        if ok: self.ping_timer.start()
        else: self.ping_timer.stop()

    def _send(self, obj: Dict):
        try:
            self.ws.sendTextMessage(json.dumps(obj, separators=(",",":")))
        except Exception:
            pass

    def _on_ws_connected(self):
        self._set_connected(True)
        # push initial config
        mode = "training" if self.rad_training.isChecked() else "trigger"
        self._send({"type":"set_mode","mode":mode})
        self._on_proc_w_changed(self.cmb_proc_w.currentText())
        self._on_every_changed(self.sp_every.value())
        # emit current view/params
        self._on_view_changed(self._view_stub())
        self._on_params_changed(self._params_stub())

    def _on_ws_disconnected(self):
        self._set_connected(False)

    def _on_ws_error(self, err):
        self.status.showMessage(f"WS error: {self.ws.errorString()}", 3000)
        self._set_connected(False)

    # -------------- Incoming messages --------------
    def _on_ws_text(self, txt: str):
        try:
            msg = json.loads(txt)
        except Exception:
            return
        t = msg.get("type","")
        if t == "hello":
            self.session_id = msg.get("session_id","-")
            self.lbl_info.setText(f"Connected — Session {self.session_id}")
        elif t == "frame":
            self._handle_frame(msg)
        elif t == "detections":
            self._handle_detections(msg)
        elif t == "ack":
            if not msg.get("ok", True):
                self.status.showMessage(f"ACK {msg.get('cmd')}: {msg.get('error')}", 4000)
        elif t == "pong":
            pass

    def _on_ws_bin(self, data: bytes):
        # (reserved: if you later switch to binary frames)
        pass

    def _b64_to_qimage(self, b64s: str) -> Optional[QtGui.QImage]:
        try:
            buf = base64.b64decode(b64s.encode("ascii"))
            img = QtGui.QImage.fromData(buf, "JPG")
            if img.isNull():
                # Fallback via cv2 (rare)
                arr = np.frombuffer(buf, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    return None
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                return QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
            return img
        except Exception:
            return None

    def _handle_frame(self, msg: Dict):
        img = self._b64_to_qimage(msg.get("jpeg_b64",""))
        if img is None:
            return
        self.last_frame_wh = (int(msg.get("w", img.width())), int(msg.get("h", img.height())))
        # FPS
        now = time.perf_counter()
        fps = 1.0 / max(1e-6, (now - self._last_framets))
        self._last_framets = now
        self._fps_acc += fps; self._fps_n += 1
        if self._fps_n >= 10:
            self.lbl_fps.setText(f"FPS: {self._fps_acc / self._fps_n:.1f}")
            self._fps_acc = 0.0; self._fps_n = 0

        # If an overlay is active, prefer it for a short window
        if time.time() < self._overlay_until and self._last_overlay_img is not None:
            qimg = self._last_overlay_img
        else:
            qimg = img

        self.video.setPixmapKeepAspect(QtGui.QPixmap.fromImage(qimg))

    def _handle_detections(self, msg: Dict):
        payload = msg.get("payload", {})
        # table
        res = payload.get("result", {})
        objects = res.get("objects", [])
        self._populate_table(objects)

        # overlay (optional)
        if "overlay_jpeg_b64" in msg:
            img = self._b64_to_qimage(msg["overlay_jpeg_b64"])
            if img:
                self._last_overlay_img = img
                self._overlay_until = time.time() + 1.0
                self.video.setPixmapKeepAspect(QtGui.QPixmap.fromImage(img))

    # -------------- Outgoing commands (UI) --------------
    def _on_mode_radio(self):
        mode = "training" if self.rad_training.isChecked() else "trigger"
        self._send({"type":"set_mode","mode":mode})

    def _send_trigger(self):
        self._send({"type":"trigger"})

    def _on_proc_w_changed(self, txt: str):
        try: w = int(txt)
        except: w = 640
        self._send({"type":"set_proc_width","width":w})

    def _on_every_changed(self, n: int):
        self._send({"type":"set_publish_every","n":int(n)})

    def _on_params_changed(self, params: Dict):
        self._send({"type":"set_params","params": dict(params)})

    def _on_view_changed(self, view: Dict):
        self._send({"type":"set_view", **dict(view)})

    def _on_slot_changed(self, idx: int):
        # update default name if empty
        if not self.ed_name.text().strip():
            self.ed_name.setText(f"Obj{idx+1}")
        # also re-apply enabled/maxinst for this slot (no harm)
        self._on_slot_state()

    def _on_slot_state(self):
        slot = int(self.cmb_slot.currentIndex())
        en = bool(self.chk_enable.isChecked())
        mx = int(self.sp_maxinst.value())
        self._send({"type":"set_slot_state","slot":slot,"enabled":en,"max_instances":mx})

    def _on_clear_active(self):
        slot = int(self.cmb_slot.currentIndex())
        self._send({"type":"clear_template","slot":slot})

    # ---- ROI capture mapping (display → processed image coords) ----
    def _map_rect_to_proc(self, rect: QtCore.QRect) -> Optional[Tuple[int,int,int,int]]:
        draw = self.video._last_draw_rect  # region where the qpixmap is painted (kept aspect)
        if rect.isNull() or draw.isNull(): return None
        sel = rect.intersected(draw)
        if sel.isEmpty(): return None
        fw, fh = self.last_frame_wh
        if fw <= 0 or fh <= 0: return None
        sx = fw / float(draw.width()); sy = fh / float(draw.height())
        x = max(0, int((sel.x() - draw.x()) * sx))
        y = max(0, int((sel.y() - draw.y()) * sy))
        w = max(1, int(sel.width() * sx))
        h = max(1, int(sel.height() * sy))
        x2 = min(fw, x + w); y2 = min(fh, y + h)
        return (x, y, x2 - x, y2 - y)

    def _map_poly_to_proc(self, qpoints: List[QtCore.QPoint]) -> Optional[List[List[float]]]:
        draw = self.video._last_draw_rect
        if not qpoints or draw.isNull(): return None
        fw, fh = self.last_frame_wh
        if fw <= 0 or fh <= 0: return None
        sx = fw / float(draw.width()); sy = fh / float(draw.height())
        pts = []
        for p in qpoints:
            if not draw.contains(p):
                px = min(max(p.x(), draw.left()), draw.right())
                py = min(max(p.y(), draw.top()), draw.bottom())
                p = QtCore.QPoint(px, py)
            xi = (p.x() - draw.x()) * sx
            yi = (p.y() - draw.y()) * sy
            pts.append([float(xi), float(yi)])
        return pts

    def _on_rect_selected(self, rect: QtCore.QRect):
        m = self._map_rect_to_proc(rect)
        if not m: return
        x,y,w,h = m
        if w < 10 or h < 10:
            QtWidgets.QMessageBox.warning(self, "ROI too small", "Please select a larger area.")
            return
        slot = int(self.cmb_slot.currentIndex())
        name = self.ed_name.text().strip() or f"Obj{slot+1}"
        maxinst = int(self.sp_maxinst.value())
        self._send({"type":"add_template_rect","slot":slot,"name":name,"rect":[x,y,w,h],"max_instances":maxinst})

    def _on_poly_selected(self, qpoints: List[QtCore.QPoint]):
        pts = self._map_poly_to_proc(qpoints)
        if not pts: return
        # quick bounds check
        arr = np.array(pts, dtype=np.float32)
        x1, y1 = float(arr[:,0].min()), float(arr[:,1].min())
        x2, y2 = float(arr[:,0].max()), float(arr[:,1].max())
        if (x2-x1) < 10 or (y2-y1) < 10:
            QtWidgets.QMessageBox.warning(self, "ROI too small", "Please select a larger polygon.")
            return
        slot = int(self.cmb_slot.currentIndex())
        name = self.ed_name.text().strip() or f"Obj{slot+1}"
        maxinst = int(self.sp_maxinst.value())
        self._send({"type":"add_template_poly","slot":slot,"name":name,"points":pts,"max_instances":maxinst})

    # -------------- Table fill --------------
    def _populate_table(self, objects_report: List[Dict]):
        rows = sum(len(o.get("detections",[])) for o in objects_report)
        self.tbl.setRowCount(rows)
        r = 0
        for obj in objects_report:
            name = obj.get("name","?")
            for det in obj.get("detections",[]):
                ctr = det.get("center")
                vals = [
                    name,
                    str(det.get("instance_id", 0)),
                    f"{det['pose']['x']:.1f}", f"{det['pose']['y']:.1f}", f"{det['pose']['theta_deg']:.1f}",
                    f"{det.get('score', 0.0):.3f}", str(det.get('inliers', 0)),
                    f"({int(ctr[0])},{int(ctr[1])})" if ctr else "—",
                ]
                for c, text in enumerate(vals):
                    self.tbl.setItem(r, c, QtWidgets.QTableWidgetItem(text))
                r += 1
        if rows == 0:
            self.tbl.setRowCount(0)

    # ---- Helpers to emit current widgets' state ----
    def _params_stub(self) -> Dict:
        # Ask the panel to emit immediately (but we also return its current snapshot)
        p = {
            "auto_exposure": self.cam_panel.chk_ae.isChecked(),
            "exposure_us": int(self.cam_panel.sp_exposure.value()),
            "gain": float(self.cam_panel.dsb_gain.value()),
            "fps": float(self.cam_panel.dsb_fps.value()),
            "awb_mode": self.cam_panel.cmb_awb.currentText(),
            "awb_rb": [float(self.cam_panel.dsb_awb_r.value()), float(self.cam_panel.dsb_awb_b.value())],
            "focus_mode": self.cam_panel.cmb_af.currentText(),
            "dioptre": float(self.cam_panel.dsb_dioptre.value()),
            "brightness": float(self.cam_panel.dsb_brightness.value()),
            "contrast": float(self.cam_panel.dsb_contrast.value()),
            "saturation": float(self.cam_panel.dsb_saturation.value()),
            "sharpness": float(self.cam_panel.dsb_sharpness.value()),
            "denoise": self.cam_panel.cmb_denoise.currentText(),
            "zoom": float(self.cam_panel.dsb_zoom.value()),
            "hdr": self.cam_panel.cmb_hdr.currentText(),
        }
        return p

    def _view_stub(self) -> Dict:
        return {
            "flip_h": self.cam_panel.chk_flip_h.isChecked(),
            "flip_v": self.cam_panel.chk_flip_v.isChecked(),
            "rot_quadrant": self.cam_panel._rot_quadrant
        }


# ---------- main ----------
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
