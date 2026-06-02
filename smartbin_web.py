import argparse
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any

from flask import Flask, Response, jsonify, render_template_string, request
from PIL import Image

from waste_sorter import (
    SERVO_INORGANIC_COMMAND,
    SERVO_ORGANIC_COMMAND,
    auto_detect_port,
    cv2,
    load_selected_model,
    predict_with_backend,
    send_command_to_esp32,
)


app = Flask(__name__)

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "smartbin_web.log")


def setup_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


HTML_PAGE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SmartBin Control</title>
  <style>
    :root {
      --ink: #17211d;
      --muted: #63706a;
      --line: #d9e2da;
      --panel: rgba(250, 253, 248, 0.86);
      --leaf: #1f8a5b;
      --steel: #2f6f8f;
      --amber: #e3a229;
      --danger: #b23b3b;
      --bg1: #eaf4e6;
      --bg2: #f7f0de;
      --bg3: #dcecef;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: "Segoe UI", Verdana, sans-serif;
      background:
        radial-gradient(circle at 18% 12%, rgba(31, 138, 91, 0.16), transparent 30%),
        linear-gradient(130deg, var(--bg1), var(--bg2) 48%, var(--bg3));
    }

    button, input, select {
      font: inherit;
    }

    .shell {
      width: min(1380px, calc(100% - 32px));
      margin: 0 auto;
      padding: 24px 0 32px;
    }

    .topbar {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }

    h1 {
      margin: 0;
      font-size: clamp(30px, 5vw, 58px);
      line-height: 0.95;
      letter-spacing: 0;
    }

    .subtitle {
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 15px;
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.72);
      white-space: nowrap;
    }

    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--danger);
      box-shadow: 0 0 0 4px rgba(178, 59, 59, 0.12);
    }

    .dot.on {
      background: var(--leaf);
      box-shadow: 0 0 0 4px rgba(31, 138, 91, 0.14);
    }

    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.55fr) minmax(340px, 0.85fr);
      gap: 18px;
      align-items: start;
    }

    .panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: 0 18px 45px rgba(58, 78, 68, 0.12);
      backdrop-filter: blur(12px);
    }

    .camera-panel {
      overflow: hidden;
    }

    .camera-head,
    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
    }

    .camera-title,
    .panel-title {
      margin: 0;
      font-size: 17px;
      font-weight: 700;
    }

    .video-wrap {
      position: relative;
      background: #101815;
      aspect-ratio: 16 / 9;
    }

    .video-wrap img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }

    .live-badge {
      position: absolute;
      top: 14px;
      left: 14px;
      padding: 8px 10px;
      border-radius: 999px;
      color: white;
      background: rgba(23, 33, 29, 0.76);
      font-size: 13px;
    }

    .metric-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      padding: 14px;
      border-top: 1px solid var(--line);
    }

    .metric {
      min-height: 92px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.68);
    }

    .metric span {
      color: var(--muted);
      font-size: 13px;
    }

    .metric strong {
      display: block;
      margin-top: 7px;
      font-size: clamp(24px, 4vw, 42px);
      line-height: 1;
    }

    .side {
      display: grid;
      gap: 14px;
    }

    .content {
      padding: 14px;
    }

    .prediction {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: center;
      padding: 14px;
      border-radius: 8px;
      background: linear-gradient(135deg, rgba(31, 138, 91, 0.12), rgba(47, 111, 143, 0.12));
      border: 1px solid var(--line);
    }

    .prediction b {
      display: block;
      font-size: 22px;
      margin-bottom: 4px;
    }

    .confidence {
      width: 82px;
      height: 82px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      font-weight: 800;
      background: conic-gradient(var(--leaf) var(--pct, 0%), rgba(255,255,255,0.8) 0);
      border: 1px solid var(--line);
    }

    .controls {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    .btn {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 11px 12px;
      background: white;
      color: var(--ink);
      cursor: pointer;
      font-weight: 700;
      transition: transform 0.14s ease, box-shadow 0.14s ease;
    }

    .btn:hover {
      transform: translateY(-1px);
      box-shadow: 0 10px 22px rgba(23, 33, 29, 0.1);
    }

    .btn.primary {
      color: white;
      background: var(--leaf);
      border-color: var(--leaf);
    }

    .btn.steel {
      color: white;
      background: var(--steel);
      border-color: var(--steel);
    }

    .btn.warn {
      color: #211609;
      background: var(--amber);
      border-color: var(--amber);
    }

    .settings {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 12px;
    }

    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 13px;
    }

    input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      color: var(--ink);
      background: rgba(255, 255, 255, 0.76);
    }

    .history {
      display: grid;
      gap: 8px;
      max-height: 310px;
      overflow: auto;
    }

    .event {
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 10px;
      align-items: center;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.64);
      font-size: 13px;
    }

    .tag {
      border-radius: 999px;
      padding: 5px 8px;
      color: white;
      font-weight: 800;
      background: var(--steel);
    }

    .tag.org {
      background: var(--leaf);
    }

    .muted {
      color: var(--muted);
    }

    .error {
      margin-top: 10px;
      color: var(--danger);
      min-height: 20px;
      font-size: 13px;
    }

    @media (max-width: 980px) {
      .topbar {
        align-items: start;
        flex-direction: column;
      }

      .grid {
        grid-template-columns: 1fr;
      }

      .metric-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }

    @media (max-width: 560px) {
      .shell {
        width: min(100% - 18px, 1380px);
        padding-top: 14px;
      }

      .metric-grid,
      .controls,
      .settings {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="topbar">
      <div>
        <h1>SmartBin</h1>
        <p class="subtitle">Camera laptop, AI classification, servo routing, and live bin statistics.</p>
      </div>
      <div class="status-pill"><span id="runDot" class="dot"></span><span id="runText">Booting</span></div>
    </section>

    <section class="grid">
      <div class="panel camera-panel">
        <div class="camera-head">
          <h2 class="camera-title">Live Camera</h2>
          <span class="muted" id="cameraText">Camera index 0</span>
        </div>
        <div class="video-wrap">
          <img src="/video_feed" alt="SmartBin camera feed">
          <div class="live-badge" id="modelText">Model loading</div>
        </div>
        <div class="metric-grid">
          <div class="metric"><span>Total sorted</span><strong id="totalCount">0</strong></div>
          <div class="metric"><span>Right bin: inorganic</span><strong id="rightCount">0</strong></div>
          <div class="metric"><span>Left bin: organic</span><strong id="leftCount">0</strong></div>
          <div class="metric"><span>Hazardous warning</span><strong id="hazardCount">0</strong></div>
          <div class="metric"><span>Servo command</span><strong id="servoCommand">CENTER</strong></div>
        </div>
      </div>

      <div class="side">
        <div class="panel">
          <div class="panel-head">
            <h2 class="panel-title">Detection</h2>
            <span class="muted" id="updatedText">Waiting</span>
          </div>
          <div class="content">
            <div class="prediction">
              <div>
                <b id="categoryText">No result</b>
                <span class="muted" id="labelText">Point trash at the camera</span>
              </div>
              <div class="confidence" id="confidenceDial">0%</div>
            </div>

            <div class="settings">
              <label>Serial port
                <input id="portInput" placeholder="COM5">
              </label>
              <label>Min confidence
                <input id="confidenceInput" type="number" min="0" max="1" step="0.05" value="0.35">
              </label>
              <label>Detect interval (s)
                <input id="intervalInput" type="number" min="0.5" step="0.5" value="2">
              </label>
              <label>Cooldown (s)
                <input id="cooldownInput" type="number" min="0" step="0.5" value="3">
              </label>
            </div>
            <p class="muted" id="serialResponseText">ESP32 response: waiting</p>
            <div class="error" id="errorText"></div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-head">
            <h2 class="panel-title">Controls</h2>
            <span class="muted" id="serialText">Serial pending</span>
          </div>
          <div class="content">
            <div class="controls">
              <button class="btn primary" data-action="start">Start AI</button>
              <button class="btn" data-action="stop">Stop AI</button>
              <button class="btn steel" data-action="center">Center servo</button>
              <button class="btn warn" data-action="reset_stats">Reset stats</button>
              <button class="btn" data-action="test_inorganic">Test right</button>
              <button class="btn" data-action="test_organic">Test left</button>
              <button class="btn steel" data-action="sweep">Sweep servo</button>
            </div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-head">
            <h2 class="panel-title">Recent Events</h2>
            <span class="muted" id="eventCount">0 events</span>
          </div>
          <div class="content">
            <div class="history" id="history"></div>
          </div>
        </div>
      </div>
    </section>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);

    async function postJson(url, payload) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload || {})
      });
      return response.json();
    }

    function settingsPayload() {
      return {
        port: $("portInput").value.trim(),
        min_confidence: Number($("confidenceInput").value),
        interval: Number($("intervalInput").value),
        cooldown: Number($("cooldownInput").value)
      };
    }

    async function sendAction(action) {
      await postJson("/api/control", { action, ...settingsPayload() });
      await refresh();
    }

    function setText(id, value) {
      $(id).textContent = value;
    }

    function renderHistory(events) {
      const holder = $("history");
      holder.innerHTML = "";
      if (!events.length) {
        holder.innerHTML = '<div class="muted">No sorted events yet.</div>';
        return;
      }

      for (const event of events) {
        const row = document.createElement("div");
        row.className = "event";
        const isOrganic = event.command === "ORGANIC";
        row.innerHTML = `
          <span class="tag ${isOrganic ? "org" : ""}">${isOrganic ? "LEFT" : "RIGHT"}</span>
          <span>${event.label} <span class="muted">(${Math.round(event.confidence * 100)}%)</span></span>
          <span class="muted">${event.time}</span>
        `;
        holder.appendChild(row);
      }
    }

    async function refresh() {
      const status = await fetch("/api/status").then((r) => r.json());
      const result = status.last_result || {};
      const pct = Math.round((result.confidence || 0) * 100);

      $("runDot").classList.toggle("on", status.detection_enabled);
      setText("runText", status.detection_enabled ? "AI running" : "AI stopped");
      setText("cameraText", `Camera index ${status.camera_index}`);
      setText("modelText", status.model_ready ? "Model ready" : "Model loading");
      setText("updatedText", status.updated_at || "Waiting");
      setText("totalCount", status.stats.total);
      setText("rightCount", status.stats.inorganic);
      setText("leftCount", status.stats.organic);
      setText("hazardCount", status.stats.hazardous || 0);
      setText("servoCommand", status.last_servo_command || "CENTER");
      setText("categoryText", result.final_category || "No result");
      setText("labelText", result.original_label ? `Model label: ${result.original_label}` : "Point trash at the camera");
      setText("confidenceDial", `${pct}%`);
      $("confidenceDial").style.setProperty("--pct", `${pct}%`);
      setText("serialText", status.serial_port ? `Serial ${status.serial_port}` : "Serial not connected");
      setText("serialResponseText", `ESP32 response: ${status.last_serial_response || "waiting"}`);
      setText("errorText", status.error || "");
      setText("eventCount", `${status.history.length} events`);

      if (document.activeElement !== $("portInput")) $("portInput").value = status.serial_port || "";
      if (document.activeElement !== $("confidenceInput")) $("confidenceInput").value = status.min_confidence;
      if (document.activeElement !== $("intervalInput")) $("intervalInput").value = status.interval;
      if (document.activeElement !== $("cooldownInput")) $("cooldownInput").value = status.cooldown;

      renderHistory(status.history);
    }

    document.querySelectorAll("[data-action]").forEach((button) => {
      button.addEventListener("click", () => sendAction(button.dataset.action));
    });

    ["portInput", "confidenceInput", "intervalInput", "cooldownInput"].forEach((id) => {
      $(id).addEventListener("change", async () => {
        await postJson("/api/settings", settingsPayload());
        await refresh();
      });
    });

    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""


class SmartBinRuntime:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.serial_lock = threading.Lock()
        self.processor: Any = None
        self.model: Any = None
        self.model_backend = "transformers"
        self.model_ready = False
        self.model_loading = False
        self.camera_index = 0
        self.camera_active = True
        self.detection_enabled = True
        self.serial_enabled = True
        self.serial_port = auto_detect_port()
        self.baudrate = 115200
        self.interval = 2.0
        self.cooldown = 3.0
        self.min_confidence = 0.35
        self.reset_after = 1.0
        self.last_detection_at = 0.0
        self.last_counted_at = 0.0
        self.last_sent_at = 0.0
        self.last_servo_command = "CENTER"
        self.last_serial_response = ""
        self.last_result: dict[str, Any] | None = None
        self.pending_command = ""
        self.pending_count = 0
        self.stable_frames_required = 2
        self.updated_at = ""
        self.error = ""
        self.stats = {"total": 0, "organic": 0, "inorganic": 0, "hazardous": 0}
        self.history: deque[dict[str, Any]] = deque(maxlen=20)
        self.frame_jpeg: bytes | None = None
        self.latest_frame: Any = None
        self.stop_event = threading.Event()

    def start(self) -> None:
        logging.info("Starting SmartBin runtime")
        threading.Thread(target=self._load_model, daemon=True).start()
        threading.Thread(target=self._camera_loop, daemon=True).start()
        threading.Thread(target=self._detection_loop, daemon=True).start()

    def _load_model(self) -> None:
        logging.info("Loading model")
        with self.lock:
            self.model_loading = True
            self.error = ""

        try:
            backend, processor, model = load_selected_model(self.model_backend)
            with self.lock:
                self.model_backend = backend
                self.processor = processor
                self.model = model
                self.model_ready = True
                self.model_loading = False
            logging.info("Model loaded")
        except Exception as exc:
            logging.exception("Model failed to load")
            with self.lock:
                self.error = str(exc)
                self.model_loading = False

    def _camera_loop(self) -> None:
        if cv2 is None:
            logging.error("opencv-python is missing")
            with self.lock:
                self.error = "opencv-python is missing. Run: python -m pip install -r requirements.txt"
            return

        camera = None
        open_index = None

        while not self.stop_event.is_set():
            with self.lock:
                camera_index = self.camera_index
                camera_active = self.camera_active

            if not camera_active:
                time.sleep(0.2)
                continue

            if camera is None or open_index != camera_index:
                if camera is not None:
                    camera.release()
                logging.info("Opening camera index %s", camera_index)
                camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera.set(cv2.CAP_PROP_FPS, 30)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                open_index = camera_index
                time.sleep(0.4)

            if not camera.isOpened():
                logging.error("Camera %s is not available", camera_index)
                with self.lock:
                    self.error = f"Camera {camera_index} is not available"
                time.sleep(1)
                continue

            ok, frame = camera.read()
            if not ok:
                logging.error("Could not read camera frame")
                with self.lock:
                    self.error = "Could not read camera frame"
                time.sleep(0.5)
                continue

            self._update_frame(frame)
            with self.lock:
                self.latest_frame = frame.copy()
            time.sleep(0.03)

        if camera is not None:
            camera.release()

    def _update_frame(self, frame: Any) -> None:
        preview = frame.copy()
        with self.lock:
            result = self.last_result
            detection_enabled = self.detection_enabled
            model_ready = self.model_ready

        label = "AI RUNNING" if detection_enabled and model_ready else "AI WAITING"
        color = (66, 145, 82) if detection_enabled and model_ready else (44, 111, 143)
        cv2.rectangle(preview, (16, 16), (260, 62), color, -1)
        cv2.putText(preview, label, (28, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)

        if result:
            overlay = f"{result['command']}  {result['confidence']:.0%}"
            cv2.rectangle(preview, (16, 74), (420, 120), (20, 24, 22), -1)
            cv2.putText(preview, overlay, (28, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)

        ok, encoded = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        if ok:
            with self.lock:
                self.frame_jpeg = encoded.tobytes()

    def _detection_loop(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()

            if frame is not None:
                self._maybe_detect(frame)

            time.sleep(0.05)

    def _maybe_detect(self, frame: Any) -> None:
        with self.lock:
            ready = self.model_ready and self.detection_enabled
            due = time.monotonic() - self.last_detection_at >= self.interval
            backend = self.model_backend
            processor = self.processor
            model = self.model

        if not ready or not due:
            return

        with self.lock:
            self.last_detection_at = time.monotonic()

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = predict_with_backend(Image.fromarray(rgb_frame), backend, processor, model)
            logging.info(
                "Detection label=%s command=%s confidence=%.3f",
                result["original_label"],
                result["command"],
                result["confidence"],
            )
            now = time.monotonic()
            should_count = self._is_stable_detection(result, now)

            with self.lock:
                self.last_result = result
                self.updated_at = datetime.now().strftime("%H:%M:%S")
                self.error = ""

            if should_count:
                self._record_event(result)
        except Exception as exc:
            logging.exception("Detection failed")
            with self.lock:
                self.error = f"Detection error: {exc}"

    def _is_stable_detection(self, result: dict[str, Any], now: float) -> bool:
        stable_key = result.get("group", result["command"])

        with self.lock:
            if stable_key == self.pending_command:
                self.pending_count += 1
            else:
                self.pending_command = stable_key
                self.pending_count = 1

            confidence_ok = result["confidence"] >= self.min_confidence
            stable_ok = self.pending_count >= self.stable_frames_required
            cooldown_ok = now - self.last_counted_at >= self.cooldown

        return confidence_ok and stable_ok and cooldown_ok

    def _record_event(self, result: dict[str, Any]) -> None:
        command = result["command"]
        sent = False
        error = ""
        responses = []

        with self.lock:
            port = self.serial_port
            serial_enabled = self.serial_enabled
            baudrate = self.baudrate
            reset_after = self.reset_after

        if serial_enabled and port:
            try:
                logging.info("Sending serial command %s to %s", command, port)
                with self.serial_lock:
                    responses = send_command_to_esp32(
                        port,
                        command,
                        baudrate,
                        reset_after,
                        response_wait=0.15,
                        boot_wait=0,
                    )
                sent = True
                logging.info("ESP32 response: %s", " | ".join(responses) if responses else "no response")
            except Exception as exc:
                logging.exception("Serial command failed")
                error = f"Serial error: {exc}"
                responses = []

        with self.lock:
            self.last_counted_at = time.monotonic()
            self.last_servo_command = command
            self.last_serial_response = " | ".join(responses) if responses else self.last_serial_response
            self.stats["total"] += 1

            if result.get("group") == "hazardous":
                self.stats["hazardous"] += 1
                self.stats["inorganic"] += 1
            elif command == SERVO_INORGANIC_COMMAND:
                self.stats["inorganic"] += 1
            else:
                self.stats["organic"] += 1

            self.history.appendleft(
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "command": command,
                    "label": result["original_label"],
                    "category": result["final_category"],
                    "confidence": result["confidence"],
                    "serial_sent": sent,
                }
            )

            if error:
                self.error = error

    def send_manual_command(self, command: str) -> None:
        with self.lock:
            port = self.serial_port
            baudrate = self.baudrate
            reset_after = self.reset_after

        if not port:
            raise RuntimeError("Serial port is not set")

        logging.info("Sending manual command %s to %s", command, port)
        with self.serial_lock:
            responses = send_command_to_esp32(
                port,
                command,
                baudrate,
                reset_after,
                response_wait=0.6,
                boot_wait=0,
            )
        logging.info("ESP32 response: %s", " | ".join(responses) if responses else "no response")
        with self.lock:
            self.last_servo_command = command
            self.last_serial_response = " | ".join(responses) if responses else "No response from ESP32"
            self.error = ""

    def update_settings(self, data: dict[str, Any]) -> None:
        with self.lock:
            port = str(data.get("port") or "").strip()
            self.serial_port = port or auto_detect_port()
            self.min_confidence = self._number(data.get("min_confidence"), self.min_confidence, 0, 1)
            self.interval = self._number(data.get("interval"), self.interval, 0.5, 30)
            self.cooldown = self._number(data.get("cooldown"), self.cooldown, 0, 60)
            self.reset_after = self._number(data.get("reset_after"), self.reset_after, 0, 30)
            self.camera_index = int(self._number(data.get("camera_index"), self.camera_index, 0, 20))

    def reset_stats(self) -> None:
        with self.lock:
            self.stats = {"total": 0, "organic": 0, "inorganic": 0, "hazardous": 0}
            self.history.clear()
            self.last_counted_at = 0

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "model_ready": self.model_ready,
                "model_loading": self.model_loading,
                "model_backend": self.model_backend,
                "camera_active": self.camera_active,
                "camera_index": self.camera_index,
                "detection_enabled": self.detection_enabled,
                "serial_port": self.serial_port,
                "last_servo_command": self.last_servo_command,
                "last_serial_response": self.last_serial_response,
                "stats": dict(self.stats),
                "last_result": self.last_result,
                "updated_at": self.updated_at,
                "error": self.error,
                "history": list(self.history),
                "min_confidence": self.min_confidence,
                "interval": self.interval,
                "cooldown": self.cooldown,
            }

    @staticmethod
    def _number(value: Any, fallback: float, minimum: float, maximum: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return fallback
        return max(minimum, min(maximum, parsed))


runtime = SmartBinRuntime()


@app.get("/")
def index() -> str:
    return render_template_string(HTML_PAGE)


@app.get("/video_feed")
def video_feed() -> Response:
    return Response(frame_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


def frame_stream():
    while True:
        with runtime.lock:
            frame = runtime.frame_jpeg

        if frame is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

        time.sleep(0.08)


@app.get("/api/status")
def api_status() -> Response:
    return jsonify(runtime.status())


@app.post("/api/settings")
def api_settings() -> Response:
    runtime.update_settings(request.get_json(silent=True) or {})
    return jsonify(runtime.status())


@app.post("/api/control")
def api_control() -> Response:
    data = request.get_json(silent=True) or {}
    runtime.update_settings(data)
    action = data.get("action")

    try:
        if action == "start":
            with runtime.lock:
                runtime.detection_enabled = True
                runtime.error = ""
        elif action == "stop":
            with runtime.lock:
                runtime.detection_enabled = False
        elif action == "reset_stats":
            runtime.reset_stats()
        elif action == "center":
            runtime.send_manual_command("CENTER")
        elif action == "sweep":
            runtime.send_manual_command("SWEEP")
        elif action == "test_inorganic":
            runtime.send_manual_command(SERVO_INORGANIC_COMMAND)
        elif action == "test_organic":
            runtime.send_manual_command(SERVO_ORGANIC_COMMAND)
        else:
            with runtime.lock:
                runtime.error = f"Unknown action: {action}"
    except Exception as exc:
        with runtime.lock:
            runtime.error = str(exc)

    return jsonify(runtime.status())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local SmartBin web dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--serial-port", default=None)
    parser.add_argument("--no-serial", action="store_true")
    parser.add_argument(
        "--model-backend",
        choices=["transformers", "keras", "fathima"],
        default="transformers",
    )
    return parser


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()
    logging.info(
        "Starting Flask server host=%s port=%s camera_index=%s serial_port=%s serial_enabled=%s",
        args.host,
        args.port,
        args.camera_index,
        args.serial_port,
        not args.no_serial,
    )
    runtime.model_backend = args.model_backend
    runtime.camera_index = args.camera_index
    runtime.serial_enabled = not args.no_serial
    runtime.serial_port = args.serial_port or runtime.serial_port
    runtime.start()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
