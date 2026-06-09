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
    list_ports,
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
  <title>SmartBin City Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #09090b;
      --panel: rgba(24, 24, 27, 0.68);
      --panel-strong: rgba(39, 39, 42, 0.78);
      --border: rgba(255, 255, 255, 0.1);
      --border-hot: rgba(248, 113, 113, 0.72);
      --text: #fafafa;
      --muted: #a1a1aa;
      --faint: #71717a;
      --green: #22c55e;
      --yellow: #eab308;
      --orange: #f97316;
      --red: #ef4444;
      --cyan: #22d3ee;
      --blue: #60a5fa;
      --violet: #a78bfa;
      --shadow: 0 24px 80px rgba(0, 0, 0, 0.46);
      --radius: 20px;
    }

    * { box-sizing: border-box; }

    html {
      background: var(--bg);
      color-scheme: dark;
    }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: "Inter", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 12% 8%, rgba(34, 211, 238, 0.18), transparent 28%),
        radial-gradient(circle at 84% 10%, rgba(167, 139, 250, 0.18), transparent 28%),
        radial-gradient(circle at 50% 100%, rgba(34, 197, 94, 0.13), transparent 34%),
        #09090b;
      overflow-x: hidden;
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: 0.28;
      background-image:
        linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px);
      background-size: 58px 58px;
      mask-image: radial-gradient(circle at center, black, transparent 78%);
    }

    button, input, select {
      font: inherit;
    }

    .shell {
      width: min(1520px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 42px;
      position: relative;
      z-index: 1;
    }

    .topbar {
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: center;
      margin-bottom: 18px;
    }

    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: #d4d4d8;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.05);
      border-radius: 999px;
      padding: 7px 10px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    h1 {
      margin: 12px 0 0;
      font-size: clamp(34px, 5vw, 78px);
      line-height: 0.92;
      letter-spacing: -0.07em;
    }

    .subtitle {
      max-width: 720px;
      margin: 14px 0 0;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.7;
    }

    .status-cluster {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 10px 13px;
      color: #e4e4e7;
      background: rgba(24, 24, 27, 0.72);
      backdrop-filter: blur(18px);
      box-shadow: 0 12px 32px rgba(0,0,0,0.24);
      white-space: nowrap;
      font-size: 13px;
      font-weight: 700;
    }

    .dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: var(--red);
      box-shadow: 0 0 0 5px rgba(239, 68, 68, 0.14);
    }

    .dot.on {
      background: var(--green);
      box-shadow: 0 0 0 5px rgba(34, 197, 94, 0.14);
      animation: pulseDot 1.8s infinite;
    }

    .dot.off {
      background: var(--red);
      box-shadow: 0 0 0 5px rgba(239, 68, 68, 0.14);
    }

    @keyframes pulseDot {
      0%, 100% { transform: scale(1); }
      50% { transform: scale(1.25); }
    }

    .glass {
      position: relative;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background:
        linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03)),
        var(--panel);
      box-shadow: var(--shadow);
      backdrop-filter: blur(22px);
      overflow: hidden;
    }

    .glass::before {
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      background: linear-gradient(135deg, rgba(255,255,255,0.18), transparent 32%);
      opacity: 0.36;
    }

    .glass > * {
      position: relative;
      z-index: 1;
    }

    .hero-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin: 22px 0;
    }

    .metric-card {
      min-height: 144px;
      padding: 18px;
      transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
      animation: fadeUp 0.55s ease both;
    }

    .metric-card:hover {
      transform: translateY(-4px);
      border-color: rgba(255,255,255,0.2);
      box-shadow: 0 28px 90px rgba(0,0,0,0.56);
    }

    .metric-label {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .metric-value {
      display: block;
      margin-top: 18px;
      font-size: clamp(32px, 4vw, 54px);
      line-height: 0.9;
      letter-spacing: -0.06em;
      font-weight: 900;
    }

    .metric-note {
      margin-top: 12px;
      color: var(--faint);
      font-size: 13px;
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(390px, 0.8fr);
      gap: 18px;
      align-items: start;
    }

    .left-column,
    .right-column {
      display: grid;
      gap: 18px;
    }

    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 18px 20px;
      border-bottom: 1px solid var(--border);
    }

    .panel-title {
      margin: 0;
      font-size: 16px;
      letter-spacing: -0.02em;
    }

    .panel-subtitle {
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
    }

    .camera-wrap {
      position: relative;
      aspect-ratio: 16 / 9;
      background: #030712;
      border-bottom: 1px solid var(--border);
    }

    .camera-wrap img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
      filter: saturate(1.06) contrast(1.03);
    }

    .camera-overlay {
      position: absolute;
      inset: 16px;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      pointer-events: none;
    }

    .camera-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 9px 11px;
      border: 1px solid rgba(255,255,255,0.14);
      border-radius: 999px;
      color: white;
      background: rgba(9,9,11,0.58);
      backdrop-filter: blur(14px);
      font-size: 12px;
      font-weight: 800;
    }

    .detection-strip {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 14px;
      align-items: center;
      padding: 18px 20px;
    }

    .prediction-title {
      margin: 0;
      font-size: clamp(20px, 2.2vw, 32px);
      letter-spacing: -0.04em;
    }

    .prediction-meta {
      margin-top: 8px;
      color: var(--muted);
      font-size: 14px;
    }

    .confidence-ring {
      --pct: 0%;
      width: 94px;
      height: 94px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      font-weight: 900;
      background:
        radial-gradient(circle at center, #111113 56%, transparent 58%),
        conic-gradient(var(--green) var(--pct), rgba(255,255,255,0.08) 0);
      border: 1px solid var(--border);
      box-shadow: inset 0 0 24px rgba(255,255,255,0.04);
    }

    .bin-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }

    .bin-card {
      padding: 20px;
      display: grid;
      grid-template-columns: 165px 1fr;
      gap: 20px;
      min-height: 300px;
      transition: border-color 0.25s ease, box-shadow 0.25s ease, transform 0.25s ease;
    }

    .bin-card:hover {
      transform: translateY(-3px);
    }

    .bin-card.critical {
      border-color: var(--border-hot);
      box-shadow: 0 0 0 1px rgba(239,68,68,0.22), 0 28px 90px rgba(239, 68, 68, 0.14);
    }

    .bin-visual {
      position: relative;
      width: 150px;
      height: 245px;
      justify-self: center;
      border: 1px solid rgba(255,255,255,0.24);
      border-radius: 24px 24px 34px 34px;
      background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
      overflow: hidden;
      box-shadow: inset 0 0 34px rgba(255,255,255,0.05), 0 24px 60px rgba(0,0,0,0.38);
    }

    .bin-visual::before {
      content: "";
      position: absolute;
      top: -14px;
      left: 18px;
      width: 114px;
      height: 16px;
      border-radius: 999px;
      background: rgba(255,255,255,0.18);
      border: 1px solid rgba(255,255,255,0.2);
    }

    .liquid {
      --level: 0%;
      --liquid: var(--green);
      position: absolute;
      left: 0;
      right: 0;
      bottom: 0;
      height: var(--level);
      background:
        linear-gradient(180deg, rgba(255,255,255,0.26), transparent 18%),
        linear-gradient(180deg, color-mix(in srgb, var(--liquid), white 8%), var(--liquid));
      transition: height 0.75s cubic-bezier(.2,.9,.2,1), background 0.3s ease;
    }

    .liquid::before,
    .liquid::after {
      content: "";
      position: absolute;
      left: -38%;
      top: -20px;
      width: 176%;
      height: 44px;
      border-radius: 45%;
      background: rgba(255,255,255,0.23);
      animation: wave 5.5s linear infinite;
    }

    .liquid::after {
      opacity: 0.45;
      animation-duration: 8s;
      animation-direction: reverse;
    }

    @keyframes wave {
      from { transform: translateX(-12%) rotate(0deg); }
      to { transform: translateX(12%) rotate(360deg); }
    }

    .bin-percent {
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      font-size: 32px;
      font-weight: 900;
      letter-spacing: -0.06em;
      text-shadow: 0 2px 18px rgba(0,0,0,0.45);
    }

    .bin-info {
      display: grid;
      align-content: center;
      gap: 12px;
    }

    .bin-info h3 {
      margin: 0;
      font-size: 22px;
      letter-spacing: -0.04em;
    }

    .status-badge {
      width: fit-content;
      border-radius: 999px;
      padding: 7px 10px;
      font-size: 12px;
      font-weight: 900;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: #06130b;
      background: var(--green);
    }

    .status-badge.filling { background: var(--yellow); color: #1c1602; }
    .status-badge.nearly { background: var(--orange); color: #1c0c02; }
    .status-badge.critical { background: var(--red); color: white; }

    .bin-meta {
      color: var(--muted);
      display: grid;
      gap: 7px;
      font-size: 13px;
    }

    .empty-btn {
      display: none;
      width: fit-content;
      border: 1px solid rgba(248,113,113,0.65);
      border-radius: 999px;
      padding: 10px 13px;
      color: white;
      background: linear-gradient(135deg, rgba(239,68,68,0.9), rgba(153,27,27,0.86));
      cursor: pointer;
      font-weight: 900;
      box-shadow: 0 0 32px rgba(239,68,68,0.22);
    }

    .empty-btn.visible {
      display: inline-flex;
    }

    .analytics-grid {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
    }

    .chart-card {
      padding: 18px;
      min-height: 290px;
    }

    .chart-card canvas {
      width: 100% !important;
      height: 230px !important;
    }

    .chart-card.large canvas {
      height: 270px !important;
    }

    .small-stat-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      padding: 18px;
    }

    .mini-card {
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px;
      background: rgba(255,255,255,0.045);
    }

    .mini-card span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }

    .mini-card strong {
      display: block;
      margin-top: 10px;
      font-size: 30px;
      letter-spacing: -0.05em;
    }

    .connection-card {
      display: grid;
      gap: 8px;
      margin: 0 18px 18px;
      padding: 16px;
      border: 1px solid var(--border);
      border-radius: 18px;
      background: rgba(255,255,255,0.04);
    }

    .connection-row {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 14px;
      font-weight: 800;
    }

    .connection-meta {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }

    .camera-setup-card {
      display: grid;
      gap: 12px;
      margin: 0 18px 18px;
      padding: 16px;
      border: 1px solid rgba(34, 211, 238, 0.24);
      border-radius: 18px;
      background:
        radial-gradient(circle at top left, rgba(34,211,238,0.12), transparent 34%),
        rgba(255,255,255,0.045);
    }

    .camera-setup-actions {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: end;
    }

    .settings-grid,
    .controls-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      padding: 18px;
    }

    label {
      display: grid;
      gap: 7px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    input,
    select {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 11px 12px;
      color: var(--text);
      outline: none;
      background: rgba(255,255,255,0.06);
      transition: border-color 0.18s ease, box-shadow 0.18s ease;
    }

    input:focus,
    select:focus {
      border-color: rgba(34,211,238,0.55);
      box-shadow: 0 0 0 4px rgba(34,211,238,0.12);
    }

    .btn {
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px 13px;
      color: var(--text);
      background: rgba(255,255,255,0.06);
      cursor: pointer;
      font-weight: 900;
      transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
    }

    .btn:hover {
      transform: translateY(-2px);
      border-color: rgba(255,255,255,0.22);
      box-shadow: 0 18px 42px rgba(0,0,0,0.28);
    }

    .btn.primary {
      border-color: rgba(34,197,94,0.62);
      background: linear-gradient(135deg, #22c55e, #15803d);
    }

    .btn.steel {
      border-color: rgba(34,211,238,0.4);
      background: linear-gradient(135deg, rgba(14,165,233,0.92), rgba(37,99,235,0.82));
    }

    .btn.warn {
      color: #1c1602;
      border-color: rgba(234,179,8,0.72);
      background: linear-gradient(135deg, #fde047, #eab308);
    }

    .btn.danger {
      border-color: rgba(239,68,68,0.64);
      background: linear-gradient(135deg, #ef4444, #991b1b);
    }

    .activity {
      display: grid;
      gap: 10px;
      padding: 18px;
      max-height: 430px;
      overflow: auto;
    }

    .event {
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 12px;
      align-items: center;
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 12px;
      background: rgba(255,255,255,0.045);
      animation: fadeUp 0.35s ease both;
    }

    .event-icon {
      width: 38px;
      height: 38px;
      display: grid;
      place-items: center;
      border-radius: 14px;
      background: rgba(34,197,94,0.14);
    }

    .event.right .event-icon {
      background: rgba(96,165,250,0.14);
    }

    .event.empty .event-icon {
      background: rgba(234,179,8,0.14);
    }

    .event-main {
      min-width: 0;
    }

    .event-title {
      font-weight: 800;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .event-sub {
      margin-top: 4px;
      color: var(--muted);
      font-size: 13px;
    }

    .muted { color: var(--muted); }

    .error {
      min-height: 20px;
      padding: 0 18px 18px;
      color: #fca5a5;
      font-size: 13px;
    }

    .alert-banner {
      display: none;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      margin: 18px 0;
      padding: 16px 18px;
      border: 1px solid var(--border-hot);
      border-radius: var(--radius);
      background:
        radial-gradient(circle at 12% 0%, rgba(248,113,113,0.3), transparent 34%),
        linear-gradient(135deg, rgba(127,29,29,0.72), rgba(24,24,27,0.82));
      box-shadow: 0 0 0 1px rgba(239,68,68,0.15), 0 0 44px rgba(239,68,68,0.18);
      animation: alertPulse 1.45s infinite;
    }

    .alert-banner.visible {
      display: flex;
    }

    .alert-title {
      font-size: 20px;
      font-weight: 950;
      letter-spacing: -0.04em;
    }

    @keyframes alertPulse {
      0%, 100% { box-shadow: 0 0 0 1px rgba(239,68,68,0.15), 0 0 34px rgba(239,68,68,0.18); }
      50% { box-shadow: 0 0 0 1px rgba(239,68,68,0.4), 0 0 64px rgba(239,68,68,0.34); }
    }

    .toast {
      position: fixed;
      right: 22px;
      bottom: 22px;
      width: min(420px, calc(100% - 32px));
      border: 1px solid rgba(255,255,255,0.14);
      border-radius: 20px;
      padding: 15px;
      background: rgba(24,24,27,0.92);
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
      transform: translateY(130%);
      opacity: 0;
      transition: transform 0.28s ease, opacity 0.28s ease;
      z-index: 20;
    }

    .toast.show {
      transform: translateY(0);
      opacity: 1;
    }

    .modal-backdrop {
      position: fixed;
      inset: 0;
      display: none;
      place-items: center;
      padding: 20px;
      background: rgba(0,0,0,0.62);
      backdrop-filter: blur(12px);
      z-index: 30;
    }

    .modal-backdrop.visible {
      display: grid;
    }

    .modal {
      width: min(460px, 100%);
      padding: 22px;
      border: 1px solid var(--border);
      border-radius: 24px;
      background:
        radial-gradient(circle at top right, rgba(34,211,238,0.13), transparent 34%),
        rgba(24,24,27,0.96);
      box-shadow: var(--shadow);
      animation: modalIn 0.22s ease both;
    }

    .modal h3 {
      margin: 0;
      font-size: 24px;
      letter-spacing: -0.05em;
    }

    .modal p {
      color: var(--muted);
      line-height: 1.6;
    }

    .modal-actions {
      display: flex;
      justify-content: flex-end;
      gap: 10px;
      margin-top: 18px;
    }

    @keyframes modalIn {
      from { opacity: 0; transform: translateY(12px) scale(0.97); }
      to { opacity: 1; transform: translateY(0) scale(1); }
    }

    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(14px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 1180px) {
      .layout,
      .analytics-grid {
        grid-template-columns: 1fr;
      }

      .hero-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }

    @media (max-width: 760px) {
      .shell {
        width: min(100% - 20px, 1520px);
        padding-top: 18px;
      }

      .topbar,
      .alert-banner {
        align-items: flex-start;
        flex-direction: column;
      }

      .status-cluster {
        justify-content: flex-start;
      }

      .hero-grid,
      .bin-grid,
      .settings-grid,
      .controls-grid,
      .small-stat-grid {
        grid-template-columns: 1fr;
      }

      .bin-card {
        grid-template-columns: 1fr;
      }

      .bin-info {
        text-align: center;
        justify-items: center;
      }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="topbar">
      <div>
        <span class="eyebrow">Smart City IoT Node</span>
        <h1>SmartBin Command Center</h1>
        <p class="subtitle">AI classification, live camera telemetry, servo routing, persistent bin fill analytics, and real-time collection alerts.</p>
      </div>
      <div class="status-cluster">
        <div class="pill"><span id="runDot" class="dot"></span><span id="runText">Booting</span></div>
        <div class="pill" id="modelText">Model loading</div>
        <div class="pill" id="serialText">Serial pending</div>
      </div>
    </section>

    <section class="alert-banner" id="alertBanner">
      <div>
        <div class="alert-title">🚨 BIN NEAR FULL</div>
        <div class="muted" id="alertMessage">Please empty the bin immediately.</div>
      </div>
      <button class="btn danger" id="alertEmptyBtn">Empty Bin</button>
    </section>

    <section class="hero-grid">
      <div class="glass metric-card">
        <div class="metric-label"><span>Total Waste Today</span><span>24H</span></div>
        <strong class="metric-value" id="todayTotal">0</strong>
        <div class="metric-note" id="todayNote">No disposal events yet</div>
      </div>
      <div class="glass metric-card">
        <div class="metric-label"><span>Left Bin Fill</span><span>Bin A</span></div>
        <strong class="metric-value" id="leftFillHero">0%</strong>
        <div class="metric-note" id="leftTodayHero">0 events today</div>
      </div>
      <div class="glass metric-card">
        <div class="metric-label"><span>Right Bin Fill</span><span>Bin B</span></div>
        <strong class="metric-value" id="rightFillHero">0%</strong>
        <div class="metric-note" id="rightTodayHero">0 events today</div>
      </div>
      <div class="glass metric-card">
        <div class="metric-label"><span>System Status</span><span id="updatedText">Waiting</span></div>
        <strong class="metric-value" id="systemStatus">Online</strong>
        <div class="metric-note" id="systemNote">Awaiting telemetry</div>
      </div>
    </section>

    <section class="layout">
      <div class="left-column">
        <div class="glass">
          <div class="panel-head">
            <div>
              <h2 class="panel-title">Live AI Camera</h2>
              <div class="panel-subtitle" id="cameraText">Camera index 0</div>
            </div>
            <div class="pill" id="servoCommand">CENTER</div>
          </div>
          <div class="camera-wrap">
            <img src="/video_feed" alt="SmartBin live camera feed">
            <div class="camera-overlay">
              <div class="camera-badge"><span class="dot on"></span> LIVE FEED</div>
              <div class="camera-badge" id="confidenceText">0% confidence</div>
            </div>
          </div>
          <div class="detection-strip">
            <div>
              <h2 class="prediction-title" id="categoryText">No result</h2>
              <div class="prediction-meta" id="labelText">Point waste at the camera</div>
            </div>
            <div class="confidence-ring" id="confidenceDial">0%</div>
          </div>
        </div>

        <section class="bin-grid">
          <article class="glass bin-card" id="leftBinCard">
            <div class="bin-visual">
              <div class="liquid" id="leftLiquid"></div>
              <div class="bin-percent" id="leftBinPct">0%</div>
            </div>
            <div class="bin-info">
              <span class="status-badge" id="leftStatus">Normal</span>
              <h3>Left Bin <span class="muted">Waste Bin A</span></h3>
              <div class="bin-meta">
                <span id="leftEventsToday">0 events today</span>
                <span id="leftLastTime">Last disposal: never</span>
                <span>Route: ORGANIC / LEFT</span>
              </div>
              <button class="empty-btn" data-empty-bin="left">Empty Bin</button>
            </div>
          </article>

          <article class="glass bin-card" id="rightBinCard">
            <div class="bin-visual">
              <div class="liquid" id="rightLiquid"></div>
              <div class="bin-percent" id="rightBinPct">0%</div>
            </div>
            <div class="bin-info">
              <span class="status-badge" id="rightStatus">Normal</span>
              <h3>Right Bin <span class="muted">Waste Bin B</span></h3>
              <div class="bin-meta">
                <span id="rightEventsToday">0 events today</span>
                <span id="rightLastTime">Last disposal: never</span>
                <span>Route: INORGANIC / RIGHT</span>
              </div>
              <button class="empty-btn" data-empty-bin="right">Empty Bin</button>
            </div>
          </article>
        </section>

        <section class="analytics-grid">
          <div class="glass chart-card large">
            <div class="panel-head" style="padding:0 0 16px;border:0;">
              <h2 class="panel-title">Daily Disposal Trend</h2>
              <span class="panel-subtitle">Last 7 days</span>
            </div>
            <canvas id="dailyTrendChart"></canvas>
          </div>
          <div class="glass chart-card">
            <div class="panel-head" style="padding:0 0 16px;border:0;">
              <h2 class="panel-title">Left vs Right Usage</h2>
              <span class="panel-subtitle">All stored events</span>
            </div>
            <canvas id="usageChart"></canvas>
          </div>
        </section>

        <div class="glass chart-card">
          <div class="panel-head" style="padding:0 0 16px;border:0;">
            <h2 class="panel-title">Weekly Disposal Count</h2>
            <span class="panel-subtitle">Mon - Sun</span>
          </div>
          <canvas id="weeklyChart"></canvas>
        </div>
      </div>

      <aside class="right-column">
        <div class="glass">
          <div class="panel-head">
            <h2 class="panel-title">Daily Statistics</h2>
            <span class="panel-subtitle">Persistent storage</span>
          </div>
          <div class="small-stat-grid">
            <div class="mini-card"><span>Today</span><strong id="statToday">0</strong></div>
            <div class="mini-card"><span>This Week</span><strong id="statWeek">0</strong></div>
            <div class="mini-card"><span>This Month</span><strong id="statMonth">0</strong></div>
            <div class="mini-card"><span>Hazardous</span><strong id="hazardCount">0</strong></div>
          </div>
        </div>

        <div class="glass">
          <div class="panel-head">
            <h2 class="panel-title">Controls</h2>
            <span class="panel-subtitle">AI + Servo</span>
          </div>
          <div class="controls-grid">
            <button class="btn primary" data-action="start">Start AI</button>
            <button class="btn" data-action="stop">Stop AI</button>
            <button class="btn steel" data-action="center">Center Servo</button>
            <button class="btn warn" data-action="reset_stats" id="resetDashboardBtn">Reset Dashboard</button>
            <button class="btn" data-action="test_inorganic">Test Right</button>
            <button class="btn" data-action="test_organic">Test Left</button>
            <button class="btn steel" data-action="sweep">Sweep Servo</button>
          </div>
        </div>

        <div class="glass">
          <div class="panel-head">
            <h2 class="panel-title">System Settings</h2>
            <span class="panel-subtitle" id="settingsState">Auto-save</span>
          </div>
          <div class="connection-card">
            <div class="connection-row">
              <span id="espDot" class="dot"></span>
              <span id="espStatusText">Checking ESP32 connection</span>
            </div>
            <div class="connection-meta" id="espStatusMeta">Waiting for serial scan</div>
            <div class="connection-row">
              <span id="servoDot" class="dot"></span>
              <span id="servoStatusText">Checking servo PWM</span>
            </div>
            <div class="connection-meta" id="servoStatusMeta">Waiting for ESP32 diagnostic response</div>
          </div>
          <div class="settings-grid">
            <label>Model Backend
              <select id="modelBackendInput">
                <option value="transformers">Transformers</option>
                <option value="keras">Keras</option>
                <option value="fathima">Fathima</option>
              </select>
            </label>
            <label>Serial Port
              <input id="portInput" placeholder="COM5">
            </label>
            <label>Camera Source
              <select id="cameraSourceInput">
                <option value="local">Laptop / USB camera</option>
                <option value="droidcam">DroidCam HTTP over Wi-Fi</option>
                <option value="phone">Phone / IP camera</option>
              </select>
            </label>
            <label>Camera Index
              <input id="cameraIndexInput" type="number" min="0" max="20" step="1" value="0">
            </label>
            <label>Phone Camera URL / IP
              <input id="cameraUrlInput" placeholder="192.168.1.23">
            </label>
            <label>Min Confidence
              <input id="confidenceInput" type="number" min="0" max="1" step="0.05" value="0.35">
            </label>
            <label>Detect Interval (s)
              <input id="intervalInput" type="number" min="0.5" step="0.5" value="2">
            </label>
            <label>Cooldown (s)
              <input id="cooldownInput" type="number" min="0" step="0.5" value="3">
            </label>
            <label>Reset After (s)
              <input id="resetAfterInput" type="number" min="0" step="0.5" value="1.5">
            </label>
          </div>
          <div class="error" id="errorText"></div>
          <div class="panel-subtitle" style="padding:0 18px 18px;" id="serialResponseText">ESP32 response: waiting</div>
        </div>

        <div class="glass">
          <div class="panel-head">
            <h2 class="panel-title">Live Activity Log</h2>
            <span class="panel-subtitle" id="eventCount">0 events</span>
          </div>
          <div class="activity" id="history"></div>
        </div>
      </aside>
    </section>
  </main>

  <div class="toast" id="toast"></div>

  <div class="modal-backdrop" id="emptyModal">
    <div class="modal">
      <h3 id="modalTitle">Empty Bin</h3>
      <p id="modalMessage">Confirm that the bin has been emptied?</p>
      <div class="modal-actions">
        <button class="btn" id="cancelEmptyBtn">Cancel</button>
        <button class="btn primary" id="confirmEmptyBtn">Confirm</button>
      </div>
    </div>
  </div>

  <script>
    const $ = (id) => document.getElementById(id);
    const STORAGE_KEY = "smartbin.city.dashboard.v2";
    const FILL_STEP = 2;

    const emptyState = () => ({
      bins: {
        left: { fill: 0, lastDisposalAt: null },
        right: { fill: 0, lastDisposalAt: null }
      },
      activities: [],
      processedEvents: {},
      lastCriticalNotice: {}
    });

    let store = loadStore();
    let charts = {};
    let pendingEmptyBin = null;

    function loadStore() {
      try {
        const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
        if (!parsed || !parsed.bins || !parsed.activities || !parsed.processedEvents) return emptyState();
        return {
          ...emptyState(),
          ...parsed,
          bins: { ...emptyState().bins, ...parsed.bins }
        };
      } catch {
        return emptyState();
      }
    }

    function saveStore() {
      store.activities = store.activities.slice(0, 500);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
    }

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
        model_backend: $("modelBackendInput").value,
        port: $("portInput").value.trim(),
        camera_source: $("cameraSourceInput").value,
        camera_index: Number($("cameraIndexInput").value),
        camera_url: $("cameraUrlInput").value.trim(),
        min_confidence: Number($("confidenceInput").value),
        interval: Number($("intervalInput").value),
        cooldown: Number($("cooldownInput").value),
        reset_after: Number($("resetAfterInput").value)
      };
    }

    function updateCameraInputState() {
      const source = $("cameraSourceInput").value;
      const useRemoteCamera = source === "phone" || source === "droidcam";
      $("cameraIndexInput").disabled = useRemoteCamera;
      $("cameraUrlInput").disabled = !useRemoteCamera;
      $("cameraUrlInput").placeholder = source === "droidcam"
        ? "DroidCam Wi-Fi IP, e.g. 192.168.1.23"
        : useRemoteCamera
          ? "http://PHONE_IP:8080/video"
          : "Only used for phone camera sources";
    }

    function cameraSettingsReady() {
      const source = $("cameraSourceInput").value;
      if ((source === "phone" || source === "droidcam") && !$("cameraUrlInput").value.trim()) {
        setText("settingsState", "Enter phone IP first");
        setText("errorText", "Nhap IP/URL camera dien thoai truoc khi chuyen nguon camera.");
        return false;
      }
      return true;
    }

    async function sendAction(action) {
      if (action === "reset_stats") {
        store = emptyState();
        saveStore();
      }
      await postJson("/api/control", { action, ...settingsPayload() });
      await refresh();
    }

    function setText(id, value) {
      const node = $(id);
      if (node) node.textContent = value;
    }

    function clampPercent(value) {
      return Math.max(0, Math.min(100, Math.round(Number(value) || 0)));
    }

    function binLabel(bin) {
      return bin === "left" ? "Left Bin" : "Right Bin";
    }

    function binRoute(bin) {
      return bin === "left" ? "Left Bin (+2%)" : "Right Bin (+2%)";
    }

    function eventKey(event) {
      return event.id ? `server-${event.id}-${event.timestamp_iso || event.time}` : `${event.timestamp_iso || event.time}-${event.command}-${event.label}-${event.confidence}`;
    }

    function parseEventDate(event) {
      if (event.timestamp_iso) return new Date(event.timestamp_iso);
      const today = new Date().toISOString().slice(0, 10);
      return new Date(`${today}T${event.time || "00:00:00"}`);
    }

    function processServerEvents(events) {
      let changed = false;
      [...(events || [])].reverse().forEach((event) => {
        const key = eventKey(event);
        if (store.processedEvents[key]) return;
        store.processedEvents[key] = true;

        if (event.disposal_success === false) {
          changed = true;
          return;
        }

        const bin = event.bin || (event.command === "ORGANIC" ? "left" : "right");
        const timestamp = parseEventDate(event).toISOString();
        const fillDelta = Number(event.fill_delta || FILL_STEP);

        store.bins[bin].fill = clampPercent(store.bins[bin].fill + fillDelta);
        store.bins[bin].lastDisposalAt = timestamp;
        store.activities.unshift({
          id: key,
          type: "disposal",
          timestamp,
          bin,
          delta: fillDelta,
          label: event.display_label || event.label || "Waste item",
          category: event.category || "",
          command: event.command,
          confidence: Number(event.confidence || 0)
        });
        changed = true;
      });

      if (changed) saveStore();
    }

    function startOfToday() {
      const d = new Date();
      d.setHours(0, 0, 0, 0);
      return d;
    }

    function startOfWeek() {
      const d = startOfToday();
      const day = d.getDay() || 7;
      d.setDate(d.getDate() - day + 1);
      return d;
    }

    function startOfMonth() {
      const d = startOfToday();
      d.setDate(1);
      return d;
    }

    function disposalActivities() {
      return store.activities.filter((item) => item.type === "disposal");
    }

    function countSince(date, bin = null) {
      return disposalActivities().filter((item) => {
        const at = new Date(item.timestamp);
        return at >= date && (!bin || item.bin === bin);
      }).length;
    }

    function formatTime(value) {
      if (!value) return "never";
      return new Intl.DateTimeFormat([], { hour: "2-digit", minute: "2-digit" }).format(new Date(value));
    }

    function statusFor(fill) {
      if (fill > 90) return { label: "Critical", cls: "critical", color: "#ef4444" };
      if (fill >= 81) return { label: "Nearly Full", cls: "nearly", color: "#f97316" };
      if (fill >= 51) return { label: "Filling", cls: "filling", color: "#eab308" };
      return { label: "Normal", cls: "normal", color: "#22c55e" };
    }

    function updateBin(bin) {
      const fill = clampPercent(store.bins[bin].fill);
      const meta = statusFor(fill);
      const capital = bin === "left" ? "left" : "right";
      const card = $(`${capital}BinCard`);
      const status = $(`${capital}Status`);
      const liquid = $(`${capital}Liquid`);

      setText(`${capital}BinPct`, `${fill}%`);
      setText(`${capital}FillHero`, `${fill}%`);
      setText(`${capital}EventsToday`, `${countSince(startOfToday(), bin)} events today`);
      setText(`${capital}TodayHero`, `${countSince(startOfToday(), bin)} events today`);
      setText(`${capital}LastTime`, `Last disposal: ${formatTime(store.bins[bin].lastDisposalAt)}`);

      liquid.style.setProperty("--level", `${fill}%`);
      liquid.style.setProperty("--liquid", meta.color);
      status.textContent = meta.label;
      status.className = `status-badge ${meta.cls}`;
      card.classList.toggle("critical", fill > 90);

      document.querySelectorAll(`[data-empty-bin="${bin}"]`).forEach((btn) => {
        btn.classList.toggle("visible", fill > 90);
      });
    }

    function updateAlert() {
      const leftFill = clampPercent(store.bins.left.fill);
      const rightFill = clampPercent(store.bins.right.fill);
      const criticalBin = leftFill > 90 ? "left" : rightFill > 90 ? "right" : null;
      const banner = $("alertBanner");

      if (!criticalBin) {
        banner.classList.remove("visible");
        return;
      }

      const fill = clampPercent(store.bins[criticalBin].fill);
      const message = `${binLabel(criticalBin)} has reached ${fill}%. Please empty the bin immediately.`;
      $("alertMessage").textContent = message;
      $("alertEmptyBtn").onclick = () => openEmptyModal(criticalBin);
      banner.classList.add("visible");

      if (store.lastCriticalNotice[criticalBin] !== fill) {
        showToast(`<strong>🚨 BIN NEAR FULL</strong><br>${message}`);
        store.lastCriticalNotice[criticalBin] = fill;
        saveStore();
      }
    }

    function showToast(html) {
      const toast = $("toast");
      toast.innerHTML = html;
      toast.classList.add("show");
      clearTimeout(showToast.timer);
      showToast.timer = setTimeout(() => toast.classList.remove("show"), 4200);
    }

    function openEmptyModal(bin) {
      pendingEmptyBin = bin;
      setText("modalTitle", `Empty ${binLabel(bin)}`);
      setText("modalMessage", "Confirm that the bin has been emptied?");
      $("emptyModal").classList.add("visible");
    }

    function closeEmptyModal() {
      pendingEmptyBin = null;
      $("emptyModal").classList.remove("visible");
    }

    function confirmEmptyBin() {
      if (!pendingEmptyBin) return;
      const bin = pendingEmptyBin;
      store.bins[bin].fill = 0;
      store.lastCriticalNotice[bin] = 0;
      store.activities.unshift({
        id: `empty-${Date.now()}-${bin}`,
        type: "empty",
        timestamp: new Date().toISOString(),
        bin,
        label: `${binLabel(bin)} emptied`,
        delta: 0
      });
      saveStore();
      closeEmptyModal();
      renderDashboard();
      showToast("✅ Bin successfully emptied");
    }

    function renderActivity() {
      const holder = $("history");
      const activities = store.activities.slice(0, 40);
      holder.innerHTML = "";
      setText("eventCount", `${activities.length} events`);

      if (!activities.length) {
        holder.innerHTML = '<div class="muted">No disposal events yet. The activity stream will update in real time.</div>';
        return;
      }

      for (const item of activities) {
        const row = document.createElement("div");
        row.className = `event ${item.type === "empty" ? "empty" : item.bin}`;
        const time = formatTime(item.timestamp);
        const route = item.type === "empty" ? "Maintenance workflow completed" : `→ ${binRoute(item.bin)}`;
        const icon = item.type === "empty" ? "✓" : item.bin === "left" ? "A" : "B";
        row.innerHTML = `
          <div class="event-icon">${icon}</div>
          <div class="event-main">
            <div class="event-title">${item.label || "Waste classified"}</div>
            <div class="event-sub">${route}${item.confidence ? ` · ${Math.round(item.confidence * 100)}% confidence` : ""}</div>
          </div>
          <div class="muted">${time}</div>
        `;
        holder.appendChild(row);
      }
    }

    function trendLabels(days = 7) {
      return Array.from({ length: days }, (_, index) => {
        const d = new Date();
        d.setDate(d.getDate() - (days - 1 - index));
        d.setHours(0, 0, 0, 0);
        return d;
      });
    }

    function sameDate(a, b) {
      return a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();
    }

    function renderCharts() {
      if (!window.Chart) return;
      const css = getComputedStyle(document.documentElement);
      const grid = "rgba(255,255,255,0.08)";
      const text = css.getPropertyValue("--muted").trim();
      const events = disposalActivities();
      const days = trendLabels(7);
      const trend = days.map((day) => events.filter((item) => sameDate(new Date(item.timestamp), day)).length);
      const leftCount = events.filter((item) => item.bin === "left").length;
      const rightCount = events.filter((item) => item.bin === "right").length;
      const weekStart = startOfWeek();
      const weekly = Array.from({ length: 7 }, (_, index) => {
        const day = new Date(weekStart);
        day.setDate(day.getDate() + index);
        return events.filter((item) => sameDate(new Date(item.timestamp), day)).length;
      });

      Chart.defaults.color = text;
      Chart.defaults.font.family = "Inter";

      const lineData = {
        labels: days.map((d) => d.toLocaleDateString([], { weekday: "short" })),
        datasets: [{
          label: "Disposals",
          data: trend,
          borderColor: "#22d3ee",
          backgroundColor: "rgba(34,211,238,0.18)",
          pointBackgroundColor: "#fafafa",
          pointRadius: 4,
          fill: true,
          tension: 0.42
        }]
      };

      const doughnutData = {
        labels: ["Left Bin", "Right Bin"],
        datasets: [{
          data: [leftCount, rightCount],
          backgroundColor: ["#22c55e", "#60a5fa"],
          borderColor: "rgba(9,9,11,0.85)",
          borderWidth: 4,
          hoverOffset: 8
        }]
      };

      const barData = {
        labels: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        datasets: [{
          label: "Weekly Count",
          data: weekly,
          backgroundColor: "rgba(167,139,250,0.78)",
          borderRadius: 12,
          borderSkipped: false
        }]
      };

      upsertChart("dailyTrendChart", "line", lineData, {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: grid } },
          y: { beginAtZero: true, ticks: { precision: 0 }, grid: { color: grid } }
        }
      });

      upsertChart("usageChart", "doughnut", doughnutData, {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "68%",
        plugins: { legend: { position: "bottom" } }
      });

      upsertChart("weeklyChart", "bar", barData, {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false } },
          y: { beginAtZero: true, ticks: { precision: 0 }, grid: { color: grid } }
        }
      });
    }

    function upsertChart(id, type, data, options) {
      if (!charts[id]) {
        charts[id] = new Chart($(id), { type, data, options });
        return;
      }
      charts[id].data = data;
      charts[id].options = options;
      charts[id].update("none");
    }

    function renderDashboard(status = null) {
      updateBin("left");
      updateBin("right");

      const today = countSince(startOfToday());
      const week = countSince(startOfWeek());
      const month = countSince(startOfMonth());
      const leftToday = countSince(startOfToday(), "left");
      const rightToday = countSince(startOfToday(), "right");
      const critical = store.bins.left.fill > 90 || store.bins.right.fill > 90;

      setText("todayTotal", today);
      setText("todayNote", `${leftToday} left · ${rightToday} right`);
      setText("statToday", today);
      setText("statWeek", week);
      setText("statMonth", month);

      if (status) {
        const system = critical ? "Critical" : status.model_ready && status.camera_active ? "Online" : "Loading";
        setText("systemStatus", system);
        setText("systemNote", status.error || (critical ? "Collection required" : "All services nominal"));
      }

      updateAlert();
      renderActivity();
      renderCharts();
    }

    async function refresh() {
      const status = await fetch("/api/status").then((r) => r.json());
      const result = status.last_result || {};
      const pct = Math.round((result.confidence || 0) * 100);
      const serialConnected = Boolean(status.serial_connected);
      const serialPorts = status.available_serial_ports || [];
      const servoPwmReady = Boolean(status.servo_pwm_ready);

      processServerEvents(status.history || []);

      $("runDot").classList.toggle("on", status.detection_enabled);
      setText("runText", status.detection_enabled ? "AI running" : "AI stopped");
      setText(
        "cameraText",
        status.camera_source === "phone" || status.camera_source === "droidcam"
          ? `${status.camera_source === "droidcam" ? "DroidCam" : "Phone camera"}${status.camera_resolved_url ? ` · ${status.camera_resolved_url}` : ""}`
          : `Camera index ${status.camera_index}`
      );
      const modelLabel = status.model_backend || "transformers";
      setText("modelText", status.model_ready ? `${modelLabel} ready` : `${modelLabel} loading`);
      setText("updatedText", status.updated_at || "Waiting");
      setText("servoCommand", status.last_servo_command || "CENTER");
      setText("categoryText", result.final_category || "No result");
      setText("labelText", result.original_label ? `${result.display_label || result.original_label} · Model label: ${result.original_label}` : "Point waste at the camera");
      setText("confidenceDial", `${pct}%`);
      setText("confidenceText", `${pct}% confidence`);
      $("confidenceDial").style.setProperty("--pct", `${pct}%`);
      setText("serialText", serialConnected ? `ESP32 ${status.serial_port}` : "ESP32 disconnected");
      setText("serialResponseText", `ESP32 response: ${status.last_serial_response || "waiting"}`);
      setText("errorText", status.error || "");
      setText("hazardCount", status.stats?.hazardous || 0);
      $("espDot").classList.toggle("on", serialConnected);
      $("espDot").classList.toggle("off", !serialConnected);
      setText("espStatusText", status.serial_status_text || (serialConnected ? "ESP32 connected" : "ESP32 disconnected"));
      setText(
        "espStatusMeta",
        serialPorts.length
          ? `Selected: ${status.serial_port || "none"} | Available: ${serialPorts.join(", ")}`
          : `Selected: ${status.serial_port || "none"} | No serial devices detected`
      );
      $("servoDot").classList.toggle("on", servoPwmReady);
      $("servoDot").classList.toggle("off", !servoPwmReady);
      setText("servoStatusText", status.servo_status_text || "Servo PWM status unknown");
      setText(
        "servoStatusMeta",
        status.servo_status_response || "Note: a 3-wire servo has no physical plug-in feedback without extra hardware."
      );

      if (document.activeElement !== $("portInput")) $("portInput").value = status.serial_port || "";
      if (document.activeElement !== $("modelBackendInput")) $("modelBackendInput").value = status.model_backend || "transformers";
      if (document.activeElement !== $("cameraSourceInput")) $("cameraSourceInput").value = status.camera_source || "local";
      if (document.activeElement !== $("cameraIndexInput")) $("cameraIndexInput").value = status.camera_index ?? 0;
      if (document.activeElement !== $("cameraUrlInput")) $("cameraUrlInput").value = status.camera_url || "";
      updateCameraInputState();
      if (document.activeElement !== $("confidenceInput")) $("confidenceInput").value = status.min_confidence;
      if (document.activeElement !== $("intervalInput")) $("intervalInput").value = status.interval;
      if (document.activeElement !== $("cooldownInput")) $("cooldownInput").value = status.cooldown;
      if (document.activeElement !== $("resetAfterInput")) $("resetAfterInput").value = status.reset_after ?? 1.5;

      renderDashboard(status);
    }

    document.querySelectorAll("[data-action]").forEach((button) => {
      button.addEventListener("click", () => sendAction(button.dataset.action));
    });

    document.querySelectorAll("[data-empty-bin]").forEach((button) => {
      button.addEventListener("click", () => openEmptyModal(button.dataset.emptyBin));
    });

    $("alertEmptyBtn").addEventListener("click", () => {
      if (store.bins.left.fill > 90) openEmptyModal("left");
      else if (store.bins.right.fill > 90) openEmptyModal("right");
    });

    $("cancelEmptyBtn").addEventListener("click", closeEmptyModal);
    $("confirmEmptyBtn").addEventListener("click", confirmEmptyBin);
    $("emptyModal").addEventListener("click", (event) => {
      if (event.target === $("emptyModal")) closeEmptyModal();
    });

    ["modelBackendInput", "portInput", "cameraSourceInput", "cameraIndexInput", "cameraUrlInput", "confidenceInput", "intervalInput", "cooldownInput", "resetAfterInput"].forEach((id) => {
      $(id).addEventListener("change", async () => {
        updateCameraInputState();
        if (!cameraSettingsReady()) return;
        setText("settingsState", "Saving...");
        await postJson("/api/settings", settingsPayload());
        setText("settingsState", "Auto-save");
        await refresh();
      });
    });

    renderDashboard();
    updateCameraInputState();
    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""


class SmartBinRuntime:
    MODEL_BACKENDS = {"transformers", "keras", "fathima"}
    CAMERA_SOURCES = {"local", "phone", "droidcam"}

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.serial_lock = threading.Lock()
        self.processor: Any = None
        self.model: Any = None
        self.model_backend = "transformers"
        self.model_ready = False
        self.model_loading = False
        self.camera_source = "local"
        self.camera_index = 0
        self.camera_url = ""
        self.camera_active = True
        self.detection_enabled = True
        self.serial_enabled = True
        self.serial_port = auto_detect_port()
        self.baudrate = 115200
        self.interval = 2.0
        self.cooldown = 3.0
        self.min_confidence = 0.35
        self.reset_after = 1.5
        self.last_detection_at = 0.0
        self.last_counted_at = 0.0
        self.last_sent_at = 0.0
        self.last_servo_command = "CENTER"
        self.last_serial_response = ""
        self.servo_pwm_ready = False
        self.servo_status_text = "Servo PWM status unknown"
        self.servo_status_response = "Waiting for ESP32 diagnostic response"
        self.servo_status_checked_at = 0.0
        self.last_result: dict[str, Any] | None = None
        self.pending_command = ""
        self.pending_count = 0
        self.stable_frames_required = 2
        self.updated_at = ""
        self.error = ""
        self.stats = {"total": 0, "organic": 0, "inorganic": 0, "hazardous": 0}
        self.event_seq = 0
        self.history: deque[dict[str, Any]] = deque(maxlen=120)
        self.frame_jpeg: bytes | None = None
        self.latest_frame: Any = None
        self.stop_event = threading.Event()

    def start(self) -> None:
        logging.info("Starting SmartBin runtime")
        self._start_model_load()
        threading.Thread(target=self._camera_loop, daemon=True).start()
        threading.Thread(target=self._detection_loop, daemon=True).start()

    def _start_model_load(self) -> None:
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self) -> None:
        with self.lock:
            backend_name = self.model_backend
            self.model_loading = True
            self.model_ready = False
            self.processor = None
            self.model = None
            self.last_result = None
            self.pending_command = ""
            self.pending_count = 0
            self.error = ""

        logging.info("Loading model backend=%s", backend_name)
        try:
            backend, processor, model = load_selected_model(backend_name)
            with self.lock:
                self.model_backend = backend
                self.processor = processor
                self.model = model
                self.model_ready = True
                self.model_loading = False
            logging.info("Model loaded backend=%s", backend)
        except Exception as exc:
            logging.exception("Model failed to load")
            with self.lock:
                self.error = str(exc)
                self.model_loading = False

    def set_model_backend(self, requested_backend: str) -> None:
        backend = str(requested_backend or "").strip().lower()
        if backend not in self.MODEL_BACKENDS:
            raise RuntimeError(f"Unsupported model backend: {requested_backend}")

        with self.lock:
            if backend == self.model_backend and (self.model_ready or self.model_loading):
                return
            self.model_backend = backend

        self._start_model_load()

    def _camera_loop(self) -> None:
        if cv2 is None:
            logging.error("opencv-python is missing")
            with self.lock:
                self.error = "opencv-python is missing. Run: python -m pip install -r requirements.txt"
            return

        camera = None
        open_signature = None

        while not self.stop_event.is_set():
            with self.lock:
                camera_source = self.camera_source
                camera_index = self.camera_index
                camera_url = self._normalized_camera_url(self.camera_source, self.camera_url)
                camera_active = self.camera_active

            if not camera_active:
                time.sleep(0.2)
                continue

            source_signature = (
                camera_source,
                camera_url if camera_source in {"phone", "droidcam"} else camera_index,
            )
            if camera is None or open_signature != source_signature:
                if camera is not None:
                    camera.release()

                if camera_source in {"phone", "droidcam"}:
                    if not camera_url:
                        source_name = "DroidCam" if camera_source == "droidcam" else "Phone camera"
                        with self.lock:
                            self.error = f"{source_name} URL is empty. Example: http://PHONE_IP:4747/video"
                        time.sleep(1)
                        continue
                    logging.info("Opening %s camera URL %s", camera_source, camera_url)
                    camera = cv2.VideoCapture(camera_url)
                else:
                    logging.info("Opening camera index %s", camera_index)
                    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera.set(cv2.CAP_PROP_FPS, 30)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                open_signature = source_signature
                time.sleep(0.4)

            if not camera.isOpened():
                source_label = camera_url if camera_source in {"phone", "droidcam"} else f"index {camera_index}"
                logging.error("Camera %s is not available", source_label)
                with self.lock:
                    self.error = f"Camera {source_label} is not available"
                time.sleep(1)
                continue

            ok, frame = camera.read()
            if not ok:
                source_label = camera_url if camera_source in {"phone", "droidcam"} else f"index {camera_index}"
                logging.error("Could not read camera frame from %s", source_label)
                with self.lock:
                    self.error = f"Could not read camera frame from {source_label}"
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
            disposal_success = (not serial_enabled) or sent
            event_time = datetime.now()
            bin_side = "right" if command == SERVO_INORGANIC_COMMAND else "left"
            self.event_seq += 1
            self.last_counted_at = time.monotonic()
            self.last_servo_command = command
            self.last_serial_response = " | ".join(responses) if responses else self.last_serial_response

            if disposal_success:
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
                    "id": self.event_seq,
                    "time": event_time.strftime("%H:%M:%S"),
                    "timestamp_iso": event_time.isoformat(timespec="seconds"),
                    "command": command,
                    "bin": bin_side,
                    "fill_delta": 2,
                    "label": result["original_label"],
                    "display_label": result.get("display_label", result["original_label"]),
                    "category": result["final_category"],
                    "confidence": result["confidence"],
                    "serial_sent": sent,
                    "disposal_success": disposal_success,
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

    @staticmethod
    def _normalized_camera_url(camera_source: str, camera_url: str) -> str:
        raw_url = str(camera_url or "").strip()
        if not raw_url:
            return ""

        if camera_source == "droidcam":
            if raw_url.startswith(("http://", "https://", "rtsp://")):
                normalized = raw_url
            else:
                host = raw_url.strip("/")
                if ":" not in host:
                    host = f"{host}:4747"
                normalized = f"http://{host}"

            lowered = normalized.lower().rstrip("/")
            if lowered.startswith(("http://", "https://")) and not lowered.endswith(("/video", "/mjpegfeed")):
                normalized = f"{normalized.rstrip('/')}/video"
            return normalized

        if raw_url.startswith(("http://", "https://", "rtsp://")):
            return raw_url
        return f"http://{raw_url}"

    def update_settings(self, data: dict[str, Any]) -> None:
        requested_backend = data.get("model_backend")
        if requested_backend is not None:
            self.set_model_backend(str(requested_backend))

        with self.lock:
            port = str(data.get("port") or "").strip()
            camera_source = str(data.get("camera_source") or self.camera_source).strip().lower()
            if camera_source not in self.CAMERA_SOURCES:
                camera_source = "local"
            camera_url = str(data.get("camera_url") or "").strip()

            if camera_source in {"phone", "droidcam"} and not camera_url:
                self.error = "Nhap IP/URL camera dien thoai truoc khi chuyen sang nguon camera dien thoai."
                camera_source = self.camera_source
                camera_url = self.camera_url

            self.serial_port = port or auto_detect_port()
            self.camera_source = camera_source
            self.camera_url = camera_url
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

    @staticmethod
    def _available_serial_ports() -> list[str]:
        if list_ports is None:
            return []

        return [port.device for port in list_ports.comports()]

    def _serial_status(self, serial_port: str | None, serial_enabled: bool) -> tuple[bool, str, list[str]]:
        available_ports = self._available_serial_ports()

        if not serial_enabled:
            return False, "ESP32 serial disabled", available_ports

        if not serial_port:
            return False, "No ESP32 port selected", available_ports

        connected = serial_port.lower() in {port.lower() for port in available_ports}
        if connected:
            return True, f"ESP32 connected on {serial_port}", available_ports

        return False, f"ESP32 not found on {serial_port}", available_ports

    def _refresh_servo_status(self, serial_port: str | None, serial_connected: bool) -> None:
        now = time.monotonic()

        with self.lock:
            stale = now - self.servo_status_checked_at >= 10
            if not stale:
                return
            self.servo_status_checked_at = now

        if not serial_connected or not serial_port:
            with self.lock:
                self.servo_pwm_ready = False
                self.servo_status_text = "Servo PWM not checked"
                self.servo_status_response = "ESP32 is not connected, so servo PWM cannot be checked."
            return

        try:
            with self.serial_lock:
                responses = send_command_to_esp32(
                    serial_port,
                    "PIN",
                    self.baudrate,
                    reset_after=0,
                    response_wait=0.45,
                    boot_wait=0,
                )

            response_text = " | ".join(responses)
            pwm_ready = "Servo signal pins:" in response_text
            with self.lock:
                self.servo_pwm_ready = pwm_ready
                self.servo_status_text = (
                    "Servo PWM ready"
                    if pwm_ready
                    else "Servo PWM response not recognized"
                )
                self.servo_status_response = (
                    response_text
                    if response_text
                    else "No diagnostic response from ESP32."
                )
        except Exception as exc:
            with self.lock:
                self.servo_pwm_ready = False
                self.servo_status_text = "Servo PWM check failed"
                self.servo_status_response = str(exc)

    def status(self) -> dict[str, Any]:
        with self.lock:
            serial_enabled = self.serial_enabled
            serial_port = self.serial_port
            status = {
                "model_ready": self.model_ready,
                "model_loading": self.model_loading,
                "model_backend": self.model_backend,
                "camera_active": self.camera_active,
                "camera_source": self.camera_source,
                "camera_index": self.camera_index,
                "camera_url": self.camera_url,
                "camera_resolved_url": self._normalized_camera_url(self.camera_source, self.camera_url),
                "detection_enabled": self.detection_enabled,
                "serial_enabled": serial_enabled,
                "serial_port": serial_port,
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
                "reset_after": self.reset_after,
            }

        serial_connected, serial_status_text, available_serial_ports = self._serial_status(
            serial_port,
            serial_enabled,
        )
        self._refresh_servo_status(serial_port, serial_connected)
        with self.lock:
            servo_pwm_ready = self.servo_pwm_ready
            servo_status_text = self.servo_status_text
            servo_status_response = self.servo_status_response

        status["serial_connected"] = serial_connected
        status["serial_status_text"] = serial_status_text
        status["available_serial_ports"] = available_serial_ports
        status["servo_pwm_ready"] = servo_pwm_ready
        status["servo_status_text"] = servo_status_text
        status["servo_status_response"] = servo_status_response
        return status

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
    parser.add_argument("--camera-source", choices=["local", "phone", "droidcam"], default="local")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-url", default="")
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
        "Starting Flask server host=%s port=%s camera_source=%s camera_index=%s camera_url=%s serial_port=%s serial_enabled=%s",
        args.host,
        args.port,
        args.camera_source,
        args.camera_index,
        args.camera_url,
        args.serial_port,
        not args.no_serial,
    )
    runtime.model_backend = args.model_backend
    runtime.camera_source = args.camera_source
    runtime.camera_index = args.camera_index
    runtime.camera_url = args.camera_url
    runtime.serial_enabled = not args.no_serial
    runtime.serial_port = args.serial_port or runtime.serial_port
    runtime.start()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
