# Project Rio — System Architecture

## Layered Architecture (Industry Standard)

```
RIO/
├── main.py                    Entry point. Zero business logic.
├── .env.example               Environment variable template
├── requirements.txt
├── setup.py
│
├── config/
│   └── settings.py            All config — reads from .env
│
├── app/
│   ├── api/                   ── Controller Layer ──
│   │   ├── server.py          Flask application factory
│   │   ├── dashboard.py       GET / → renders live dashboard
│   │   ├── status.py          GET /api/status + /api/events
│   │   ├── ingest.py          POST /frame (ESP32-CAM receiver)
│   │   └── stream.py          GET /stream (MJPEG generator)
│   │
│   ├── core/                  ── Service Layer ──
│   │   ├── analyzer.py        YOLOv8 detection, pose, SDP blur
│   │   ├── alert_manager.py   Cooldown, clip recording, dispatch
│   │   ├── fps_controller.py  Adaptive MONITOR ↔ EVIDENCE
│   │   ├── capture.py         Unified frame source
│   │   └── hud.py             OpenCV HUD overlay
│   │
│   ├── models/                ── Model Layer ──
│   │   ├── app_state.py       Thread-safe live system state
│   │   ├── detection.py       Detection dataclass
│   │   └── threat.py          ThreatLevel enum
│   │
│   ├── services/              ── Infrastructure Layer ──
│   │   ├── notifier.py        AbstractNotifier (interface)
│   │   ├── telegram_notifier.py  Telegram alert + video
│   │   └── ngrok_tunnel.py    Public tunnel manager
│   │
│   └── templates/
│       └── dashboard.html     Live web dashboard
│
├── firmware/
│   └── esp32_cam.ino          HC-SR04 + camera + HTTP POST
│
├── utils/
│   └── logger.py              Centralised logging
│
├── docs/                      Architecture, pitch script
├── scripts/                   Shell helpers
└── tests/                     Unit test suite
```

## Data Flow

```
[HC-SR04 Proximity Trigger]
        │ < 150cm detected
        ▼
  ESP32-CAM (captures JPEG)
        │ HTTP POST /frame
        ▼
  app/api/ingest.py  ───────────┐
                                │
  webcam / video file  ─────────┤
                                ▼
                     app/core/capture.py
                                │ frame
                                ▼
                     app/core/analyzer.py
                      ├─ YOLOv8 detect
                      ├─ YOLOv8 pose → masked face?
                      ├─ Loitering timer
                      ├─ SDP blur (LOW/NONE threats)
                      └─ Detection[]
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
   fps_controller.py    alert_manager.py   app/models/app_state.py
   MONITOR ↔ EVIDENCE   cooldown check     thread-safe write
                        clip save
                        Telegram dispatch
                                            │
                               ┌────────────┴───────────┐
                               ▼                        ▼
                    app/api/status.py          app/api/stream.py
                    /api/status (JSON)          /stream (MJPEG)
                               │                        │
                               └────────────┬───────────┘
                                            ▼
                                 app/templates/dashboard.html
```

## Threat Classification Logic

| Condition | Level | Action |
|-----------|-------|--------|
| Masked face OR loitering > 3s | HIGH | Alert + clip + Telegram |
| Confidence > 75% | MEDIUM | Log only |
| Confidence 50–75% | LOW | SDP blur applied |
| Non-person class | IGNORE | Skip |

## Hardware BOM

| Component | Cost |
|-----------|------|
| ESP32-CAM | ₹350 |
| HC-SR04 Ultrasonic | ₹50 |
| FTDI USB-Serial | ₹200 |
| Jumper wires | ₹120 |
| **Total** | **₹720** |
| Commercial equivalent | ₹50,000+ |
