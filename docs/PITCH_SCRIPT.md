# 🎤 Project Rio — Judge Pitch Script
### HackArena '26 | Team: Pirate2Pirate

---

## THE OPENING LINE (Money Heist hook)
> *"In Money Heist, Rio was the hacker — the guy who controlled every camera,
> every feed, every blind spot. Today, we flipped that.
> We built the system that catches the Rio."*

Pause. Let it land. Then continue.

---

## THE PROBLEM (30 seconds)

> *"Every CCTV system today has the same three problems —
> it records 24/7 wasting power, it has zero privacy recording everyone including
> your own family, and it spams you with alerts every time a cat walks past.
> Commercial AI solutions to fix this? ₹50,000 minimum.
> We built the same thing for ₹720."*

---

## THE DEMO — 5 WOW MOMENTS (in order)

---

### 🎭 WOW #1 — Masked Face Alert (do this FIRST)
**Setup:** Have Telegram open on your phone visible to judges.

> *"Watch what happens when I cover my face."*

→ Cover face with paper / mask.
→ Within 2–3 seconds: 🚨 Telegram alert fires on your phone.
→ Hold up phone to judges.

> *"That just sent a 5-second video clip to my phone.
> Automatically. No button pressed."*

---

### 🌫️ WOW #2 — SDP Face Blur (right after)
**Setup:** Uncover face. Stand normally in frame.

→ Point to the dashboard stream.
→ Show the cyan **"SDP"** label on the bounding box.
→ Show the purple **"SDP ACTIVE 🎭"** badge in the sidebar.

> *"The moment it recognises me as non-threatening,
> it blurs my face automatically. We call this
> Software Defined Privacy — your family's faces
> never get stored in clear. Only threats do."*

---

### ⏱️ WOW #3 — Intelligence Buffer (the 5-walks demo)
**Setup:** Reset cooldown by restarting, or wait 60s.

> *"Now watch this. I'm going to walk past the camera 5 times."*

→ Walk past 5 times in quick succession.
→ Point to the orange countdown bar on dashboard: **"Intelligence Buffer: 58s..."**

> *"Five walks. One alert. The system understood
> that after the first alert, the same person walking
> past is not a new threat. 60-second intelligence buffer.
> No spam. No cry-wolf fatigue."*

---

### ⚡ WOW #4 — Adaptive FPS (point to dashboard)

→ Trigger a HIGH threat again.
→ Point to FPS badge switching: **MONITOR @ 5fps → EVIDENCE @ 20fps**

> *"The moment a real threat is detected, the system
> jumps from 5 frames per second to 20 — 4x more detail
> exactly when you need it. When nothing is happening,
> it drops back down. 70% storage saved."*

---

### 💰 WOW #5 — Cost Comparison (point to dashboard panel)

→ Point to the cost comparison card on the dashboard.

> *"Bottom right of the screen.
> Commercial enterprise AI CCTV: ₹50,000 plus.
> Project Rio: ₹720 total.
> 98.6% cheaper. And unlike those systems —
> ours protects your privacy by default."*

---

## THE CLOSE

> *"Three things no commercial system does together:
> active-on-demand power saving, real-time privacy protection,
> and behavioural AI — not just motion detection.
> Project Rio. The eyes that never blink."*

---

## IF JUDGES ASK QUESTIONS

**"How is this different from just using YOLOv8?"**
> *"YOLOv8 alone just draws boxes. We added a 3-tier architecture —
> hardware-level ultrasonic trigger, skeletal pose analysis for
> behavioural classification, and Software Defined Privacy.
> The AI isn't just seeing. It's thinking."*

**"What if Wi-Fi goes down?"**
> *"The ESP32-CAM has a MicroSD slot. We use it as a failsafe buffer —
> frames write locally if the hub is unreachable."*

**"Can it scale to multiple cameras?"**
> *"Yes — each ESP32-CAM posts to the same Flask hub endpoint.
> The hub handles multiple streams concurrently.
> And each unit costs ₹600."*

**"Why HC-SR04 instead of PIR?"**
> *"PIR detects heat — it fires for animals, moving shadows, even sunlight.
> Ultrasonic measures exact distance. We get precise zone control —
> 150cm trigger means inside the doorway only, not the street outside."*

---

## TEAM ROLES (mention naturally)
- Hardware wiring + ESP32 firmware → [teammate name]
- Python AI hub + YOLOv8 pipeline → [teammate name]
- Dashboard + Telegram integration → [teammate name]

---

*"The heist is over. Rio is watching."* 🔴
