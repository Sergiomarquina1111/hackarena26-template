// ═══════════════════════════════════════════════════════
//  ThreatSense AI-DVR  |  threatsense_esp32cam.ino
//  Hardware : ESP32-CAM (AI-Thinker) + HC-SR04
//  Power    : Laptop USB → FTDI → ESP32-CAM 5V
// ═══════════════════════════════════════════════════════

#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>

// ── CONFIG — edit these 4 lines only ────────────────────
const char* WIFI_SSID          = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD      = "YOUR_WIFI_PASSWORD";
const char* HUB_URL            = "http://192.168.1.100:5000/frame";
const int   TRIGGER_DISTANCE_CM = 150;
// ────────────────────────────────────────────────────────

const int   CONFIRM_READINGS   = 2;
const unsigned long IDLE_TIMEOUT_MS = 10000;

// HC-SR04 pins
#define TRIG_PIN  12
#define ECHO_PIN  13

// AI-Thinker camera pins (do not change)
#define PWDN_GPIO_NUM   32
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM    0
#define SIOD_GPIO_NUM   26
#define SIOC_GPIO_NUM   27
#define Y9_GPIO_NUM     35
#define Y8_GPIO_NUM     34
#define Y7_GPIO_NUM     39
#define Y6_GPIO_NUM     36
#define Y5_GPIO_NUM     21
#define Y4_GPIO_NUM     19
#define Y3_GPIO_NUM     18
#define Y2_GPIO_NUM      5
#define VSYNC_GPIO_NUM  25
#define HREF_GPIO_NUM   23
#define PCLK_GPIO_NUM   22

bool cameraActive    = false;
int  consecutiveHits = 0;
unsigned long lastTriggerMs = 0;

// ── Camera init ──────────────────────────────────────────
bool initCamera() {
  camera_config_t cfg;
  cfg.ledc_channel = LEDC_CHANNEL_0; cfg.ledc_timer = LEDC_TIMER_0;
  cfg.pin_d0=Y2_GPIO_NUM; cfg.pin_d1=Y3_GPIO_NUM; cfg.pin_d2=Y4_GPIO_NUM;
  cfg.pin_d3=Y5_GPIO_NUM; cfg.pin_d4=Y6_GPIO_NUM; cfg.pin_d5=Y7_GPIO_NUM;
  cfg.pin_d6=Y8_GPIO_NUM; cfg.pin_d7=Y9_GPIO_NUM;
  cfg.pin_xclk=XCLK_GPIO_NUM; cfg.pin_pclk=PCLK_GPIO_NUM;
  cfg.pin_vsync=VSYNC_GPIO_NUM; cfg.pin_href=HREF_GPIO_NUM;
  cfg.pin_sscb_sda=SIOD_GPIO_NUM; cfg.pin_sscb_scl=SIOC_GPIO_NUM;
  cfg.pin_pwdn=PWDN_GPIO_NUM; cfg.pin_reset=RESET_GPIO_NUM;
  cfg.xclk_freq_hz=20000000; cfg.pixel_format=PIXFORMAT_JPEG;
  if (psramFound()) {
    cfg.frame_size=FRAMESIZE_VGA; cfg.jpeg_quality=12; cfg.fb_count=2;
  } else {
    cfg.frame_size=FRAMESIZE_QVGA; cfg.jpeg_quality=20; cfg.fb_count=1;
  }
  esp_err_t err = esp_camera_init(&cfg);
  if (err!=ESP_OK){ Serial.printf("[CAM] init failed 0x%x\n",err); return false; }
  Serial.println("[CAM] OK");
  return true;
}

// ── HC-SR04 ──────────────────────────────────────────────
long distanceCm() {
  digitalWrite(TRIG_PIN, LOW);  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH); delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long d = pulseIn(ECHO_PIN, HIGH, 30000);
  return d==0 ? 999 : d/58L;
}

// ── Wi-Fi ────────────────────────────────────────────────
void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("[WiFi] Connecting to %s", WIFI_SSID);
  for (int i=0; i<20 && WiFi.status()!=WL_CONNECTED; i++) {
    delay(500); Serial.print(".");
  }
  if (WiFi.status()==WL_CONNECTED)
    Serial.printf("\n[WiFi] Connected: %s\n", WiFi.localIP().toString().c_str());
  else
    Serial.println("\n[WiFi] FAILED — check credentials");
}

// ── Capture & POST ───────────────────────────────────────
void captureAndPost() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) { Serial.println("[CAM] capture failed"); return; }

  if (WiFi.status()!=WL_CONNECTED) {
    esp_camera_fb_return(fb); return;
  }

  HTTPClient http;
  http.begin(HUB_URL);
  http.addHeader("Content-Type","image/jpeg");
  http.addHeader("X-Distance-CM", String(distanceCm()));

  int code = http.POST(fb->buf, fb->len);
  Serial.printf("[HTTP] %d  (%u bytes)\n", code, fb->len);
  http.end();
  esp_camera_fb_return(fb);
}

// ── Setup ────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Serial.println("\n[BOOT] ThreatSense ESP32-CAM");
  pinMode(TRIG_PIN, OUTPUT); pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);
  if (!initCamera()) { while(true) delay(1000); }
  connectWiFi();
  Serial.printf("[BOOT] Ready — trigger < %dcm\n", TRIGGER_DISTANCE_CM);
}

// ── Loop ─────────────────────────────────────────────────
void loop() {
  long dist    = distanceCm();
  bool inRange = (dist > 0 && dist < TRIGGER_DISTANCE_CM);
  Serial.printf("[SONIC] %ldcm\n", dist);

  if (inRange) {
    consecutiveHits++;
    lastTriggerMs = millis();
    if (consecutiveHits >= CONFIRM_READINGS && !cameraActive) {
      Serial.println("[TRIGGER] Presence confirmed — streaming!");
      cameraActive = true;
    }
  } else {
    consecutiveHits = 0;
  }

  if (cameraActive) {
    captureAndPost();
    if (!inRange && millis()-lastTriggerMs > IDLE_TIMEOUT_MS) {
      Serial.println("[IDLE] Going idle.");
      cameraActive = false;
    }
  }

  delay(cameraActive ? 50 : 200);
}
