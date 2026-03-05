import os, subprocess, sys
PACKAGES = ["ultralytics","opencv-python","numpy","requests","flask","pyngrok"]
def main():
    print("=" * 55 + "\n  Project Rio — Setup\n" + "=" * 55)
    subprocess.run(sys.executable + " -m pip install " + " ".join(PACKAGES), shell=True)
    try:
        from ultralytics import YOLO
        YOLO("yolov8n.pt"); YOLO("yolov8n-pose.pt")
        print("  Models ready")
    except Exception as e:
        print(f"  Models will download on first run: {e}")
    for d in ("clips", "logs"):
        os.makedirs(d, exist_ok=True)
        print(f"  Created {d}/")
    print("\nDone! Copy .env.example to .env, fill tokens, then:\n  python main.py\n")
if __name__ == "__main__":
    main()
