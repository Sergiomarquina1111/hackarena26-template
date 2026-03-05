#!/bin/bash
# Project Rio — one-time setup
pip install -r requirements.txt
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8n-pose.pt')"
mkdir -p clips logs
cp .env.example .env
echo "Done. Fill in .env with your tokens then: python main.py"
