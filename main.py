import torch
import cv2

# main.py
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_weapon(frame):
    results = model(frame)[0]
    for result in results.boxes:
        cls = int(result.cls[0])
        conf = float(result.conf[0])
        if conf > 0.5:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{model.names[cls]} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return frame


# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Use webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)

    # Convert results to a displayable image
    img = results.render()[0]  # render boxes and labels
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Show the frame
    cv2.imshow("Weapon Detection - Real Time", img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# main.py - Entry point 🚀

from util import load_custom_model, detect_on_image, detect_realtime

# Load your trained/custom YOLO model
model = load_custom_model('yolov5s.pt')

# Uncomment this to test on static image
# detect_on_image(model, 'assets/test_image.jpg')

# For real-time detection
detect_realtime(model)

