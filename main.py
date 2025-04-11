import torch
import cv2

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

