import cv2
from ultralytics import YOLO
import os

model = YOLO("yolov8n.pt")  # or your trained weights like 'best.pt'

folder = "assets"
for file in os.listdir(folder):
    if file.endswith(".jpg"):
        path = os.path.join(folder, file)
        results = model(path)  # Don't use show=True
        annotated_frame = results[0].plot()

        # Show the image until you press a key
        cv2.imshow(f"Detection - {file}", annotated_frame)
        cv2.waitKey(0)  # 0 = wait until any key is pressed
        cv2.destroyAllWindows()
