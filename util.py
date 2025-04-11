# util.py - By Arkja Gaur 💻🛡️

import torch
import cv2

def load_custom_model(weights='yolov5s.pt'):
    """
    Loads a YOLOv5 model from local weights.
    :param weights: Path to the .pt file
    :return: model object
    """
    print("[INFO] Loading model... ✨")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=True)
    print("[INFO] Model loaded successfully ✅")
    return model

def detect_on_image(model, image_path):
    """
    Runs inference on a given image and displays the result.
    """
    print(f"[INFO] Detecting objects in {image_path} 🖼️")
    results = model(image_path)
    results.print()
    results.show()
    return results

def detect_realtime(model):
    """
    Opens webcam for real-time detection.
    """
    print("[INFO] Starting webcam for real-time detection 🎥")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Couldn't read frame from webcam ❌")
            break

        results = model(frame)
        rendered_frame = results.render()[0]
        cv2.imshow("🔍 Arkja's Real-Time Detection", rendered_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quitting webcam... 🛑")
            break

    cap.release()
    cv2.destroyAllWindows()
