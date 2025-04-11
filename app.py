import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
from datetime import datetime
import time
import smtplib
import ssl
from email.message import EmailMessage

# --------------------------
# Email Alert Function with Image Attachment
# --------------------------
def send_email(detected_obj, confidence, timestamp, image_path):
    sender_email = "arkjakki@gmail.com"             # 🔁 Your email
    receiver_email = "arkjakki@gmail.com"           # 🔁 Receiver email
    app_password = "odzraygrxyrjkvza"               # 🔁 Your app password

    subject = f"🚨 Weapon Detection Alert: {detected_obj} Detected"
    body = f"""⚠️ ALERT!

Object Detected: {detected_obj}
Confidence: {confidence:.2f}
Time: {timestamp}

Image from the moment is attached below.
"""

    em = EmailMessage()
    em['From'] = sender_email
    em['To'] = receiver_email
    em['Subject'] = subject
    em.set_content(body)

    # Attach the saved image
    with open(image_path, 'rb') as img:
        em.add_attachment(img.read(), maintype='image', subtype='jpeg', filename=image_path)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
            smtp.login(sender_email, app_password)
            smtp.send_message(em)
        st.success("📧 Alert email sent with image!")
    except Exception as e:
        st.error(f"❌ Email failed: {e}")

# --------------------------
# Streamlit UI
# --------------------------
st.title("🛡️ Weapon Detection System with Alerts")

model = YOLO("yolov8n.pt")  # Use your trained model (or 'best.pt')

run = st.checkbox("Start Webcam")
frame_placeholder = st.empty()
alert_placeholder = st.empty()

dangerous_objects = {"knife", "scissors", "gun", "grenade"}

if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Webcam not accessible.")
            break

        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        found = False

        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            obj = results[0].names[cls]

            if obj.lower() in dangerous_objects:
                found = True
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if conf > 0.5:
                    alert_placeholder.warning(
                        f"🚨 Detected '{obj}' with {conf:.2f} confidence at {timestamp}"
                    )

                    with open("detection_log.txt", "a") as log_file:
                        log_file.write(f"{timestamp} - {obj} detected (Confidence: {conf:.2f})\n")

                    # Save the current frame
                    img_filename = f"alert_{timestamp.replace(':', '-')}.jpg"
                    cv2.imwrite(img_filename, frame)

                    # Send email with image
                    send_email(obj, conf, timestamp, img_filename)

                else:
                    alert_placeholder.info(
                        f"⚠️ '{obj}' detected with low confidence ({conf:.2f}). Not logged as threat."
                    )
                break

        if not found:
            alert_placeholder.info("✅ No threats detected.")

        frame_placeholder.image(annotated_frame, channels="BGR")
        time.sleep(0.1)

    cap.release()
else:
    st.info("Click the checkbox above to start detection.")
