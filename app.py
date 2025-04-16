from flask import Flask, render_template, Response
import cv2
from main import detect_weapon
import ssl
import smtplib
from email.message import EmailMessage

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


def send_alert_email(sender_email, receiver_email, app_password, subject, body, image_path=None):
    try:
        em = EmailMessage()
        em["From"] = sender_email
        em["To"] = receiver_email
        em["Subject"] = subject
        em.set_content(body)

        # Attach image if path is provided
        if image_path:
            with open(image_path, "rb") as img:
                em.add_attachment(img.read(), maintype="image", subtype="jpeg", filename="alert.jpg")

        # Secure connection
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
            smtp.login(sender_email, app_password)
            smtp.send_message(em)

        print("📧 Alert email sent with image!")

    except Exception as e:
        print(f"❌ Email failed: {e}")

app = Flask(__name__)
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = detect_weapon(frame)
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
