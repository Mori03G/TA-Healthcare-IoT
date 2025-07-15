from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory
import cv2
import os
import requests
import numpy as np
from datetime import datetime
from tensorflow import keras
from tensorflow.keras.preprocessing import image

app = Flask(__name__)


model_path = "/home/rifqi_izzax3/model/VGG16_notuner_noaug.keras"
model = keras.models.load_model(model_path)


camera_urls = {
    1: 'http://100.97.204.50:8080/?action=stream',
    2: 'http://100.124.167.8:8080/?action=stream',
    3: 'http://100.89.159.37:8080/?action=stream'
}


capture_folder = 'static/captures'
os.makedirs(capture_folder, exist_ok=True)


def gen_frames(url):
    stream = requests.get(url, stream=True)
    bytes_data = b''
    for chunk in stream.iter_content(chunk_size=1024):
        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8')
        b = bytes_data.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')


def capture_frame(url):
    stream = requests.get(url, stream=True)
    bytes_data = b''
    for chunk in stream.iter_content(chunk_size=1024):
        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8')
        b = bytes_data.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            nparr = np.frombuffer(jpg, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
    return None


def classify_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)

    result = "Faringitis" if prediction[0][0] < 0.5 else "Normal"
    return result


@app.route('/')
def index():
    return redirect(url_for('camera_page', cam_id=1))


@app.route('/camera/<int:cam_id>')
def camera_page(cam_id):
    if cam_id not in camera_urls:
        return "Invalid camera ID", 400

    files = [f for f in os.listdir(capture_folder) if f.startswith(f'cam{cam_id}_')]
    files.sort(reverse=True)
    latest_image = files[0] if files else None
    return render_template('index3.html', cam_id=cam_id, latest_image=latest_image)


@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id):
    cam_url = camera_urls.get(cam_id)
    if not cam_url:
        return "Invalid camera ID", 400
    return Response(gen_frames(cam_url), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture/<int:cam_id>', methods=['POST'])
def capture(cam_id):
    cam_url = camera_urls.get(cam_id)
    if not cam_url:
        return "Invalid camera ID", 400

    frame = capture_frame(cam_url)
    if frame is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'cam{cam_id}_{timestamp}.jpg'
        filepath = os.path.join(capture_folder, filename)
        cv2.imwrite(filepath, frame)
        print(f"[INFO] Gambar disimpan: {filepath}")
        return redirect(url_for('classify', filename=filename))
    else:
        return "Failed to capture frame", 500
    
@app.route('/classify/<filename>')
def classify(filename):
    image_path = os.path.join(capture_folder, filename)
    result = classify_image(image_path)
    return render_template('result2.html', filename=filename, result=result)


@app.route('/static/captures/<filename>')
def get_image(filename):
    return send_from_directory(capture_folder, filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)