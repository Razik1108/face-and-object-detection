from flask import Flask, Response, jsonify
import cv2
import torch

app = Flask(__name__)

# Load YOLOv8 model (Uses YOLOv5s as YOLOv8 runs on Ultralytics framework)
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

# Initialize video capture (0 for webcam)
video_capture = cv2.VideoCapture(0)

# Global flag for detection state
detection_enabled = False

def detect_objects(frame):
    """ Applies YOLO object detection """
    results = model(frame)
    detections = results.pandas().xyxy[0]  # Get results as Pandas DataFrame

    for _, row in detections.iterrows():
        x1, y1, x2, y2, conf, cls = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], row['name']
        label = f"{cls} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def generate_frames():
    """ Generates video frames with or without detection """
    global detection_enabled
    while True:
        success, frame = video_capture.read()
        if not success:
            break

        if detection_enabled:
            frame = detect_objects(frame)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    """ Serves the web UI """
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Omni Recognition</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #222; color: #eee; text-align: center; }
        header { background-color: #444; color: white; padding: 20px; }
        main { padding: 20px; }
        .video-section { margin: 20px auto; max-width: 700px; background-color: #333; border-radius: 8px; }
        .video-container img { width: 100%; border-radius: 8px; }
        .controls { margin: 20px; }
        .controls button { margin: 5px; padding: 10px 20px; font-size: 16px; background-color: #0078d4; color: white; border: none; border-radius: 5px; cursor: pointer; }
        .controls button:hover { background-color: #005fa3; }
        .status-log { margin: 20px auto; max-width: 700px; background-color: #444; padding: 10px; border-radius: 8px; }
        .status-log #log { font-family: 'Courier New', monospace; text-align: left; max-height: 200px; padding: 10px; color: #00ff00; background-color: #222; border-radius: 8px; border: 1px solid #555; overflow-y: auto; }
    </style>
</head>
<body>
    <header>
        <h1>Omni Recognition</h1>
    </header>
    <main>
        <section class="video-section">
            <div class="video-container">
                <h2>Live Camera Feed</h2>
                <img id="video-feed" src="/video_feed" alt="Live Video Feed">
            </div>
        </section>
        <section class="controls">
            <button id="start-btn" onclick="startDetection()">Start Detection</button>
            <button id="stop-btn" onclick="stopDetection()">Stop Detection</button>
        </section>
        <section class="status-log">
            <h2>Status</h2>
            <div id="log"></div>
        </section>
    </main>
    <footer>
        <p>&copy; 2024 Omni Recognition. All rights reserved.</p>
    </footer>
    <script>
        function startDetection() {
            fetch('/start')
                .then(response => response.json())
                .then(data => logStatus(data.status))
                .catch(err => logStatus("Error: " + err));
        }

        function stopDetection() {
            fetch('/stop')
                .then(response => response.json())
                .then(data => logStatus(data.status))
                .catch(err => logStatus("Error: " + err));
        }

        function logStatus(message) {
            const logDiv = document.getElementById("log");
            const logEntry = document.createElement("p");
            logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            logDiv.appendChild(logEntry);
            logDiv.scrollTop = logDiv.scrollHeight;
        }
    </script>
</body>
</html>
"""

@app.route('/video_feed')
def video_feed():
    """ Serves the video stream """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start_detection():
    """ Enables object detection """
    global detection_enabled
    detection_enabled = True
    return jsonify({"status": "Detection started"})

@app.route('/stop')
def stop_detection():
    """ Disables object detection """
    global detection_enabled
    detection_enabled = False
    return jsonify({"status": "Detection stopped"})

if __name__ == '__main__':
    app.run(debug=True)
