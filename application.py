from flask import Flask, Response, render_template_string
import cv2
import threading
from ultralytics import YOLO
import time

# Initialize Flask app
app = Flask(__name__)

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")  # Ensure this file exists

# Global variables
latest_frame = None
lock = threading.Lock()
video_stream = None
processing_thread = None
running = False
detection_started = False  # Flag to control detection state

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Omni Recognition</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 0;
            background-color: #222;
            color: #eee;
            text-align: center;
        }

        header {
            background-color: #444;
            color: white;
            padding: 20px 0;
        }

        header h1 {
            margin: 0;
            font-size: 28px;
        }

        main {
            padding: 20px;
        }

        .video-section {
            margin: 20px auto;
            max-width: 700px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            background-color: #333;
            border-radius: 8px;
            overflow: hidden;
        }

        .video-container img {
            width: 100%;
            border-radius: 8px;
        }

        .controls {
            margin: 20px;
        }

        .controls button {
            margin: 5px;
            padding: 10px 20px;
            font-size: 16px;
            background-color: #0078d4;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
        }

        .controls button:hover {
            background-color: #005fa3;
        }

        .status-log {
            margin: 20px auto;
            max-width: 700px;
            background-color: #444;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }

        .status-log #log {
            font-family: 'Courier New', monospace;
            text-align: left;
            overflow-y: auto;
            max-height: 200px;
            padding: 10px;
            color: #00ff00;
            background-color: #222;
            border-radius: 8px;
            border: 1px solid #555;
        }
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
                .then(response => response.text())
                .then(data => {
                    logStatus(data);
                })
                .catch(err => {
                    logStatus("Error starting detection: " + err);
                });
        }

        function stopDetection() {
            fetch('/stop')
                .then(response => response.text())
                .then(data => {
                    logStatus(data);
                })
                .catch(err => {
                    logStatus("Error stopping detection: " + err);
                });
        }

        function logStatus(message) {
            const logDiv = document.getElementById("log");
            const logEntry = document.createElement("p");
            logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            logDiv.appendChild(logEntry);
            logDiv.scrollTop = logDiv.scrollHeight; // Auto-scroll to the latest log
        }
    </script>
</body>
</html>
"""


def preprocess_frame(frame):
    """
    Resize frame for faster processing.
    """
    return cv2.resize(frame, (640, 480))


def detect_activity(frame):
    """
    Run YOLO model on the frame.
    """
    results = yolo_model(frame)
    detections = results[0].boxes.data.cpu().numpy() if len(results) > 0 else []

    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection[:6]
        label = f"{yolo_model.names[int(class_id)]} ({confidence:.2f})"
        color = (0, 255, 0)

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


def background_frame_processing():
    """
    Continuously capture and process frames.
    """
    global latest_frame, running, video_stream

    print("Initializing video stream...")
    video_stream = cv2.VideoCapture(0)  # Open webcam
    if not video_stream.isOpened():
        print("Error: Could not access webcam.")
        running = False
        return

    print("Video stream initialized.")
    while running:
        success, frame = video_stream.read()
        if not success:
            print("Error: Failed to capture frame. Retrying...")
            time.sleep(0.1)
            continue

        processed_frame = detect_activity(preprocess_frame(frame))

        # Update the latest frame safely
        with lock:
            latest_frame = processed_frame

    video_stream.release()
    print("Video stream released.")


def start_detection():
    """
    Start background processing for detection.
    """
    global running, processing_thread

    if running:
        print("Detection already running.")
        return

    running = True
    print("Starting detection thread...")
    processing_thread = threading.Thread(target=background_frame_processing, daemon=True)
    processing_thread.start()


def stop_detection():
    """
    Stop the background processing.
    """
    global running, video_stream

    running = False
    if video_stream:
        print("Stopping video stream.")
        video_stream.release()


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/start")
def start():
    global detection_started
    if not detection_started:
        start_detection()
        detection_started = True
        return "Detection Started"
    return "Detection Already Running"


@app.route("/video_feed")
def video_feed():
    """
    Endpoint to provide video stream.
    """
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stop")
def stop():
    """
    Stop detection and release resources.
    """
    global detection_started
    stop_detection()
    detection_started = False
    return "Detection Stopped"


def generate_frames():
    """
    Stream processed frames to the frontend.
    """
    while True:
        with lock:
            if latest_frame is None:
                time.sleep(0.1)  # Delay to reduce spam
                continue

            _, buffer = cv2.imencode(".jpg", latest_frame)
            frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
