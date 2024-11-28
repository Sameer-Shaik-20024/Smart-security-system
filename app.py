#Importing necessary libraries
from flask import Flask, render_template, Response, send_file, jsonify
import cv2
import time
import os
from ultralytics import YOLO  
import iptechniques  #Custom module for Digital Image Processing (DIP) techniques

# Initialize Flask app	
app = Flask(__name__)

# Initialize video capture from the default camera
video_capture = cv2.VideoCapture(0)

# Load the YOLO model globally with a pre-trained weight file
model = YOLO('best (3).pt')

# Directories to   save captured and processed images
IMAGE_DIR = 'captured_images'
PROCESSED_DIR = 'processed_images'
INTRUDER_DIR = 'intruder_images'

# List to track paths of captured intruder images along with timestamps
captured_image_paths = []

# Create directories if not exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(INTRUDER_DIR, exist_ok=True)

# Function to save images and apply image processing techniques to intruder snapshots
def save_and_process_intruder(image, image_name, timestamp):
    intruder_image_path = os.path.join(INTRUDER_DIR, image_name)
    cv2.imwrite(intruder_image_path, image) # Save the captured image
    captured_image_paths.append({'path': intruder_image_path, 'timestamp': timestamp})
    iptechniques.process(intruder_image_path)  # # Apply Digital Image Processing (DIP) techniques

# Global variable to track the time of the last intruder snapshot
last_intruder_snapshot_time = 0
INTRUDER_SNAPSHOT_INTERVAL = 5  # Interval in seconds between intruder snapshots

# Valid users and confidence thresholds
VALID_USERS = {"sameer", "purna"}
CONFIDENCE_THRESHOLDS = {"sameer": 0.8, "purna": 0.8}  # Minimum confidence for "purna" or "sameer" to be valid

# Generator function to stream video with detection and labeling
def generate_frames():
    global last_intruder_snapshot_time  # To track the last intruder snapshot time
    while True:
        success, frame = video_capture.read() # Capture a frame from the video stream
        if not success:
            break

        # Run YOLO detection on the frame
        results = model(frame, conf=0.6, save=False, save_crop=False, iou=0.2)

        for result in results:
            for box in result.boxes:
                cls = box.cls[0]  # Class index
                conf = box.conf[0]  # Confidence score
                label = model.names[int(cls)]  # Class name from YOLO model

                # Log detected label and confidence
                print(f"Detected: {label}, Confidence: {conf:.2f}")

                # Validate detection
                if label in VALID_USERS:
                    user_threshold = CONFIDENCE_THRESHOLDS.get(label, 0.8)
                    if conf < user_threshold:
                        label = "intruder"  # Treat user as intruder if confidence is below their threshold
                else:
                    label = "intruder"

                # Save and process intruder images only if the interval has elapsed
                if label == "intruder":
                    current_time = time.time()
                    if current_time - last_intruder_snapshot_time >= INTRUDER_SNAPSHOT_INTERVAL:
                        last_intruder_snapshot_time = current_time  # Update the last snapshot time
                        timestamp = time.strftime("%b %d, %Y %I:%M:%S %p")  # Get the timestamp
                        intruder_name = f"intruder_{int(current_time)}.jpg"
                        save_and_process_intruder(frame, intruder_name, timestamp)

                # Annotate the frame
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                if label == "intruder":
                    color = (0, 0, 255)  # Red for intruders
                else:
                    color = (0, 255, 0)  # Green for valid users                  
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Encode and yield the frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to stream the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to fetch the most recent intruder image
@app.route('/get_recent_image')
def get_recent_image():
    if captured_image_paths:
        latest_path = captured_image_paths[-1]['path']
        if os.path.exists(latest_path):
            return send_file(latest_path, mimetype='image/jpeg')
    return "No recent image found", 404

# Route to fetch a processed version of the most recent intruder image
@app.route('/get_processed_image/<filter_type>')
def get_processed_image(filter_type):
    if captured_image_paths:
        latest_path = captured_image_paths[-1]['path']
        processed_image = iptechniques.apply_filter(latest_path, filter_type)
        if processed_image is not None:
            _, img_encoded = cv2.imencode('.jpg', processed_image)
            return Response(img_encoded.tobytes(), mimetype='image/jpeg')
    return "Processed image not found", 404

# Route to fetch logs of intruder images along with processing options
@app.route('/get_intruder_logs')
def get_intruder_logs():
    logs = []
    for image_data in captured_image_paths:
        processed_images = {}
        filters = ['original', 'median', 'highpass', 'histogram', 'edge', 'sobel', 'unsharp']
        for filter_type in filters:
            processed_images[filter_type] = f"/get_processed_image/{filter_type}?image={os.path.basename(image_data['path'])}"
        processed_images['timestamp'] = image_data['timestamp']
        logs.append(processed_images)
    return jsonify(logs)

# Route to fetch the application's logo image
@app.route('/image/logo')
def get_logo():
    image_path = os.path.join("logo", "logo.png")
    return send_file(image_path, mimetype='image/png')

# Entry point for the Flask application
if __name__ == '__main__':
    app.run(debug=False) # Run the app with debugging disabled for production
