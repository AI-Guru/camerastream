import cv2
import threading
from flask import Flask, render_template, Response, request
import socket
import logging
import time
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
FPS = 10  # Default frames per second
last_frame_time = 0
last_frame = None
last_etag = None

# Global variables
camera = None
lock = threading.Lock()
should_run = True

def get_ip_address():
    """Get the local IP address of the machine"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def initialize_camera():
    """Initialize the camera object"""
    global camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        logger.error("Error: Could not open camera.")
        return False
    return True

def camera_capture_loop():
    """Continuously capture frames from the camera in a dedicated thread"""
    global last_frame, lock, should_run
    
    while should_run:
        if camera is None or not camera.isOpened():
            logger.error("Camera not available")
            time.sleep(1)
            continue
            
        success, frame = camera.read()
        if not success:
            logger.error("Failed to capture frame from camera")
            time.sleep(0.1)
            continue
            
        # Convert to jpg format
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        # Update the last frame with lock for thread safety
        with lock:
            last_frame = buffer.tobytes()

def get_frame():
    """Get the latest frame and control frame rate"""
    global last_frame, last_frame_time, last_etag
    
    current_time = time.time()
    if current_time - last_frame_time < 1.0/FPS:
        return last_frame, last_etag
    
    # Get new frame from camera
    with lock:
        frame = last_frame
    
    # Update frame and timestamp
    last_frame_time = current_time
    
    # Generate ETag for this frame
    last_etag = hashlib.md5(frame).hexdigest()
    
    return frame, last_etag

@app.route('/')
def index():
    """Home page route"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route with ETag support"""
    # Each client connection gets its own generator instance
    # with its own last_sent_etag variable
    def generate():
        last_sent_etag = None
        
        while True:
            frame, current_etag = get_frame()
            
            # Only send new frame if ETag changed from last sent frame
            if current_etag != last_sent_etag:
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n'
                      b'ETag: ' + current_etag.encode() + b'\r\n\r\n' + frame + b'\r\n')
                last_sent_etag = current_etag
            
            time.sleep(1.0/FPS)  # Control frame rate

    return Response(generate(),
                  mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/config', methods=['POST'])
def update_config():
    """Update configuration settings like FPS"""
    global FPS
    if 'fps' in request.json:
        new_fps = int(request.json['fps'])
        if 1 <= new_fps <= 30:  # Reasonable FPS range
            FPS = new_fps
            return {"status": "success", "fps": FPS}
    return {"status": "error", "message": "Invalid FPS value"}, 400

def start_camera_stream():
    """Start the camera stream thread"""
    if not initialize_camera():
        return
        
    logger.info("Camera initialized successfully.")
    # Start the camera capture loop in this thread
    camera_capture_loop()

if __name__ == '__main__':
    # Start camera streaming in a separate thread
    t = threading.Thread(target=start_camera_stream)
    t.daemon = True
    t.start()
    
    # Get the local IP address
    ip_address = get_ip_address()
    port = 5000
    
    logger.info(f"Starting server at http://{ip_address}:{port}")
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True, use_reloader=False)