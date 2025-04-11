import cv2
import threading
from flask import Flask, render_template, Response
import socket
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
camera = None
last_frame = None
lock = threading.Lock()
should_run = True
fps = 30  # Frames per second for capture

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
    
    frame_interval = 1.0 / fps
    
    while should_run:
        start_time = time.time()
        
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
        
        # Calculate sleep time to maintain desired fps
        elapsed = time.time() - start_time
        sleep_time = max(0, frame_interval - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

def generate_frames():
    """Yield the latest frame to clients"""
    global last_frame, lock
    
    while True:
        # Wait until we have a frame
        if last_frame is None:
            time.sleep(0.1)
            continue
            
        # Get the latest frame with lock for thread safety
        with lock:
            frame_to_yield = last_frame
            
        # Yield the frame in the format expected by Flask
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_to_yield + b'\r\n')
        
        # Small delay to control client-side framerate
        time.sleep(0.033)  # ~30fps for clients

@app.route('/')
def index():
    """Home page route"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

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