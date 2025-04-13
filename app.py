import cv2
import sys
import threading
from flask import Flask, render_template, Response, request
import socket
import logging
import time
import hashlib
import numpy as np
import base64
import io
from PIL import Image
import queue
from concurrent.futures import ThreadPoolExecutor
from langchain.chat_models import init_chat_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration (all fixed parameters)
FPS = 10  # Frames per second
last_frame_time = 0
last_frame = None
last_etag = None

# Action detection configuration (all fixed parameters)
ACTION_THRESHOLD = 5000  # Threshold for detecting motion
MIN_ACTION_INTERVAL = 1  # Minimum interval (seconds) between action frame captures
MAX_ACTION_INTERVAL = 60  # Maximum interval (seconds) before allowing a new action capture
last_action_frame = None
last_action_frame_etag = None
last_action_time = 0
previous_frame_gray = None

# Global variables
camera = None
lock = threading.Lock()
should_run = True
detection_executor = ThreadPoolExecutor(max_workers=1)
detection_queue = queue.Queue(maxsize=5)  # Limit queue size to prevent memory issues
last_detection_result = None  # Store the latest detection result

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

def run_detection(frame_bytes, timestamp):
    """
    Run LLM-based detection on the provided frame
    
    Args:
        frame_bytes: The JPEG-encoded frame bytes
        timestamp: When the frame was captured
    
    Returns:
        Detection results
    """
    try:
        # Convert bytes back to image for processing
        np_array = np.frombuffer(frame_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # Convert to PIL Image for processing.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        # Resize the image if new_size is provided.
        new_size = None
        if new_size:
            image = image.resize(new_size)

        # Convert the image to base64.
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        image_bytes = buffer.read()
        image_data = base64.b64encode(image_bytes).decode("utf-8")
        
        # Create the message for the LLM.
        system_message = "Du bist ein enthusiastischer Ornithologe."

        # Create the new message with the image.
        text = "Hier ist ein Bild von einer Kamera. Wenn du einen Vogel siehst, beschreibe das Bild im Duktus von Steve Irving. Wenn nicht, dann antworte mit 'Kein Vogel gesehen'."
        user_message = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            {"type": "text", "text": text},
        ]

        # Create the messages.
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        # Create the LLM.
        chat_model_parameters = {}
        chat_model_parameters["base_url"] = "hordak:11434"
        chat_model_parameters["model_provider"] = "ollama"
        chat_model_parameters["model"] = "gemma3:27b"
        chat_model_parameters["temperature"] = 1.0
        llm = init_chat_model(**chat_model_parameters)

        # Run the LLM.
        response = llm.invokes(messages).content
        logger.info(f"LLM response: {response}")
        
        # Store result with timestamp and LLM response text
        result = {
            "timestamp": timestamp,
            "text": response,
        }
        
        # Store the detection result in a global variable to make it accessible
        with lock:
            global last_detection_result
            last_detection_result = result
        
        logger.info(f"Detection complete: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in AI detection: {str(e)}")
        return None

def process_detection_queue():
    """Process frames in the detection queue"""
    while True:
        try:
            # Get frame from queue (blocking with timeout)
            frame_data = detection_queue.get(timeout=1)
            if frame_data is None:  # Sentinel value to stop processing
                break
                
            frame_bytes, timestamp = frame_data
            
            # Submit detection task to the executor
            future = detection_executor.submit(run_detection, frame_bytes, timestamp)
            # We could store or use the future result here if needed
            
        except queue.Empty:
            # Queue is empty, continue waiting
            continue
        except Exception as e:
            logger.error(f"Error processing detection queue: {str(e)}")

def camera_capture_loop():
    """Continuously capture frames from the camera in a dedicated thread"""
    global last_frame, lock, should_run, last_action_frame, last_action_time, previous_frame_gray, last_action_frame_etag
    
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
        
        # Motion detection
        current_time = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if previous_frame_gray is not None:
            # Calculate absolute difference between current and previous frame
            frame_delta = cv2.absdiff(previous_frame_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            
            # Dilate the thresholded image to fill in holes
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Calculate amount of motion
            motion_score = np.sum(thresh) / 255
            
            # Check if there's significant motion and enough time has passed since last action frame
            if (motion_score > ACTION_THRESHOLD and 
                (current_time - last_action_time > MIN_ACTION_INTERVAL or 
                 current_time - last_action_time > MAX_ACTION_INTERVAL)):
                
                logger.info(f"Motion detected with score {motion_score}, capturing action frame")
                
                # Store action frame with lock for thread safety
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    with lock:
                        last_action_frame = frame_bytes
                        last_action_frame_etag = hashlib.md5(last_action_frame).hexdigest()
                    last_action_time = current_time
                    
                    # Add frame to detection queue
                    try:
                        # Use non-blocking put with a timeout to avoid getting stuck
                        detection_queue.put((frame_bytes, current_time), block=True, timeout=0.5)
                    except queue.Full:
                        logger.warning("Detection queue is full, skipping AI analysis for this frame")
        
        # Update previous frame
        previous_frame_gray = gray
            
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
    
    # Check if we have a valid frame
    if last_frame is None:
        return None, None
    
    current_time = time.time()
    if current_time - last_frame_time < 1.0/FPS:
        return last_frame, last_etag
    
    # Get new frame from camera
    with lock:
        frame = last_frame
    
    # Update frame and timestamp
    last_frame_time = current_time
    
    # Generate ETag for this frame only if frame exists
    if frame is not None:
        try:
            last_etag = hashlib.md5(frame).hexdigest()
        except TypeError:
            logger.error("TypeError when generating ETag - frame may be invalid")
            return None, None
    
    return frame, last_etag

@app.route('/')
def index():
    """Home page route"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route with ETag support"""
    # Each client connection gets its own generator instance
    def generate():
        last_sent_etag = None
        
        while True:
            frame, current_etag = get_frame()
            
            # Skip this iteration if frame is None
            if frame is None:
                time.sleep(0.1)
                continue
                
            # Only send new frame if ETag changed from last sent frame
            if current_etag != last_sent_etag:
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n'
                      b'ETag: ' + current_etag.encode() + b'\r\n\r\n' + frame + b'\r\n')
                last_sent_etag = current_etag
            
            time.sleep(1.0/FPS)  # Control frame rate

    return Response(generate(),
                  mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/action_frame')
def action_frame():
    """Return the latest action frame"""
    global last_action_frame
    
    if last_action_frame is None:
        return "No action detected yet", 404
        
    with lock:
        frame = last_action_frame
        etag = last_action_frame_etag
    
    response = Response(frame, mimetype='image/jpeg')
    response.headers['ETag'] = etag
    return response

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
    
    # Start the detection queue processing thread
    detection_thread = threading.Thread(target=process_detection_queue)
    detection_thread.daemon = True
    detection_thread.start()
    
    # Get the local IP address
    ip_address = get_ip_address()
    port = 5000
    if "dev" in sys.argv:
        port = 5001

    logger.info(f"Starting server at http://{ip_address}:{port}")
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True, use_reloader=False)