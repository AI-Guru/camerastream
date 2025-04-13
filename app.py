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
last_detection_result = Nonelast_detection_result = None  # Store the latest detection result

def get_ip_address():
    """Get the local IP address of the machine"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachableble
        s.connect(('10.255.255.255', 1))255', 1))
        IP = s.getsockname()[0]kname()[0]
    except Exception:
        IP = '127.0.0.1' '127.0.0.1'
    finally:
        s.close()se()
    return IP    return IP

def initialize_camera():
    """Initialize the camera object""" the camera object"""
    global camera
    camera = cv2.VideoCapture(0)(0)
    if not camera.isOpened():
        logger.error("Error: Could not open camera.")("Error: Could not open camera.")
        return FalseFalse
    return True    return True

def run_detection(frame_bytes, timestamp):_detection(frame_bytes, timestamp):
    """
    Run LLM-based detection on the provided frameRun LLM-based detection on the provided frame
    
    Args:
        frame_bytes: The JPEG-encoded frame bytestes
        timestamp: When the frame was captured    timestamp: When the frame was captured
    
    Returns:
        Detection results Detection results
    """
    global last_detection_result
    try:
        # Convert bytes back to image for processing
        np_array = np.frombuffer(frame_bytes, np.uint8)        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # Convert to PIL Image for processing.2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        image = Image.fromarray(image)
        image = Image.fromarray(image)
age if new_size is provided.
        # Resize the image if new_size is provided.one
        new_size = None
        if new_size:            image = image.resize(new_size)
            image = image.resize(new_size)
o base64.
        # Convert the image to base64.
        buffer = io.BytesIO()fer, format="JPEG")
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        image_bytes = buffer.read()image_data = base64.b64encode(image_bytes).decode("utf-8")
        image_data = base64.b64encode(image_bytes).decode("utf-8")
        
        # Create the message for the LLM.        system_message = "Du bist ein enthusiastischer Ornithologe."
        system_message = "Du bist ein enthusiastischer Ornithologe."

        # Create the new message with the image. ein Bild von einer Kamera. Wenn du einen Vogel siehst, beschreibe das Bild im Duktus von Steve Irving. Wenn nicht, dann antworte mit 'Kein Vogel gesehen'."
        text = "Hier ist ein Bild von einer Kamera. Wenn du einen Vogel siehst, beschreibe das Bild im Duktus von Steve Irving. Wenn nicht, dann antworte mit 'Kein Vogel gesehen'."
        user_message = [l": {"url": f"data:image/jpeg;base64,{image_data}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},   {"type": "text", "text": text},
            {"type": "text", "text": text},        ]
        ]
 messages.
        # Create the messages.
        messages = [ge},
            {"role": "system", "content": system_message},   {"role": "user", "content": user_message},
            {"role": "user", "content": user_message},        ]
        ]

        # Create the LLM.
        chat_model_parameters = {}
        chat_model_parameters["base_url"] = "hordak:11434"lama"
        chat_model_parameters["model_provider"] = "ollama"7b"
        chat_model_parameters["model"] = "gemma3:27b"
        chat_model_parameters["temperature"] = 1.0        llm = init_chat_model(**chat_model_parameters)
        llm = init_chat_model(**chat_model_parameters)

        # Run the LLM.
        response = llm.invokes(messages).contentlogger.info(f"LLM response: {response}")
        logger.info(f"LLM response: {response}")
        sult with timestamp and LLM response text
        # Example detection result
        result = {stamp,
            "timestamp": timestamp,   "text": response,
            "text": response,}
        }
        etection result in a global variable to make it accessible
        logger.info(f"Detection complete: {result}")with lock:
        tection_result
        # Update the global detection result
        with lock:
            last_detection_result = result        logger.info(f"Detection complete: {result}")
        
        return result
        ption as e:
    except Exception as e:er.error(f"Error in AI detection: {str(e)}")
        logger.error(f"Error in AI detection: {str(e)}")
        return None

def process_detection_queue(): in the detection queue"""
    """Process frames in the detection queue"""
    while True:
        try:# Get frame from queue (blocking with timeout)
            # Get frame from queue (blocking with timeout)t=1)
            frame_data = detection_queue.get(timeout=1)
            if frame_data is None:  # Sentinel value to stop processing
                break    
                mestamp = frame_data
            frame_bytes, timestamp = frame_data
             detection task to the executor
            # Submit detection task to the executor_executor.submit(run_detection, frame_bytes, timestamp)
            future = detection_executor.submit(run_detection, frame_bytes, timestamp)
            # We could store or use the future result here if needed            
            :
        except queue.Empty:
            # Queue is empty, continue waiting
            continue    except Exception as e:
        except Exception as e:ror(f"Error processing detection queue: {str(e)}")
            logger.error(f"Error processing detection queue: {str(e)}")

def camera_capture_loop():re frames from the camera in a dedicated thread"""
    """Continuously capture frames from the camera in a dedicated thread"""e, lock, should_run, last_action_frame, last_action_time, previous_frame_gray, last_action_frame_etag
    global last_frame, lock, should_run, last_action_frame, last_action_time, previous_frame_gray, last_action_frame_etag
    
    while should_run:ne or not camera.isOpened():
        if camera is None or not camera.isOpened():
            logger.error("Camera not available")
            time.sleep(1)
            continue    
            amera.read()
        success, frame = camera.read()
        if not success: camera")
            logger.error("Failed to capture frame from camera")
            time.sleep(0.1)    continue
            continue
        
        # Motion detection
        current_time = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) = cv2.GaussianBlur(gray, (21, 21), 0)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if previous_frame_gray is not None:# Calculate absolute difference between current and previous frame
            # Calculate absolute difference between current and previous frameevious_frame_gray, gray)
            frame_delta = cv2.absdiff(previous_frame_gray, gray) 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            
            # Dilate the thresholded image to fill in holesions=2)
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Calculate amount of motionon_score = np.sum(thresh) / 255
            motion_score = np.sum(thresh) / 255
            eck if there's significant motion and enough time has passed since last action frame
            # Check if there's significant motion and enough time has passed since last action frame
            if (motion_score > ACTION_THRESHOLD and TION_INTERVAL or 
                (current_time - last_action_time > MIN_ACTION_INTERVAL or t_time - last_action_time > MAX_ACTION_INTERVAL)):
                 current_time - last_action_time > MAX_ACTION_INTERVAL)):
                Motion detected with score {motion_score}, capturing action frame")
                logger.info(f"Motion detected with score {motion_score}, capturing action frame")
                
                # Store action frame with lock for thread safetyframe)
                ret, buffer = cv2.imencode('.jpg', frame)et:
                if ret:
                    frame_bytes = buffer.tobytes() lock:
                    with lock:
                        last_action_frame = frame_bytes
                        last_action_frame_etag = hashlib.md5(last_action_frame).hexdigest() current_time
                    last_action_time = current_time
                                # Add frame to detection queue
                    # Add frame to detection queue
                    try:blocking put with a timeout to avoid getting stuck
                        # Use non-blocking put with a timeout to avoid getting stuck            detection_queue.put((frame_bytes, current_time), block=True, timeout=0.5)
                        detection_queue.put((frame_bytes, current_time), block=True, timeout=0.5)e.Full:
                    except queue.Full: queue is full, skipping AI analysis for this frame")
                        logger.warning("Detection queue is full, skipping AI analysis for this frame")
        vious frame
        # Update previous frameious_frame_gray = gray
        previous_frame_gray = gray
            to jpg format
        # Convert to jpg format, frame)
        ret, buffer = cv2.imencode('.jpg', frame)        if not ret:
        if not ret:inue
            continue
            ead safety
        # Update the last frame with lock for thread safety    with lock:
        with lock:tes()
            last_frame = buffer.tobytes()

def get_frame():"""Get the latest frame and control frame rate"""
    """Get the latest frame and control frame rate"""ame_time, last_etag
    global last_frame, last_frame_time, last_etag
    
    # Check if we have a valid frameif last_frame is None:
    if last_frame is None:
        return None, None
    me()
    current_time = time.time()if current_time - last_frame_time < 1.0/FPS:
    if current_time - last_frame_time < 1.0/FPS:etag
        return last_frame, last_etag
    # Get new frame from camera
    # Get new frame from camera
    with lock:e
        frame = last_frame
    
    # Update frame and timestamprent_time
    last_frame_time = current_time
    frame only if frame exists
    # Generate ETag for this frame only if frame existsif frame is not None:
    if frame is not None:
        try:            last_etag = hashlib.md5(frame).hexdigest()
            last_etag = hashlib.md5(frame).hexdigest()TypeError:
        except TypeError:logger.error("TypeError when generating ETag - frame may be invalid")
            logger.error("TypeError when generating ETag - frame may be invalid")None
            return None, None
        return frame, last_etag
    return frame, last_etag

@app.route('/')
def index():
    """Home page route"""emplate('index.html')
    return render_template('index.html')
te('/video_feed')
@app.route('/video_feed')
def video_feed():t"""
    """Video streaming route with ETag support"""lient connection gets its own generator instance
    # Each client connection gets its own generator instance
    def generate():
        last_sent_etag = None
        
        while True:e, current_etag = get_frame()
            frame, current_etag = get_frame()
            None
            # Skip this iteration if frame is None
            if frame is None:
                time.sleep(0.1)
                continue
                # Only send new frame if ETag changed from last sent frame
            # Only send new frame if ETag changed from last sent frame
            if current_etag != last_sent_etag:                yield (b'--frame\r\n'
                yield (b'--frame\r\n'-Type: image/jpeg\r\n'
                      b'Content-Type: image/jpeg\r\n' frame + b'\r\n')
                      b'ETag: ' + current_etag.encode() + b'\r\n\r\n' + frame + b'\r\n')                last_sent_etag = current_etag
                last_sent_etag = current_etag
            eep(1.0/FPS)  # Control frame rate
            time.sleep(1.0/FPS)  # Control frame rate
(),
    return Response(generate(),              mimetype='multipart/x-mixed-replace; boundary=frame')
                  mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/action_frame')on_frame():
def action_frame():the latest action frame"""
    """Return the latest action frame"""
    global last_action_frame
    if last_action_frame is None:
    if last_action_frame is None:
        return "No action detected yet", 404
        
    with lock:        frame = last_action_frame
        frame = last_action_frame_frame_etag
        etag = last_action_frame_etag
    mimetype='image/jpeg')
    response = Response(frame, mimetype='image/jpeg')eaders['ETag'] = etag
    response.headers['ETag'] = etagrn response
    return response

@app.route('/detection_result')tream thread"""
def get_detection_result():    if not initialize_camera():
    """Return the latest detection result as JSON"""
    global last_detection_result
    
    if last_detection_result is None:era capture loop in this thread
        return {"status": "no_detection", "message": "No detection has been run yet"}, 404pture_loop()
    
    with lock:
        result = last_detection_result.copy()
    art_camera_stream)
    # Format the timestamp for display
    result["formatted_time"] = time.strftime('%H:%M:%S', time.localtime(result["timestamp"]))t.start()
    
    return resultrocessing thread
hread = threading.Thread(target=process_detection_queue)
def start_camera_stream():on = True
    """Start the camera stream thread"""d.start()
    if not initialize_camera():    
        return
        























    app.run(host='0.0.0.0', port=port, debug=True, threaded=True, use_reloader=False)    logger.info(f"Starting server at http://{ip_address}:{port}")        port = 5001    if "dev" in sys.argv:    port = 5000    ip_address = get_ip_address()    # Get the local IP address        detection_thread.start()    detection_thread.daemon = True    detection_thread = threading.Thread(target=process_detection_queue)    # Start the detection queue processing thread        t.start()    t.daemon = True    t = threading.Thread(target=start_camera_stream)    # Start camera streaming in a separate threadif __name__ == '__main__':    camera_capture_loop()    # Start the camera capture loop in this thread    logger.info("Camera initialized successfully.")    port = 5000
    if "dev" in sys.argv:
        port = 5001

    logger.info(f"Starting server at http://{ip_address}:{port}")
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True, use_reloader=False)