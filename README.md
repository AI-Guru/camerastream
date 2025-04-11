# Camera Stream Web Application

A lightweight web application that streams video from a connected camera to web browsers. Built with Flask and OpenCV, this application enables real-time camera streaming that can be accessed from any device on your local network.

## Features

- Real-time video streaming at 30 FPS
- Accessible from any device with a web browser on the local network
- Thread-safe implementation for optimal performance
- Dockerized for easy deployment and cross-platform compatibility

## Prerequisites

- Python 3.9+ (if running without Docker)
- Web camera (USB webcam or built-in camera)
- Docker and Docker Compose (for containerized deployment)

## Installation

### Option 1: Using Docker (Recommended)

1. Install Docker and Docker Compose:
   ```
   chmod +x installdocker.sh
   ./installdocker.sh
   ```

2. Build and run the container:
   ```
   sudo docker compose up -d
   ```

3. Access the stream at `http://<your-ip-address>:5000`

### Option 2: Local Installation

1. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python app.py
   ```

3. Access the stream at `http://<your-ip-address>:5000`

## Configuration

You can modify these settings in `app.py`:

- `fps` - Frames per second for camera capture (default: 30)
- `port` - Web server port (default: 5000)

## Project Structure

```
├── app.py                 # Main application file
├── docker-compose.yaml    # Docker Compose configuration
├── Dockerfile             # Docker container definition
├── installdocker.sh       # Docker installation script
├── requirements.txt       # Python dependencies
└── templates/             # HTML templates for the web interface
    └── index.html         # Main web page
```

## Troubleshooting

### Camera Access Issues

- If running with Docker, ensure the webcam device is properly mapped in docker-compose.yaml
- Default camera is set to index 0 (`cv2.VideoCapture(0)`). If you have multiple cameras, you may need to change this value.

### Performance Optimization

- If experiencing network lag, try reducing the FPS value in `app.py`
- For low-powered devices, consider lowering the resolution in the OpenCV capture settings

## License

This project is open source and available under the [MIT License](https://opensource.org/licenses/MIT).

## Acknowledgments

- [Flask](https://flask.palletsprojects.com/)
- [OpenCV](https://opencv.org/)