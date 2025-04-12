# Camera Stream

A Flask application that streams camera images with optimized performance.

## Features

- Real-time camera streaming
- Configurable frame rate (FPS)
- ETag support to reduce bandwidth usage

## Configuration

You can adjust the streaming frame rate by sending a POST request to the `/config` endpoint:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"fps": 15}' http://localhost:5000/config
```

## Performance Optimization

- **FPS Control**: The stream is limited to a configurable FPS to reduce CPU usage and bandwidth
- **ETag Implementation**: Browsers and clients that support ETags will only receive new frames when the content has changed
