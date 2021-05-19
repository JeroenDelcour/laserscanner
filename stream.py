import io
import picamera
import logging
import socketserver
from threading import Condition
from http import server
import gpiozero
from time import sleep

PAGE="""\
<html>
<head>
<title>picamera MJPEG streaming demo</title>
</head>
<body>
<h1>PiCamera MJPEG Streaming Demo</h1>
<img src="stream.mjpg" width="1640" height="1232" />
</body>
</html>
"""

class StreamingOutput(object):
    def __init__(self, camera, short_shutter=200, long_shutter="auto"):
        self.camera = camera
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()
        self._longshutter_mode = True
        self.short_shutter = short_shutter
        if long_shutter == "auto":
            self.long_shutter = camera.exposure_speed
        else:
            self.long_shutter = long_shutter

    def write(self, buf):
        if self._longshutter_mode:
            self.camera.shutter_speed = self.long_shutter
        else:
            self.camera.shutter_speed = self.short_shutter
        self._longshutter_mode = not self._longshutter_mode
        if buf.startswith(b'\xff\xd8'):
            # New frame, copy the existing buffer's content and notify all
            # clients it's available
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
        return self.buffer.write(buf)

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

laser = gpiozero.LED(17)
laser.on()

with picamera.PiCamera(resolution='1640x1232', framerate=25, sensor_mode=4) as camera:
    # camera.shutter_speed = 200
    camera.iso = 800
    sleep(2)  # let automatic gain settle
    camera.exposure_mode = "off"
    output = StreamingOutput(camera, long_shutter=20000)
    camera.start_recording(output, format='mjpeg')
    try:
        address = ('', 8000)
        server = StreamingServer(address, StreamingHandler)
        server.serve_forever()
    finally:
        laser.off()
        camera.stop_recording()

