import socket
import time

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, JpegEncoder
from picamera2.outputs import FileOutput

picam2 = Picamera2()
video_config = picam2.create_video_configuration(
        {"size": (960, 540)},
        raw=picam2.sensor_modes[1]  # force full frame sensor mode, no cropping
)
picam2.configure(video_config)
encoder = H264Encoder(1000000)
# encoder = JpegEncoder()

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    sock.connect(("192.168.2.2", 8090))
    stream = sock.makefile("wb")
    picam2.start_recording(encoder, FileOutput(stream))
    try:
        while True:
            pass
    finally:
        picam2.stop_recording()

