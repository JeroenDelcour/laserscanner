import numpy as np
import picamera
import picamera.array
import gpiozero
from time import sleep


class MyOutput:
    def __init__(self, file, callback=lambda f: None):
        self.file = open(file, "wb")
        self.callback = callback
        self.frame = 0
    
    def write(self, buf):
        self.callback(self.frame)
        self.file.write(buf)
        self.frame += 1

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


camera = picamera.PiCamera(resolution=(1280, 720), framerate=60)
camera.iso = 800
sleep(2)  # let automatic gain settle
long_exposure = camera.exposure_speed
short_exposure = 1000
camera.exposure_mode = "off"

def callback(frame):
    camera.shutter_speed = long_exposure if frame % 2 == 0 else short_exposure

laser = gpiozero.LED(17)
laser.on()
camera.start_recording(MyOutput("video.h264", callback=callback), format="h264")
camera.wait_recording(5)
camera.stop_recording()
laser.off()
