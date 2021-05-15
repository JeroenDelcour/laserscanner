import socket
import time
import picamera

camera = picamera.PiCamera()
camera.resolution = (1280, 720)
camera.framerate = 60

server_socket = socket.socket()
server_socket.bind(('0.0.0.0', 8000))
server_socket.listen(0)

while True:
    # Accept a single connection and make a file-like object out of it
    with server_socket.accept()[0].makefile('wb') as connection:
        camera.start_recording(connection, format='mjpeg')
        while True:
            pass
    camera.stop_recording()

server_socket.close()

