import asyncio
import cv2
import time
import json
import websockets
from pprint import pprint

from pyquaternion import Quaternion
from picamera2 import Picamera2
import numpy as np

import board
import busio
from adafruit_bno08x import (
    BNO_REPORT_LINEAR_ACCELERATION,
    BNO_REPORT_GAME_ROTATION_VECTOR,
)
from adafruit_bno08x.i2c import BNO08X_I2C

class Camera:
    def __init__(self):
        focal_length = 3.04e-3  # meters
        vertical_resolution = 800
        horizontal_resolution = 600
        fx = fy = 1355 / 2
        cx, cy = 0.5 * horizontal_resolution, 0.5 * vertical_resolution
        self.camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        self.distortion_coefficients = np.array(
            [
                0.22128342058032355,
                -0.5663376863990286,
                -0.0001804474513748153,
                -0.001201953225667692,
                0.2602535953452802,
            ],
            dtype=np.float32,
        )


        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.charuco_board = cv2.aruco.CharucoBoard_create(
            squaresX=5, squaresY=7, squareLength=0.015, markerLength=0.011, dictionary=self.aruco_dict
        )

        self.tvec = None
        self.rvec = None

        # image = self.charuco_board.generateImage(outSize=(600, 500), marginSize=40, borderBits=1)
        # cv2.imwrite("charuco_board.png", image)

    def estimate_pose(self, image):
        corners, ids, rejected = cv2.aruco.detectMarkers(image, dictionary=self.aruco_dict)
        if len(corners) > 0:
            retval, corners, ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, image, self.charuco_board
            )
            success, self.rvec, self.tvec = cv2.aruco.estimatePoseCharucoBoard(
                corners,
                ids,
                self.charuco_board,
                self.camera_matrix,
                self.distortion_coefficients,
                rvec=self.rvec,
                tvec=self.tvec,
                useExtrinsicGuess=True if self.tvec is not None else False,
            )
            if success:
                # position and quaternion of the board in the camera frame
                angle = np.linalg.norm(self.rvec[:,0])
                axis = self.rvec[:,0] / angle
                quat_cam = Quaternion(axis=axis, radians=angle)
                pos_cam = self.tvec[:,0]

                # invert so we have the camera from the board's frame
                quat_cam = quat_cam.inverse
                pos_cam = -pos_cam

                # rotate 90 degrees around X axis so Y is up instead of Z
                quat_cam = Quaternion(axis=(1, 0, 0), degrees=-90) * quat_cam
                pos_cam = quat_cam.rotate(pos_cam)

                return True, pos_cam, quat_cam

        # pose could not be estimated
        return False, None, None


class IMU:
    def __init__(self):
        i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
        self.bno = BNO08X_I2C(i2c)

        self.bno.enable_feature(BNO_REPORT_LINEAR_ACCELERATION)
        self.bno.enable_feature(BNO_REPORT_GAME_ROTATION_VECTOR)

    def read(self):
        x, y, z, w = self.bno.game_quaternion
        quat = Quaternion(w, x, y, z)
        acceleration = np.array(self.bno.linear_acceleration)

        # transform from IMU to camera frame
        imu2cam = Quaternion(axis=[1, 0, 0], degrees=180)
        acceleration = imu2cam.rotate(acceleration)
        # rotate 90 degrees around X axis so Y is up
        y_up = Quaternion(axis=[1, 0, 0], degrees=-90)
        quat = y_up * quat * imu2cam
        # acceleration = imu2cam.rotate(acceleration)
        acceleration = quat.rotate(acceleration)

        return quat, acceleration


async def sense(websocket):
    # initialize PiCamera
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (800, 600), "format": "BGR888"})
    picam2.align_configuration(config)
    print(config["main"])
    picam2.configure(config)
    picam2.start()

    imu = IMU()
    camera = Camera()

    # state
    position = np.zeros(3)
    velocity = np.zeros(3)
    accel = np.zeros(3)
    quat = Quaternion()
    quat_correction = Quaternion()
    quat_cam = Quaternion()
    position_cam = np.zeros(3)

    t_prev = time.time()
    while True:
        # update quaternion with IMU readings
        quat, acceleration = imu.read()

        # update timestep
        now = time.time()
        dt = now - t_prev
        # dt = 1/200
        t_prev = now
        print(f"update rate: {1 / dt}")

        # braking force
        acceleration += 1 * -velocity

        # update velocity and position with IMU readings
        velocity += acceleration * dt
        position += velocity * dt

        # apply correction from camera pose estimation
        image = picam2.capture_array("main")
        success, new_position_cam, new_quat_cam = camera.estimate_pose(image)
        if success:
            position_cam = new_position_cam
            quat_cam = new_quat_cam

            # orientation correction
            gamma = 0.5
            if quat.norm == 0:
                quat_inverse = Quaternion()
            else:
                quat_inverse = quat.inverse
            dq = quat_cam * quat_inverse
            quat_correction = Quaternion(axis=dq.axis, angle=gamma * dq.angle) * quat_correction

            # position and velocity correction
            alpha = 0.5
            beta = 0.5
            correction = position_cam - position
            velocity += beta * correction
            position += alpha * correction

        message = {
            "quaternion": {
                "x": quat.x,
                "y": quat.y,
                "z": quat.z,
                "w": quat.w,
            },
            "quaternion_cam": {
                "x": quat_cam.x,
                "y": quat_cam.y,
                "z": quat_cam.z,
                "w": quat_cam.w,
            },
            "position": {
                "x": position[0],
                "y": position[1],
                "z": position[2],
            },
            "velocity": {
                "x": velocity[0],
                "y": velocity[1],
                "z": velocity[2],
            },
            "acceleration": {
                "x": acceleration[0],
                "y": acceleration[1],
                "z": acceleration[2],
            }
        }
        # pprint(message)

        await websocket.send(json.dumps(message))


async def main():
    async with websockets.serve(sense, host=None, port=8765):
        print("Server is ready")
        await asyncio.Future()


asyncio.run(main())
